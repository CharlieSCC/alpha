import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from alpha.config.config import *


def filter_df_industry(stock_id, df_industry,):
    df_industry["stock_id"] = df_industry["secu"].str[:6]
    df_industry = df_industry[df_industry["stock_id"].isin(stock_id)]
    df_industry = df_industry[df_industry["valid"] == 1]
    return df_industry[["stock_id", "report_date", "industry_code"]]


def filter_df_supply_chain(df_supply_chain):
    return df_supply_chain[["primary_code", "related_code", "relationship", "importance"]]


def graph_construction(df_industry, df_supply_chain, year_list, stock_id):
    df = pd.merge(left=df_industry, right=df_supply_chain, left_on='industry_code', right_on='primary_code', how="left")
    df = pd.merge(left=df, right=df_industry, left_on=['related_code', "report_date"], right_on=['industry_code', 'report_date'], how="inner")
    df["relationship"] = df["relationship"] * df["importance"]
    df = df[["report_date", "stock_id_x", "industry_code_x", "stock_id_y", "industry_code_y", "relationship"]]
    template = pd.DataFrame(index=stock_id, columns=stock_id)
    for y in year_list:
        res = df[df["report_date"] == "{}-12-31".format(str(y-1))]
        res = res[["stock_id_x", "stock_id_y", "relationship"]].set_index([["stock_id_x", "stock_id_y"]]).unstack()
        _, res = res.align(template, join="right",)
        res.to_hdf(os.path.join(DATA_PATH, "Ashare_data/graph_data/adjacent_matrix_{}.h5".format(str(y))), key="graph")


if __name__ == "__main__":
    stock_id = pd.read_hdf("/home/chencheng/Ashare_data/basic_data/stock_id.h5", key="stock_id")
    df_industry = pd.read_csv("/home/daily/fin_secu_primary_product.csv", sep=";", usecols=["secu", "report_date", "industry_code", "valid"])
    df_supply_chain = pd.read_csv("/home/daily/supply_chain_relation.csv", sep=";", usecols=["primary_code", "related_code", "relationship", "importance"])
    df_industry = filter_df_industry(stock_id, df_industry)
    df_supply_chain = filter_df_supply_chain(df_supply_chain)
    graph_construction(df_industry, df_supply_chain, [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022], stock_id)
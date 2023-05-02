# Data, Factor, Model and Evaluate

### Attention
    The LICENSE of the project is GPL3, which means No matter how you modify or use code, you need to open source it.
    
    这个项目的证书是GPL3，意味着无论以何种方式修改或者使用代码，都需要开源。    
    
    Do not close the source after modification. Failure to do so will result in legal liability。

    请不要修改后闭源。否则将带来法律上的责任。

### Data
    All data must be BOD. 
    For the universe, if it changes at 2014-12, then it is named as xxx-2015.
    FOr the graph, if it uses the 2014-12 finacial report, then it is named as xxx-2015. 
    
    --- Ashare_data --- 1day_data   --- pv.h5
                    --- basic_data  --- stock_id.h5
                                    --- trade_dates.h5
                                    --- zz800.h5
                                    --- zz1800.h5
                    --- factor_data --- alphas_101_alpha_001.h5
                                    --- alphas_101_alpha_002.h5
                    --- graph_data  --- adjacent_matrix_2015.h5
                                    --- adjacent_matrix_2016.h5      

### Factor
    Now we only include pv factor, which is published in alpha 101.

### Model
    Recurrent Model:
    - LSTM
    - GRU
    - Transformer
    
    Graph Model
    - LSTM_GCN
    - TGC
    - HATS
    - THGNN
    
### Evaluate
    - eval.py:            Basic Metrics for Finacial Prediction 
    - backtest.py:        Backtest for Model Signal
    - plot.py:            Plotting
    
Credits
-------

-  `Xiang Sheng <https://github.com/a-campbell>`
-  `Zhikang Xu <https://github.com/a-campbell>`

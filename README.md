# Data, Factor, Model and Evaluate

### Attention：

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
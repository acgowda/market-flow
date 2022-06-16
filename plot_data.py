import pandas as pd
import numpy as np

import plotly 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.express as px

import yfinance as yf
from ta.trend import MACD

def plot_ticker(df, target):
    dt_all = pd.date_range(start=df.index[0],end=df.index[-1])
    # retrieve the dates that ARE in the original datset
    dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df.index)]
    # define dates with missing values
    dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]

    # add moving averages to df
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA5_pct'] = df['Close'].rolling(window=5).mean().pct_change()

    # MACD
    macd = MACD(close=df['Close'],
                window_slow=26,
                window_fast=12, 
                window_sign=9)

    df['macd'] = macd.macd()
    df['macd_diff'] = macd.macd_diff()
    df['macd_signal'] = macd.macd_signal()


    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.01, 
                        row_heights=[0.7,0.1,0.2])


    df = df.iloc[33:]
    
    # Plot OHLC on 1st subplot (using the codes from before)
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                showlegend=False))
    # add moving average traces
    fig.add_trace(go.Scatter(x=df.index, 
                            y=df['MA5'], 
                            line=dict(color='blue', width=2), 
                            name='MA 5'))
    fig.add_trace(go.Scatter(x=df.index, 
                            y=df['MA20'], 
                            line=dict(color='orange', width=2), 
                            name='MA 20'))

    # Plot volume trace on 2nd row 
    colors = ['green' if row['Close'] - row['Open'] >= 0 
            else 'red' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, 
                        y=df['Volume'],
                        marker_color=colors
                        ), row=2, col=1)

    # Plot MACD trace on 3rd row
    colors = ['green' if val >= 0 else 'red' for val in df['macd_diff']]
    fig.add_trace(go.Bar(x=df.index, 
                        y=df['macd_diff'],
                        marker_color=colors,
                        name = 'Difference'
                        ), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index,
                            y=df['macd'],
                            line=dict(color='orange', width=2),
                            name = 'MACD'
                            ), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index,
                            y=df['macd_signal'],
                            line=dict(color='blue', width=1),
                            name = 'Signal Line'
                            ), row=3, col=1)

    fig.update_layout(title = f'{target} Price',
                      height=500, width=750, 
                      showlegend=False, 
                      xaxis_rangeslider_visible=False,
                      xaxis_rangebreaks=[dict(values=dt_breaks)])

    # update y-axis label
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig

def plot_returns(returns, predictions, target):
        
    yhat = np.argmax(predictions, 1)
    df = pd.DataFrame({'returns': returns})

    df['strategy'] = (np.array([1 if y == 1 else -1 for y in yhat])*df['returns'] + 1).cumprod() - 1
    # df['strategy'] =  ((yhat - 0.25) * df['returns'] + 1).cumprod() - 1
    df['returns'] =  (df['returns'] + 1).cumprod() - 1

    df['returns_minus_strategy'] = df['strategy'] - df['returns']
    
    #datetime stuff
    dt_all = pd.date_range(start=df.index[0],end=df.index[-1])
    # retrieve the dates that ARE in the original datset
    dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(df.index)]
    # define dates with missing values
    dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                    vertical_spacing=0.01)
    
    fig.add_trace(go.Scatter(x=df.index, 
                    y=df['strategy'], 
                    line=dict(color='blue', width=2), 
                    name='Model Strategy'),row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, 
                    y=df['returns'], 
                    line=dict(color='orange', width=2), 
                    name='Long Strategy'),row=1, col=1)

    fig.update_layout(title = f'Model Strategy vs. {target} Actual Returns',
                    height=500, width=750, 
                    showlegend=False, 
                    xaxis_rangeslider_visible=False,
                    xaxis_rangebreaks=[dict(values=dt_breaks)])

    fig.update_yaxes(title_text="Returns", row=1, col=1)
    fig.update_yaxes(title_text="Difference", row=2, col=1)

    return fig 

from flask import Flask, g, render_template, request

import pickle
import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

import json
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import yfinance as yf
from ta.trend import MACD

from get_data import get_preds_data

from flask_wtf import FlaskForm
from wtforms import BooleanField, StringField, SubmitField


app = Flask(__name__)
app.config['SECRET_KEY'] = 'jonathnisthebest'

class target_form(FlaskForm):
    target = StringField('target')
    submit = SubmitField('Submit')

class plotly_form(FlaskForm):
    MA5_bool = BooleanField('MA5')
    MA20_bool = BooleanField('MA20')
    volume_bool = BooleanField('Volume')
    MACD_bool = BooleanField('MACD')
    render = SubmitField('Render')


@app.route('/')
def main():
    return render_template('main.html')

@app.route('/about/')
def about():
    return render_template('about.html')

target = ''

@app.route('/test/', methods=['POST', 'GET'])
def test():

    tform = target_form()
    pform = plotly_form()

    global target

    #if tform.validate_on_submit(): # or pform.validate_on_submit():
    if pform.render.data:
        try:
            stock = yf.Ticker(target)
            df = stock.history(period='6mo')
            df.drop(['Dividends','Stock Splits'],axis = 1,inplace = True)
        except:
            return render_template('test.html', tform=tform, error=True)

        accuracy, pred = make_pred(target)

        fig = graph(df, MA5_bool=pform.MA5_bool.data, MA20_bool=pform.MA20_bool.data, 
                    volume_bool=pform.volume_bool.data, MACD_bool=pform.MACD_bool.data)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('test.html', tform=tform, pform=pform, target=target, 
                               graphJSON=graphJSON, accuracy=accuracy, pred=pred)

    if tform.submit.data:
        # assign the user's input to target
        target = tform.target.data

        try:
            stock = yf.Ticker(target)
            df = stock.history(period='6mo')
            df.drop(['Dividends','Stock Splits'],axis = 1,inplace = True)
        except:
            return render_template('test.html', tform=tform, error=True)

        accuracy, pred = make_pred(target)

        fig = graph(df)
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('test.html', tform=tform, pform=pform, target=target, 
                               graphJSON=graphJSON, accuracy=accuracy, pred=pred)

    return render_template('test.html', tform=tform, pform=pform)



def make_pred(target):
    # assign model to the pre-trained model.pkl
    model = pickle.load(open('model/model.pkl', 'rb'))

    past = get_preds_data(target)
    today = past.iloc[-1:].dropna(1)
    past.dropna(inplace=True)

    X = past.drop(columns=['close'])
    y = past['close']
    
    loss, accuracy = model.evaluate(X,y)
    accuracy = np.round(accuracy*100, 1)

    d = {0: 'down', 1: 'up'}
    pred = d[int(tf.math.argmax(model.predict(today), 1))]

    return accuracy, pred



def graph(df, MA5_bool=False, MA20_bool=False, volume_bool=False, MACD_bool=False):

    # removing all empty dates
    # build complete timeline from start date to end date
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

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.01, 
                        row_heights=[0.7,0.1,0.2])

    # Plot OHLC on 1st subplot (using the codes from before)
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                showlegend=False))
    # add moving average traces
    if MA5_bool:
        fig.add_trace(go.Scatter(x=df.index, 
                                y=df['MA5'], 
                                line=dict(color='blue', width=2), 
                                name='MA 5'))
    if MA20_bool:
        fig.add_trace(go.Scatter(x=df.index, 
                                y=df['MA20'], 
                                line=dict(color='orange', width=2), 
                                name='MA 20'))

    # Plot volume trace on 2nd row 
    if volume_bool:
        colors = ['green' if row['Close'] - row['Open'] >= 0 
                else 'red' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, 
                            y=df['Volume'],
                            marker_color=colors
                            ), row=2, col=1)

    # Plot MACD trace on 3rd row
    if MACD_bool:
        colors = ['green' if val >= 0 
                else 'red' for val in macd.macd_diff()]
        fig.add_trace(go.Bar(x=df.index, 
                            y=macd.macd_diff(),
                            marker_color=colors,
                            name = 'Difference'
                            ), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index,
                                y=macd.macd(),
                                line=dict(color='orange', width=2),
                                name = 'MACD'
                                ), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index,
                                y=macd.macd_signal(),
                                line=dict(color='blue', width=1),
                                name = 'Signal Line'
                                ), row=3, col=1)

        fig.update_layout(height=500, width=750, 
                        showlegend=False, 
                        xaxis_rangeslider_visible=False,
                        xaxis_rangebreaks=[dict(values=dt_breaks)])

    # update y-axis label
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if volume_bool:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    if MACD_bool:
        fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig
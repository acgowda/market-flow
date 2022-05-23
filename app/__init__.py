from flask import Flask, g, render_template, request

# import sklearn as sk
# import matplotlib.pyplot as plt
# import numpy as np
import pickle

# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure

# import io
# import base64

import pandas as pd
import json
import plotly
import plotly.express as px



app = Flask(__name__)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/test/', methods=['POST', 'GET'])
def test():
    if request.method == 'GET':
        return render_template('test.html')
    else:
        try:
            # assign the user's input to target
            target = request.form['target']

            # assign model to the pre-trained model.pkl
            # model = pickle.load(open('model/model.pkl', 'rb'))

            ##### perform prediction on target with the model





            #####

            ##### create the plotly figure here. a random example is shown.

            df = pd.DataFrame({
                'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 'Bananas'],
                'Amount': [4, 1, 2, 2, 4, 5],
                'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
            })
            fig = px.bar(df, x='Fruit', y='Amount', color='City', barmode='group')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            #####


            # once generated, return the prediction and figure here
            return render_template('test.html', target=target, graphJSON=graphJSON)

        except: # if user's entry is not valid
            return render_template('test.html', error=True)

# MarketFlow: Stock Market Predictions
Group members: Jonathn Chang, Ananda Gowda, Shannan Liu

## Overview

In this project, we designed and trained an Artificial Neural Network (ANN) that can be used to make predictions on a specified stock ticker. 

We extract live data and perform data cleaning, encoding, and normalization in the `get_data.py` file. 

We trained our model Tensorflow and Keras on Google Colab using data from Yahoo Finance on the weekly price changes of all the stocks in the S&P 500. The code can be viewed in `model/model_testing.ipynb` file.

The model can be accessed via our web app built using Flask. This app allows users to input a desired stock, and outputs a prediction for the next week, as well as an interactive plot that displays the model's performance on the most recent 6 months of data. Another plot displays the model's net profit over two years when using it to make trades and compares it to the actual returns. The details of this web app can be viewed in the `app` folder. To run the web app locally, clone the repo and run `flask run` while in the folder.

Finally, we deployed our app onto the web using Heroku. See [https://market-flow.herokuapp.com](https://market-flow.herokuapp.com)


## Data Processing
Our first step is to perform data processing, which is done in the `get_data.py` file. 

**describe get_data.py here; at least one code snippet and one figure**




## Model Training

Now, we build and train our model. We read in the .csv file created in the previous section. Loading in the DataFrame gives stock data with the following columns:
```python
['Date', 'open', 'high', 'low', 'close', 'volume', 'ma4', 'ma21', 'ma52',
 '^GSPC-close', '^GSPC-volume', '^GSPC-ma4', '^GSPC-ma21', '^GSPC-ma52',
 '^VIX-close', '^VIX-ma4', '^VIX-ma21', '^VIX-ma52', '^IXIC-close',
 '^IXIC-volume', '^IXIC-ma4', '^IXIC-ma21', '^IXIC-ma52', '^DJI-close',
 '^DJI-volume', '^DJI-ma4', '^DJI-ma21', '^DJI-ma52', '^HSI-close',
 '^HSI-volume', '^HSI-ma4', '^HSI-ma21', '^HSI-ma52', '^FTSE-close',
 '^FTSE-volume', '^FTSE-ma4', '^FTSE-ma21', '^FTSE-ma52', '^FCHI-close',
 '^FCHI-volume', '^FCHI-ma4', '^FCHI-ma21', '^FCHI-ma52', 'GC=F-close',
 'GC=F-volume', 'GC=F-ma4', 'GC=F-ma21', 'GC=F-ma52', 'CL=F-close',
 'CL=F-volume', 'CL=F-ma4', 'CL=F-ma21', 'CL=F-ma52', 'month_1',
 'month_10', 'month_11', 'month_12', 'month_2', 'month_3', 'month_4',
 'month_5', 'month_6', 'month_7', 'month_8', 'month_9']
```

The 'close' column is separated as the target variable, and the rest of the columns are used as predictor variables. We create our ANN with a Keras Sequential model:
```python
model = tf.keras.models.Sequential([
    layers.Dense(128, input_shape=(61,), activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(2)
])
```
and train it over 30 epochs. The training history is shown below.
![training_history.png](/images/training_history.png)

We save the model to be used in the web app using `model.save()`.



## Flask App
Finally, we deployed our code onto a web app via Flask. The `app` folder contains the `__init__.py` file which initializes the web app, a `templates` folder containing html files that are rendered by Flask, and a `static` folder containing styling and a favicon for the web app.

First, we create the Flask app:
```python
from flask import Flask
app = Flask(__name__)
```

We use regular routing to describe the behavior of the web app when accessing various pages:
```python
@app.route('/')
def main():
    return render_template('main.html')

@app.route('/about/')
def about():
    return render_template('about.html')
```

These render the `main.html` and `about.html` files found in the `templates` folder. In particular, our html uses Jinja, which allows for html files to extend a 'base' file, as well as convenient parsing of Python variables. You can read more about Jinja [here](https://jinja.palletsprojects.com/en/3.1.x/).

Our implementation of the `test()` function performs a POST request to retrieve the user's desired stock and makes a prediction on this stock, as well as evaluates the model on the past 6 months. Then, it uses the `plot_ticker()` and `plot_returns()` functions in the `plot_data.py` file which build and return prediction and returns plots. We use JSON encoding to process the figure and render it with Flask along with the `test.html` file.

The following code is needed to encode each figure:
```python
tickerJSON = json.dumps(plot_ticker(df, target), cls=plotly.utils.PlotlyJSONEncoder)
```

and to insert them into the html:
```html
<div id='chart' class='chart'></div>
<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
<script type='text/javascript'>
  var graphs = {{tickerJSON | safe}};
  Plotly.plot('chart',graphs,{});
</script>
```

Here is a screenshot of the `test.html` page after a user makes a submission:
![test_page.jpeg](/images/test_page.jpeg)


## References:
https://python.plainenglish.io/a-simple-guide-to-plotly-for-plotting-financial-chart-54986c996682
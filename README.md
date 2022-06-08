# MarketFlow: Stock Market Predictions
Group members: Jonathn Chang, Ananda Gowda, Shannan Liu

## Overview

In this project, we designed and trained an Artificial Neural Network (ANN) that can be used to make predictions on a specified stock ticker. 

We extract live data and perform data cleaning, encoding, and normalization in the `get_data.py` file. 

We trained our model Tensorflow and Keras on Google Colab using data from Yahoo Finance on the weekly price changes of all the stocks in the S&P 500. The code can be viewed in `model/model_testing.ipynb` file.

The model can be accessed via our web app built using Flask. This app allows users to input a desired stock, and outputs a prediction for the next week, as well as an interactive plot that displays the model's performance on the most recent 6 months of data. The details of this web app can be viewed in the `app` folder. To run the web app locally, clone the repo and run `flask run` while in the folder.

Finally, we deployed our app onto the web using Heroku. See [https://market-flow.herokuapp.com](https://market-flow.herokuapp.com)


## Data Processing
Our first step is to perform data processing, which is done in the `get_data.py` file. 

# describe get_data.py here; at least one code snippet and one figure




## Model Training

Now, we build and train our model. 





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
![test_page.jpeg](/images/HW-0_output.png)








## References:
https://python.plainenglish.io/a-simple-guide-to-plotly-for-plotting-financial-chart-54986c996682
# MarketFlow: Stock Market Predictions
Group members: Jonathn Chang, Ananda Gowda, Shannan Liu

In this project, we designed and trained an Artificial Neural Network (ANN) that can be used to make predictions on a specified stock ticker. 

We extract live data from  and perform data cleaning, encoding, and normalization in the `get_data.py` file. 

We trained our model Tensorflow and Keras on Google Colab using data from Yahoo Finance on the weekly price changes of all the stocks in the S&P 500. The code can be viewed in `model\model_testing.ipynb` file.

The model can be accessed via our web app built using Flask. This app allows users to input a desired stock, and outputs a prediction for the next week, as well as an interactive plot that displays the model's performance on the most recent 6 months of data. The details of this web app can be viewed in the `app` folder. To run the web app locally, clone the repo and run `flask run` while in the folder.

Finally, we deployed our app onto the web using Heroku. See [https://market-flow.herokuapp.com](https://market-flow.herokuapp.com)

References:
https://python.plainenglish.io/a-simple-guide-to-plotly-for-plotting-financial-chart-54986c996682

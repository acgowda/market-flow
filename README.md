# marketflow: Stock Market Predictions
Group members: Jonathn Chang, Ananda Gowda, Shannan Liu

In this project, we design a DNN model that can be used to make stock predictions on a specified target. 

We extract data from the yfinance API and perform data cleaning, encoding, and normalization in the `get_data.py` file. 

We train our model with a feed-forward DNN using Keras and Tensorflow. This model is built and trained with Colab, in the `model_testing.ipynb` file.

The model is presented in a web app built using Flask. This app allows users to input a desired target, and outputs a prediction for the next day, as well as an interactive plot that displays the model's performacne on the most recent 60 days of data. The details of this web app can be viewed in the `app` folder. To run the web app locally, clone the repo and run `flask run` while in the folder.

Finally, we deployed our app onto the web using Heroku. See [https://market-flow.herokuapp.com](https://market-flow.herokuapp.com)

References:
https://python.plainenglish.io/a-simple-guide-to-plotly-for-plotting-financial-chart-54986c996682

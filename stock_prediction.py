# Stock Price Prediction using Google Stock Data

# The daily mean price of Google Stock is predicted using dates as training set and mean price as target

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import metrics
    import datetime as dt
    from sklearn.linear_model import LinearRegression
    from math import sqrt
    import pickle

    f = open('final_model.sav')
    model = pickle.load(open('final_model.sav', 'rb'))

except:

    df = pd.read_csv('data/Google_Stock_Price_Train.csv')

    # Some Pre-Processing:

    df.drop(['High', 'Low', 'Volume'], axis=1, inplace=True)

    df['Open'] = df["Open"].astype(float)
    df['Close'] = df['Close'].str.replace(',', '').astype(float)

    df['mean_price'] = (df['Open']+df['Close'])/2

    # Since Regression Models do not accept string as training data type, we need to convert the dates from str to pandas datetime
    df['Date'] = pd.to_datetime(df['Date'])
    # This will convert the datetime value to float
    df['Date'] = df['Date'].map(dt.datetime.toordinal)

    x_train = df['Date'].values
    y_train = df['mean_price'].values
    x_train = x_train.reshape(-1, 1)

    # Building the Machine Learning Model using LinearRegression:

    model = LinearRegression().fit(x_train, y_train)
    filename = "final_model.sav"

    pickle.dump(model, open(filename, 'wb'))

# Testing on Actual Input Data:
# The input data will be in string format, but our model expects a 2D numpy array:


def make_prediction(data):
    # input_str = '04/27/2020'
    inputx = pd.to_datetime(data)
    inputx = dt.datetime.toordinal(inputx)
    inputx = np.asarray(inputx).reshape(1, -1)

    return(model.predict(inputx)[0])


# for test
# data = '04/27/2020'
# result = make_prediction(data)

# print('The stock price of Google on {} as predicted by the model is ${}'.format(data, result))

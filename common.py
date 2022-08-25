import os
import pandas as pd
from termcolor import colored
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from keras.models import model_from_json
import ipdb

EQ = r'wigtech.txt'
# DB_PATH= r'temp/pl/wse indices/'
DB_PATH = r'Dropbox/projects/forecast/data/'
SERIALIZED_MODEL = 'arima.model'
COL_DATE = '<DATE>'
COL_OPEN = '<OPEN>'
COL_CLOSE = '<CLOSE>'
COL_LOW = '<LOW>'
COL_HIGH = '<HIGH>'

# Get all files in a given directory and its directories
def get_data(eq):
    fp = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), eq), 'r')
    df = pd.read_csv(fp)
    # Convert data in string to date
    df[COL_DATE] = df[COL_DATE].apply(pd.to_datetime, format="%Y%m%d")
    # Restrict the data which is to be plotted to several months
    df = df[df[COL_DATE] >= '2017-01-01']
    # Make a date an index of the frame
    df.set_index(COL_DATE, inplace=True)
    df_w = df.resample('1w').agg({COL_OPEN: 'first', COL_HIGH: 'max',
                                 COL_LOW: 'min', COL_CLOSE: 'last'})
    df_w['change'] = df_w[COL_CLOSE].pct_change()*100
    return(df_w['change'][1:].to_numpy())

def plot_forecast(forecasts, y_test):
    fig, axes = plt.subplots(1, 1, figsize=(5, 4))
    window = [i for i in range(26)]
    axes.plot(forecasts[window], color='green', marker='o',
              label='Predicted')
    axes.plot(y_test[window], color='red', label='Actual', marker='x')

    axes.legend()
    plt.xlabel("Week")
    plt.ylabel("Change [%]")
    plt.savefig("prediction.png")
    plt.show()

    return

def save_model(model_file, model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def load_model(model_file):
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return(loaded_model)


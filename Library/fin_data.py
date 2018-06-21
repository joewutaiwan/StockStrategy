import csv
import numpy as np
import pandas as pd

def read_csv(name):
    with open(name, 'rb') as csvfile:
        rows =  csv.reader(csvfile)
        return np.asarray(list(rows))
def read_y_csv(name):
    with open(name, 'rb') as csvfile:
        rows =  csv.reader(csvfile)
        tmp = []
        for row in rows:
            tmp.append(row[0])
        return np.asarray(list(tmp))

def load_data(train_test_ratio = 0.5):
    
    xdata = read_csv("xdata.csv")
    ydata = read_y_csv("ydata.csv")

    train_len = int(train_test_ratio * len(xdata))

    (train_x, test_x) = (xdata[:train_len], xdata[train_len:])
    (train_y, test_y) = (ydata[:train_len], ydata[train_len:])

    return (train_x, train_y), (test_x, test_y)

def df_to_nparray(df):
    arr = []
    for index, row in df.iterrows():
        c = np.asarray(row)
        arr.append(c)
    return np.asarray(arr)

def preprocess_xdata(name):
    df = pd.read_csv(name, encoding = 'utf8')
    #for key in df:
    #    df[key] = (df[key] - df[key].mean()) / df[key].std()
    return df_to_nparray(df)

def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-0.3 * z))
    return s

def preprocess_ydata(name):
    df = pd.read_csv(name, encoding = 'utf8')
    df['EPS'] = sigmoid(df['EPS'])*10
    return df_to_nparray(df)

def load_data_preprocess(train_test_ratio = 0.5):
    xdata = preprocess_xdata("xdata.csv")
    ydata = preprocess_ydata("ydata.csv")

    train_len = int(train_test_ratio * len(xdata))

    (train_x, test_x) = (xdata[:train_len], xdata[train_len:])
    (train_y, test_y) = (ydata[:train_len], ydata[train_len:])

    return (train_x, train_y), (test_x, test_y)

def parse_csv(name):
    df = pd.read_csv(name, encoding = 'utf8')
    return df_to_nparray(df)

def load_company_data():
    xdata = parse_csv("company_xdata.csv")
    ydata = parse_csv("company_ydata.csv")
    zdata = parse_csv("company_zdata.csv")
    return (xdata, ydata, zdata)

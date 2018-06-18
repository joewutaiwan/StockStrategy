import csv
import numpy as np

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

load_data(0.5)
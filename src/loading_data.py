"""
    author: Joanna Sokołowska
             https://github.com/jsokolowska
"""

import pandas as pd
import numpy as np


def load_data(which="both"):
    df = None
    if which == "mat":
        df = pd.read_csv("../student-alcohol-consumption/student-mat.csv")
    elif which == "por":
        df = pd.read_csv("../student-alcohol-consumption/student-por.csv")
    elif which == "both":
        df1 = pd.read_csv("../student-alcohol-consumption/student-mat.csv")
        df2 = pd.read_csv("../student-alcohol-consumption/student-por.csv")
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        print("Option not recognized")
    return df


def load_classic_dataset():
    outlook = 'sunny,sunny,overcast,rain,rain,rain,overcast,sunny,sunny,rain,sunny,overcast,overcast,rain'.split(',')
    temp = 'hot,hot,hot,mild,cool,cool,cool,mild,cool,mild,mild,mild,hot,mild'.split(',')
    humidity = 'high,high,high,high,normal,normal,normal,high,normal,normal,normal,high,normal,high'.split(",")
    windy = 'false,true,false,false,false,true,true,false,false,false,true,true,false,true'.split(',')
    result = 'N,N,P,P,P,N,P,N,P,P,P,P,P,N'.split(",")
    dataset = {'outlook': outlook, 'temperature': temp, 'humidity': humidity, 'windy': windy, 'result': result}
    return pd.DataFrame(dataset)


def divide(df: pd.DataFrame, training_set_percent=0.7):
    if training_set_percent > 0.9 or training_set_percent < 0.5:
        print("Unreasonably small training set. Quiting....")
        raise ValueError
    msk = np.random.rand(len(df)) < training_set_percent
    training = df[msk]
    test = df[~msk]
    return training, test

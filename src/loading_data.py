"""
    authors:
    Joanna Sokolowska - https://github.com/jsokolowska
    Rafal Uzarowicz - https://github.com/RafalUzarowicz

todo:
 - add some protection to data loading
"""

import pandas as pd
import numpy as np


def divide(df: pd.DataFrame, training_set_percent=0.7):
    if training_set_percent > 0.9 or training_set_percent < 0.5:
        raise ValueError("Unreasonably small training set. Quiting....")
    msk = np.random.rand(len(df)) < training_set_percent
    training = df[msk]
    test = df[~msk]
    return training, test


class ID3DatasetLoader:
    def __init__(self):
        self.dataset = None

    def get_dataset(self) -> pd.DataFrame:
        return self.dataset

    def check_if_dataset_is_good(self) -> None:
        if type(self.dataset) is not pd.DataFrame or self.dataset is None or self.dataset.isnull().values.any() or len(self.dataset.columns) < 2:
            self.dataset = None

    def load_csv_dataset(self, filepath: str) -> None:
        self.dataset = pd.read_csv(filepath)
        self.check_if_dataset_is_good()

    def load_json_dataset(self, filepath: str) -> None:
        self.dataset = pd.read_json(filepath)
        self.check_if_dataset_is_good()

    def load_excel_dataset(self, filepath: str) -> None:
        self.dataset = pd.read_excel(filepath)
        self.check_if_dataset_is_good()

    def load_sql_table_dataset(self, filepath: str) -> None:
        self.dataset = pd.read_sql_table(filepath)
        self.check_if_dataset_is_good()

    def load_table_dataset(self, filepath: str) -> None:
        self.dataset = pd.read_table(filepath)
        self.check_if_dataset_is_good()

    def load_example_dataset(self, which: str="both", which_classification_attribute: str = "Walc"):
        if which == "mat":
            df = pd.read_csv("../example_datasets/student-mat.csv")
        elif which == "por":
            df = pd.read_csv("../example_datasets/student-por.csv")
        elif which == "both":
            df1 = pd.read_csv("../example_datasets/student-mat.csv")
            df2 = pd.read_csv("../example_datasets/student-por.csv")
            df = pd.concat([df1, df2], ignore_index=True)
        else:
            raise ValueError("Option not recognized.")
        if df is not None:
            df.drop(columns=["G1", "G2", "G3"], axis=1)
            if which_classification_attribute == "Walc":
                df.drop(columns=["Dalc"], axis=1)
            elif which_classification_attribute == "Dalc":
                df.drop(columns=["Walc"], axis=1)
            elif which_classification_attribute == "both":
                pass
            else:
                raise ValueError("Attribute not recognized.")
        self.dataset = df
        self.check_if_dataset_is_good()

    def load_classic_dataset(self):
        outlook = 'sunny,sunny,overcast,rain,rain,rain,overcast,sunny,sunny,rain,sunny,overcast,overcast,rain'.split(
            ',')
        temp = 'hot,hot,hot,mild,cool,cool,cool,mild,cool,mild,mild,mild,hot,mild'.split(',')
        humidity = 'high,high,high,high,normal,normal,normal,high,normal,normal,normal,high,normal,high'.split(",")
        windy = 'false,true,false,false,false,true,true,false,false,false,true,true,false,true'.split(',')
        result = 'N,N,P,P,P,N,P,N,P,P,P,P,P,N'.split(",")
        dataset = {'outlook': outlook, 'temperature': temp, 'humidity': humidity, 'windy': windy, 'result': result}
        df = pd.DataFrame(dataset)
        self.dataset = df

    def load_classic_with_numbers_dataset(self):
        outlook = 'sunny,sunny,overcast,rain,rain,rain,overcast,sunny,sunny,rain,sunny,overcast,overcast,rain'.split(
            ',')
        temp = 'hot,hot,hot,mild,cool,cool,cool,mild,cool,mild,mild,mild,hot,mild'.split(',')
        humidity = 'high,high,high,high,normal,normal,normal,high,normal,normal,normal,high,normal,high'.split(",")
        windy = 'false,true,false,false,false,true,true,false,false,false,true,true,false,true'.split(',')
        number_int = '1,8,2,3,5,1,2,7,6,2,4,7,9,8'.split(',')
        number_float = '1.0,8.0,2.0,3.0,5.0,1.0,2.0,7.0,6.0,2.0,4.0,7.0,9.0,8.0'.split(',')
        result = 'N,N,P,P,P,N,P,N,P,P,P,P,P,N'.split(",")
        dataset = {'outlook': outlook, 'temperature': temp, 'humidity': humidity, 'windy': windy,
                   'number_int': number_int, 'number_float': number_float, 'result': result}
        df = pd.DataFrame(dataset)
        self.dataset = df

    def load_classic_with_triple_classification_dataset(self):
        outlook = 'sunny,sunny,overcast,rain,rain,rain,overcast,sunny,sunny,rain,sunny,overcast,overcast,rain'.split(
            ',')
        temp = 'hot,hot,hot,mild,cool,cool,cool,mild,cool,mild,mild,mild,hot,mild'.split(',')
        humidity = 'high,high,high,high,normal,normal,normal,high,normal,normal,normal,high,normal,high'.split(",")
        windy = 'false,true,false,false,false,true,true,false,false,false,true,true,false,true'.split(',')
        result = 'N,N,T,P,P,N,P,T,P,P,T,T,P,N'.split(",")
        dataset = {'outlook': outlook, 'temperature': temp, 'humidity': humidity, 'windy': windy, 'result': result}
        df = pd.DataFrame(dataset)
        self.dataset = df

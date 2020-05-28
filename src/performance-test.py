"""
    author: Joanna Sokołowska - https://github.com/jsokolowska

 """

import numpy as np
import pandas as pd
from src.loading_data import ID3DatasetLoader
from src.id3 import ID3


def cross_validate(division_num: int = 10):
    def slice_df(dataframe: pd.DataFrame) -> []:
        df_list = []
        slice_size = len(dataframe) // division_num
        for k in range(division_num-1):
            frac = slice_size/len(dataframe)
            split_set = dataframe.sample(frac=frac)
            df_list.append(split_set)
            dataframe = dataframe.drop(split_set.index)
        df_list.append(dataframe)
        return df_list

    print("------ Test 1 - Algorithm effectiveness ------")
    loader = ID3DatasetLoader()
    loader.load_example_dataset("both", "both")
    df = loader.get_dataset()
    daily_df = df.drop("Walc", axis=1)
    weekend_df = df.drop("Dalc", axis=1)

    datasets = [(daily_df, "Dalc"), (weekend_df, "Walc")]
    for dataset in datasets:
        print("--- For target attribute: " + dataset[1] + " ---")
        slices = slice_df(dataset[0])
        target = dataset[1]
        accuracy = []

        for i in range(division_num):
            test = slices[i]
            train = dataset[0].drop(test.index)
            id3 = ID3(train, target, use_ranges_for_numeric=False, use_window=False)
            test["pred"] = id3.predict(test)
            slice_accuracy = len(test[test["pred"] == test[target]])/len(test)
            accuracy.append(slice_accuracy * 100)

        average = sum(accuracy)/len(accuracy)
        print("Average accuracy: " + str(average) + "%")
        print("For individual slices: " + str(accuracy))


cross_validate(10)

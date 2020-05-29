"""
    author: Joanna Sokołowska - https://github.com/jsokolowska

 """

import numpy as np
import pandas as pd
import random
from src.loading_data import ID3DatasetLoader
from src.id3 import ID3


def cross_validate(division_num: int = 10):
    def slice_df(dataframe: pd.DataFrame) -> []:
        df_list = []
        slice_size = len(dataframe) // division_num
        for k in range(division_num - 1):
            frac = slice_size / len(dataframe)
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
        print("For target attribute: " + dataset[1])
        slices = slice_df(dataset[0])
        target = dataset[1]
        accuracy = []

        for i in range(division_num):
            test = slices[i]
            train = dataset[0].drop(test.index)
            id3 = ID3(train, target, use_ranges_for_numeric=False, use_window=False)
            test["pred"] = id3.predict(test)
            slice_accuracy = len(test[test["pred"] == test[target]]) / len(test)
            accuracy.append(slice_accuracy * 100)

        average = sum(accuracy) / len(accuracy)
        print("Average accuracy: " + str(average) + "%")
        print("For individual slices: " + str(accuracy))


def corrupt(noise_lvl: float, dataset: pd.DataFrame, target_att) -> pd.DataFrame:
    all_att_values = {}
    for column in dataset.columns:
        if column != target_att:
            value_list = []
            for value in dataset[column].unique():
                value_list.append(value)
            all_att_values[column] = value_list

    for index, row in dataset.iterrows():
        for key in all_att_values.keys():
            rand_nr = np.random.uniform()
            if rand_nr < noise_lvl:
                all_vals = all_att_values[key][:]
                all_vals.remove(row[key])
                if len(all_vals):
                    new_val = random.choice(all_vals)
                    row[key] = new_val
    return dataset


def noise_level_test(*, start: float = 0.05, stop: float = 0.95, step: float = 0.10, repeats=10):
    print("------ Test 2 - Noise level test ------")
    print("Target attribute: Dalc")
    target_att = "Dalc"
    loader = ID3DatasetLoader()
    loader.load_example_dataset("both")
    dataset = loader.get_dataset()
    curr_lvl = start
    results = {}
    while curr_lvl < stop:
        ctr_cts = []
        ctr_ts = []
        tr_cts = []
        for i in range(repeats):
            msk = np.random.rand(len(dataset)) < 0.15
            test = dataset[msk].copy()
            train = dataset[~msk].copy()
            ctrain = corrupt(curr_lvl, train, target_att)
            ctest = corrupt(curr_lvl, test, target_att)

            id3 = ID3(train, target_att)
            cid3 = ID3(ctrain, target_att)
            test["corr_pred"] = cid3.predict(test)
            ctest["pred"] = id3.predict(ctest)
            ctest["corr_pred"] = cid3.predict(ctest)

            test_len = len(test)
            clean_test_corr_train = 100 * len(test[test["corr_pred"] == test[target_att]])/test_len
            clean_train_corr_test = 100 * len(ctest[ctest["pred"] == ctest[target_att]])/test_len
            corr_train_corr_test = 100 * len(ctest[ctest["corr_pred"] == ctest[target_att]])/test_len
            ctr_cts.append(corr_train_corr_test)
            ctr_ts.append(clean_test_corr_train)
            tr_cts.append(clean_train_corr_test)

        res = (sum(ctr_ts)/len(ctr_ts), sum(ctr_cts)/len(ctr_cts), sum(tr_cts)/len(tr_cts))
        results[str(curr_lvl)] = res
        print("Noise level: " + str(curr_lvl))
        print("\t Corrupt train, clean test: " + str(res[0]))
        print("\t Corrupt train, corrupt test: " + str(res[1]))
        print("\t Clean train, corrupt test: " + str(res[2]))
        curr_lvl += step


noise_level_test(start=0.30, stop=0.35, repeats=3)

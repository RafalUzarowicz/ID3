"""
    author: Joanna Sokołowska - https://github.com/jsokolowska
"""

import numpy as np
import pandas as pd
import random
from src.loading_data import ID3DatasetLoader
from src.id3 import ID3
import matplotlib.pyplot as plt


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

    with open("cross_validate_test.txt", 'w') as file:
        loader = ID3DatasetLoader()
        loader.load_example_dataset("both")
        df = loader.get_dataset()
        daily_df = df.drop("Walc", axis=1)
        weekend_df = df.drop("Dalc", axis=1)

        datasets = [(daily_df, "Dalc"), (weekend_df, "Walc")]
        for dataset in datasets:
            file.write("For target attribute: " + dataset[1])
            slices = slice_df(dataset[0])
            target = dataset[1]
            accuracy = []
            pivot_vals = []
            mixed_vals = []
            ranges_vals = []

            for i in range(division_num):
                test = slices[i]
                train = dataset[0].drop(test.index)
                target = dataset[1]
                # msk = np.random.rand(len(df)) < 0.15
                # test = df[msk].copy()
                # train = df[~msk].copy()
                pivot = ID3(train, target, use_ranges_for_numeric=False)
                mixed = ID3(train, target, use_ranges_for_numeric=False,
                            numeric_att=["age", "absences", "G1", "G2", "G3"])
                ranges = ID3(train, target, use_ranges_for_numeric=True)

                test["pred"] = pivot.predict(test)
                pivot_acc = len(test[test[target] == test["pred"]]) / len(test)
                pivot_vals.append(pivot_acc * 100)

                test["pred"] = mixed.predict(test)
                mixed_acc = len(test[test[target] == test["pred"]]) / len(test)
                mixed_vals.append(mixed_acc * 100)

                test["pred"] = ranges.predict(test)
                ranges_acc = len(test[test[target] == test["pred"]]) / len(test)
                ranges_vals.append(ranges_acc * 100)

        file.write("Mixed, ranges, pivot\n")
        for numeric, name in [mixed_vals, ranges_vals, pivot_vals], ["Mixed: ", "Ranges: ", "Pivot: "]:
            avg = sum(numeric) / len(numeric)
            line = name + str(avg) + " " + str(numeric)
            file.write(line + "\n")


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
                    dataset.loc[index, key] = new_val
    return dataset

def noise_level_test(*, start: float = 0.05, stop: float = 0.95, step: float = 0.10, repeats=10):
    with open('noise_test.txt', 'w')as file:
        file.write("Noise lvl test for Dalc\n corrupt train, clean test; corrupt train, corrupt test; clean train, "
                   "corrupt test; \n")

        target_att = "Dalc"
        loader = ID3DatasetLoader()
        loader.load_example_dataset("both")
        dataset = loader.get_dataset()
        curr_lvl = start
        while curr_lvl < stop:
            ctr_cts = []
            ctr_ts = []
            tr_cts = []
            for i in range(repeats):
                print("N: " + str(curr_lvl) + " R: " + str(i))
                msk = np.random.rand(len(dataset)) < 0.15
                test = dataset[msk].copy()
                train = dataset[~msk].copy()
                ctrain = corrupt(curr_lvl, train.copy(), target_att)
                ctest = corrupt(curr_lvl, test.copy(), target_att)

                id3 = ID3(train, target_att)
                cid3 = ID3(ctrain, target_att)
                test["corr_pred"] = cid3.predict(test)
                ctest["pred"] = id3.predict(ctest)
                ctest["corr_pred"] = cid3.predict(ctest)

                test_len = len(test)
                clean_test_corr_train = 100 * len(test[test["corr_pred"] == test[target_att]]) / test_len
                clean_train_corr_test = 100 * len(ctest[ctest["pred"] == ctest[target_att]]) / test_len
                corr_train_corr_test = 100 * len(ctest[ctest["corr_pred"] == ctest[target_att]]) / test_len
                ctr_cts.append(corr_train_corr_test)
                ctr_ts.append(clean_test_corr_train)
                tr_cts.append(clean_train_corr_test)

            res = [sum(ctr_ts) / len(ctr_ts), sum(ctr_cts) / len(ctr_cts), sum(tr_cts) / len(tr_cts)]
            for i in range(len(res)):
                res[i] = '{:.4f}'.format(res[i])
            line = ";".join(res)
            file.write(line + "\n")
            curr_lvl += step


def test_train_size(*, start=0.4, stop=1, step=0.1, repeats=10):
    with open("test_train_size.txt", "w") as file:
        file.write("Classification attribute - Dalc\n")
        target_att = "Dalc"
        curr_size = start
        loader = ID3DatasetLoader()
        loader.load_example_dataset("both")
        df = loader.get_dataset()
        results = []
        train_results = []
        while curr_size <= stop:
            if curr_size + step > stop:
                curr_size = stop - step / 2
            accuracy = []
            train_accuracy = []
            for i in range(repeats):
                msk = np.random.rand(len(df)) < curr_size
                train = df[msk].copy()
                test = df[~msk].copy()
                id3 = ID3(train, target_att)
                test["pred"] = id3.predict(test)
                train["pred"] = id3.predict(train)
                res = 100 * len(test[test["pred"] == test[target_att]]) / len(test)
                accuracy.append(res)
                train_res = 100 * len(train[train["pred"] == train[target_att]]) / len(train)
                train_accuracy.append(train_res)
            percent = sum(accuracy) / len(accuracy)
            line = '{:.4f}'.format(percent)
            results.append(percent)
            train_results.append(sum(train_accuracy) / len(train_accuracy))
            file.write(str(curr_size) + " : " + line)
            curr_size += step


print("Performing cross validation for all numeric methods...")
cross_validate()
print("Results have been saved to cross_validate_test.txt\n")
print("Performing training set size test...")
test_train_size()
print("Results have been saved to test_train_size.txt\n")
print("Performing noise level test...\n [Warining: it will take a lot of time, "
      "for test results please do not kill the process until this test has been finished]")
noise_level_test()
print("Results have been saved to noise_text.txt\n")


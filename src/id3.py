"""
    authors:
    Joanna Sokolowska - https://github.com/jsokolowska
    Rafal Uzarowicz - https://github.com/RafalUzarowicz

todo:
 - wczytywanie danych z dowolnego csv
"""

from numpy import log2
import pandas as pd
import numpy as np
from src.loading_data import load_classic_dataset, load_example_dataset, divide


class ID3:
    def __init__(self, dataset: pd.DataFrame, classification_attribute: str):
        self.dataset = dataset
        print(dataset)
        self.class_attr = classification_attribute
        self.tree = None
        if self.class_attr not in self.dataset.columns:
            raise ValueError("Wrong classification attribute name!")
        self.classification_values = list(set(self.dataset[self.class_attr].to_list()))
        print(self.classification_values)
        self.is_continuous = {}
        self.prepare_tree()
        print(self.tree)

    def prepare_data(self):
        def is_number(string):
            try:
                float(string)
                return True
            except ValueError:
                return False
        self.is_continuous = {}
        for val in list(self.dataset.columns):
            if all(is_number(n) for n in self.dataset[val]):
                self.is_continuous[val] = True
                self.dataset[val] = pd.to_numeric(self.dataset[val], errors='coerce')
            else:
                self.is_continuous[val] = False
        print(self.is_continuous)

    def entropy(self, data: pd.DataFrame) -> float:
        entro = 0.0
        for val in self.classification_values:
            if len(data[data[self.class_attr] == val]) > 0:
                fraction = len(data[data[self.class_attr] == val]) / len(data)
                entro += - fraction * log2(fraction)
        return entro

    def average(self, data: pd.DataFrame, attribute):
        if not self.is_continuous[attribute]:
            raise ValueError("Wrong type of attribute for average function.")
        values_list = data[attribute].to_list()
        if not len(values_list) > 0:
            raise ValueError("Empty dataset.")
        val_sum = 0.0
        for val in values_list:
            val_sum += val
        return val_sum / len(values_list)

    def gain(self, data: pd.DataFrame, attribute: str) -> float:
        if attribute not in self.dataset.columns or attribute == self.class_attr:
            raise ValueError("Wrong attribute for information gain!")
        info_gain = self.entropy(data)
        if self.is_continuous.get(attribute, None) is False:
            for val in data[attribute].unique():
                info_gain -= self.entropy(data[data[attribute] == val]) * len(data[data[attribute] == val]) / len(data)
        elif self.is_continuous.get(attribute, None) is True:
            pivot = self.average(data, attribute)
            info_gain -= self.entropy(data[data[attribute] >= pivot]) * len(data[data[attribute] >= pivot]) / len(data)
            info_gain -= self.entropy(data[data[attribute] < pivot]) * len(data[data[attribute] < pivot]) / len(data)
        return info_gain

    def find_maximum_gain(self, data: pd.DataFrame):
        best_attr = ""
        best_gain = -1.0
        for key in data.columns:
            if key is not self.class_attr:
                current_gain = self.gain(data, key)
                if current_gain >= best_gain:
                    best_attr = key
                    best_gain = current_gain
        return best_attr

    def prepare_tree(self):
        all_attributes = list(self.dataset.columns)
        all_attributes.remove(self.class_attr)
        temp_data = self.dataset

        def build_tree(data: pd.DataFrame):

            def end_conditions(end_values:list, key):
                first_val = end_values[0]

                if all(n == first_val for n in end_values):
                    return first_val
                elif len(all_attributes) == 0:
                    values_counter = {k: 0 for k in self.classification_values}
                    for value in end_values:
                        values_counter[value] += 1
                    best_val = self.classification_values[0]
                    best_count = values_counter[best_val]
                    for value in end_values:
                        curr_count = values_counter[value]
                        if curr_count > best_count:
                            best_val = value
                            best_count = curr_count
                    return best_val
                else:
                    return build_tree(split_set)

            max_gain_att = self.find_maximum_gain(data)
            all_attributes.remove(max_gain_att)

            node = {max_gain_att: {}}

            if self.is_continuous[max_gain_att]:
                node[max_gain_att]["pivot"] = self.average(data, max_gain_att)

                # less than pivot
                split_set = data[data[max_gain_att] < node[max_gain_att]["pivot"]]
                split_set = split_set.drop(columns=max_gain_att, axis=1)

                end_values = list(split_set[self.class_attr])
                node[max_gain_att][0] = end_conditions(end_values, 0)

                # more than pivot
                split_set = data[data[max_gain_att] >= node[max_gain_att]["pivot"]]
                split_set = split_set.drop(columns=max_gain_att, axis=1)

                end_values = list(split_set[self.class_attr])
                node[max_gain_att][1] = end_conditions(end_values, 1)
            else:
                for val in data[max_gain_att].unique():
                    split_set = data[data[max_gain_att] == val]
                    split_set = split_set.drop(columns=max_gain_att, axis=1)

                    end_values = list(split_set[self.class_attr])
                    node[max_gain_att][val] = end_conditions(end_values, val)

            return node

        self.prepare_data()
        self.tree = build_tree(temp_data)

    def predict(self):
        pass


dane = load_classic_dataset()
d1, d2 = divide(dane, 0.8)
print(d1)
print()
print(d2)
# idetrzy = ID3(dane, "result")


# def predict(tree: dict, dataset: pd.DataFrame):
#     predictions = []
#     for index, row in dataset.iterrows():
#         key_list = list(tree.keys())
#         attribute = key_list[0]
#         cut_tree = tree[key_list[0]][row[attribute]]
#         while type(cut_tree) is dict:
#             key_list = list(cut_tree.keys())
#             attribute = key_list[0]
#             cut_tree = cut_tree[key_list[0]][row[attribute]]
#         predictions.append(cut_tree)
#
#     return predictions


# def id3(dataset: pd.DataFrame):  # may not work properly for small datasets
#     msk = np.random.rand(len(dataset)) < 0.4
#     print(msk)
#     window = dataset[msk].__deepcopy__()
#     nr_misses = 1  # can be anything greater than 0
#     root = None
#
#     while nr_misses > 0:
#         root = build_tree(window)
#         predictions = predict(root, dataset)
#         col_name = "predictions"
#         dataset[col_name] = predictions
#         misses = get_misclasified(dataset)
#         dataset = dataset.drop(columns=col_name)
#         nr_misses = len(misses)
#         if nr_misses:
#             misses = misses.drop(col_name, axis=1)
#             window = window.append(misses)
#
#     return root



# def count_good(data_frame: pd.DataFrame):
#     count = 0
#     for i, row in data_frame.iterrows():
#         if row[data_frame.columns[-1]] == row[data_frame.columns[-2]]:
#             count += 1
#     return count
#
#
# def get_misclasified(data_frame: pd.DataFrame):
#     # last column is assumed to be predictions and one before that - actual values
#     mis = pd.DataFrame()
#     for i, row in data_frame.iterrows():
#         target = data_frame.columns[-2]
#         prediction = data_frame.columns[-1]
#         if row[target] != row[prediction]:
#             mis.append(row)
#     return mis
#
#
# # todo - clean dataset - several attributes have a bit too many values - G1, G2, G3 and absences -> divide them into
# #  several ranges and assign new values according to that
# #   !!! throughout whole process it is assumed that target attiribute is the last column of dataframe
# data = load_classic_dataset()
# root = id3(data)
# predictions = predict(root, data)
# data["pred"] = predictions
# res = count_good(data)
# if res == len(data):
#     print("Yupi ya yey!")
# else:
#     print("Res: " + str(res) + " out of " + str(len(data)))

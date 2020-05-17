"""
    author: Joanna Sokołowska
             https://github.com/jsokolowska

 todo:
 - wybieranie jak dzielic przedzial liczbowy
 - dowolne wybieranie aatrybutu jako tego na ktory klasyfikujemy
 - wczytywanie danych z dowolnego csv
 -
"""

from numpy import log2
import pandas as pd
import numpy as np
from src.loading_data import load_classic_dataset, load_example_dataset, divide


class id3:
    def __init__(self, dataset: pd.DataFrame, classification_attribute: str):
        self.dataset = dataset
        self.class_attr = classification_attribute
        self.tree = None
        if self.class_attr not in self.dataset.columns:
            raise ValueError("Wrong classification attribute name!")
        self.classification_values = list(set(self.dataset[self.class_attr].to_list()))
        self.is_continuous = {}
        self.prepare_data()
        self.prepare_tree()

    def prepare_data(self):
        for val in self.dataset.columns is not self.class_attr:
            if all(isinstance(n, int) for n in self.dataset[val] ) or all(isinstance(n, float) for n in self.dataset[val] ):
                self.is_continuous[val] = True
            else:
                self.is_continuous[val] = False


    def entropy(self, data: pd.DataFrame) -> float:
        entro = 0.0
        for val in self.classification_values:
            fraction = len(data[data[self.class_attr] == val]) / len(data)
            entro += - fraction * log2(fraction)
        return entro

    def average(self, attribute):
        valSum = 0.0
        for val in self.dataset[attribute].to_list():
            valSum += val
        return valSum / len(self.dataset[attribute].to_list())

    def gain(self, data: pd.DataFrame, attribute: str) -> float:
        if attribute not in self.dataset.columns or attribute == self.class_attr:
            raise ValueError("Wrong attribute for information gain!")
        info_gain = self.entropy(self.dataset)
        if self.is_continuous.get(attribute, None) is False:
            for val in data[attribute].unique():
                info_gain -= self.entropy(data[data[attribute] == val])*len(data[data[attribute] == val])/len(data)
        elif self.is_continuous.get(attribute, None) is True:
            pivot = self.average(attribute)
            info_gain -= self.entropy(data[data[attribute] < pivot]) * len(data[data[attribute] < pivot]) / len(data)
            info_gain -= self.entropy(data[data[attribute] >= pivot]) * len(data[data[attribute] >= pivot]) / len(data)
        return info_gain

    def find_maximum_gain(self, data: pd.DataFrame):
        best_attr = ""
        best_gain = -1.0
        for key in self.dataset.columns():
            current_gain = self.gain(data, key)
            if current_gain >= best_gain:
                best_attr = key
                best_gain = current_gain
        return best_attr

    def prepare_tree(self):
        pass


    def predict(self):
        pass

#
# dane = load_classic_dataset()
# idetrzy = id3(dane, "result")


# def entropy(dataset: pd.DataFrame):
#     target_att = dataset.columns[-1]
#     att_entropy = 0
#     values = dataset[target_att].unique()
#     for value in values:
#         fraction = len(dataset[dataset[target_att] == value]) / len(dataset)
#         att_entropy -= fraction * log2(fraction)
#     return att_entropy


# def attribute_entropy(attribute: str, dataset: pd.DataFrame):
#     att_values = dataset[attribute].unique()
#     dataset_len = len(dataset)
#     att_entropy = 0
#     for value in att_values:
#         split_data = dataset[dataset[attribute] == value]
#         ent = entropy(split_data)
#         multiplier = len(split_data) / dataset_len
#         att_entropy += ent * multiplier
#     return att_entropy


# def choose_biggest_entropy(dataset: pd.DataFrame):
#     set_entropy = entropy(dataset)
#
#     att_entropy = {}
#     for key in dataset.columns[:-1]:
#         ent = attribute_entropy(key, dataset)
#         att_entropy[key] = ent
#
#     info_gain = {}
#     for key, value in att_entropy.items():
#         info_gain[key] = set_entropy - value
#     biggest = -1
#     biggest_att = ""
#     for key, value in info_gain.items():
#         if value > biggest:
#             biggest = value
#             biggest_att = key
#     return biggest_att


# def build_tree(dataset: pd.DataFrame, tree=None):
#     max_entropy_att = choose_biggest_entropy(dataset)
#     att_values = dataset[max_entropy_att].unique()
#     target_att = dataset.columns[-1]
#
#     if tree is None:
#         tree = {max_entropy_att: {}}
#
#     for value in att_values:
#         split_set = dataset[dataset[max_entropy_att] == value]
#         split_set = split_set.drop(columns=max_entropy_att)
#         end_values = split_set[target_att].unique()
#         nr_val = len(end_values)
#         if nr_val != 2:
#             tree[max_entropy_att][value] = end_values[0]
#         else:
#             tree[max_entropy_att][value] = build_tree(split_set)
#     return tree


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


# def predict_for_multiple(dataframe: pd.DataFrame, trees: list):
#     # todo classify using several trees - all while hoping that the item gets classified as "P" in only one of them
#     pass
#
#
# def id3_for_multiple(dataframe: pd.DataFrame):
#     # todo run id3 several times - once for each possible value of target attribute performing P-vs-all
#     #  classification (copy and clean dataset of redundant non-P values
#     pass
#
#
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

"""
    authors:
    Joanna Sokolowska - https://github.com/jsokolowska
    Rafal Uzarowicz - https://github.com/RafalUzarowicz
"""

from numpy import log2
import pandas as pd
import numpy as np
from src.loading_data import load_classic_dataset, load_example_dataset, divide


class ID3:
    def __init__(self, dataset: pd.DataFrame, classification_attribute: str):
        self.dataset = None
        self.class_attr = None
        self.tree = None
        self.classification_values = None
        self.is_continuous = None
        self.prepare_algorithm(dataset, classification_attribute)

    def prepare_algorithm(self, dataset: pd.DataFrame, classification_attribute: str):
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

        self.classification_values = list(set(self.dataset[self.class_attr].to_list()))
        self.is_continuous = {}
        for val in list(self.dataset.columns):
            if all(is_number(n) for n in self.dataset[val]):
                self.is_continuous[val] = True
                self.dataset[val] = pd.to_numeric(self.dataset[val], errors='coerce')
            else:
                self.is_continuous[val] = False

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
            if not key == self.class_attr:
                current_gain = self.gain(data, key)
                if current_gain >= best_gain:
                    best_attr = key
                    best_gain = current_gain
        return best_attr

    def prepare_tree(self):
        all_attributes = list(self.dataset.columns)
        temp_data = self.dataset.__deepcopy__()

        def build_tree(data: pd.DataFrame):

            def end_conditions(end_val: list, curr_dataset: pd.DataFrame):
                # print(end_val)
                if len(all_attributes) == 1:
                    values_counter = {k: 0 for k in self.classification_values}
                    for value in end_val:
                        values_counter[value] += 1
                    best_val = self.classification_values[0]
                    best_count = values_counter[best_val]
                    for value in end_val:
                        curr_count = values_counter[value]
                        if curr_count > best_count:
                            best_val = value
                            best_count = curr_count
                    return best_val

                first_val = end_val[0]

                if all(n == first_val for n in end_val):
                    return first_val

                return build_tree(curr_dataset)

            data = data[all_attributes]
            max_gain_att = self.find_maximum_gain(data)
            all_attributes.remove(max_gain_att)

            node = {max_gain_att: {}}

            if self.is_continuous[max_gain_att]:
                node[max_gain_att]["pivot"] = self.average(data, max_gain_att)

                # less than pivot
                split_set = data[data[max_gain_att] < node[max_gain_att]["pivot"]]
                split_set = split_set.drop(columns=max_gain_att, axis=1)
                if len(split_set) > 0:
                    end_values = list(split_set[self.class_attr])
                    node[max_gain_att][0] = end_conditions(end_values, split_set)

                # more than pivot
                split_set = data[data[max_gain_att] >= node[max_gain_att]["pivot"]]
                split_set = split_set.drop(columns=max_gain_att, axis=1)

                end_values = list(split_set[self.class_attr])
                node[max_gain_att][1] = end_conditions(end_values, split_set)
            else:
                for val in data[max_gain_att].unique():
                    split_set = data[data[max_gain_att] == val]
                    split_set = split_set.drop(columns=max_gain_att, axis=1)

                    end_values = list(split_set[self.class_attr])
                    node[max_gain_att][val] = end_conditions(end_values, split_set)

            return node

        def get_misclasified():
            # last column is assumed to be predictions and one before that - actual values
            try:
                predictions = self.predict(self.dataset)
            except ValueError:
                raise ValueError("Cannot calculate missed data on wrong tree.")
            col_name = "predictions"
            self.dataset[col_name] = predictions
            mis = pd.DataFrame()
            for i, row in self.dataset.iterrows():
                target = self.dataset.columns[self.dataset.columns.get_loc(self.class_attr)]
                prediction = self.dataset.columns[self.dataset.columns.get_loc(col_name)]
                if row[target] != row[prediction]:
                    mis.append(row)
            self.dataset = self.dataset.drop(columns=col_name)
            return mis

        # self.prepare_data()
        # self.tree = build_tree(temp_data)

        msk = np.random.rand(len(self.dataset)) < 0.4
        window = self.dataset[msk].__deepcopy__()
        nr_misses = 1
        self.tree = None

        while nr_misses > 0:
            self.prepare_data()
            self.tree = build_tree(window)
            print(self.tree)
            try:
                misses = get_misclasified()
            except ValueError:
                break
            nr_misses = len(misses)
            if nr_misses:
                misses = misses.iloc[:, :-1]
                window = window.append(misses)

    def predict(self, dataset: pd.DataFrame):
        predictions = []
        for index, row in dataset.iterrows():
            is_found = True
            cut_tree = self.tree
            while type(cut_tree) is dict:
                key_list = list(cut_tree.keys())
                attribute = key_list[0]
                if self.is_continuous.get(attribute):
                    if row[attribute] >= cut_tree[attribute]["pivot"]:
                        cut_tree = cut_tree[attribute][1]
                    else:
                        cut_tree = cut_tree[attribute][0]
                else:
                    if row[attribute] not in cut_tree[attribute].keys():
                        raise ValueError("Tree is not big enough.")
                    cut_tree = cut_tree[attribute][row[attribute]]
            if is_found:
                predictions.append(cut_tree)
        return predictions


# dane = load_classic_dataset()
# dane = load_example_dataset()
# d1, d2 = divide(dane, 0.8)
# print(d1)
# print()
# print(d2)
# idetrzy = ID3(dane, "result")
# idetrzy = ID3(dane, "Walc")




def count_good(data_frame: pd.DataFrame, classification_attribute: str):
    count = 0
    for i, row in data_frame.iterrows():
        if row[classification_attribute] == row["pred"]:
            count += 1
    return count


data = load_classic_dataset()
idetrzy = ID3(data, "result")
predictions = idetrzy.predict(data)
data["pred"] = predictions
res = count_good(data, idetrzy.class_attr)
if res == len(data):
    print("Yupi ya yey!")
else:
    print("Res: " + str(res) + " out of " + str(len(data)))

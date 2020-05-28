"""
    authors:
    Joanna Sokołowska - https://github.com/jsokolowska
    Rafał Uzarowicz - https://github.com/RafalUzarowicz
"""

from numpy import log2
import pandas as pd
import numpy as np


class ID3:
    def __init__(self, dataset: pd.DataFrame = None, target_att: str = None, use_ranges_for_numeric=True, use_window=True):
        self.use_ranges_for_numeric = use_ranges_for_numeric
        self.dataset = None
        self.target_att = None
        self.tree = None
        self.target_att_values = None
        self.is_att_num = None
        self.avg_att_values_for_num = None
        self.att_range_dividers = {}
        self.use_window = use_window
        self.initialize_algorithm(dataset, target_att)
        self.cntr= 0

    def __str__(self):
        return "Classification values: " + str(self.target_att_values) + "\nTree: " + str(self.tree)

    def initialize_algorithm(self, dataset: pd.DataFrame, target_att: str) -> None:
        if dataset is None or target_att is None:
            raise ValueError("Wrong arguments.")
        self.dataset = dataset
        self.target_att = target_att
        self.tree = None
        if self.target_att not in self.dataset.columns:
            raise ValueError("Wrong classification attribute name!")
        # self.target_att_values = list(set(self.dataset[self.target_att].to_list()))
        self.is_att_num = {}
        self.prepare_tree()

    def prepare_data(self) -> None:
        def is_number(string):
            try:
                float(string)
                return True
            except ValueError:
                return False

        self.target_att_values = list(set(self.dataset[self.target_att].to_list()))
        self.is_att_num = {}
        self.avg_att_values_for_num = int(self.find_average_attribute_values_number())
        for val in list(self.dataset.columns):
            if all(is_number(n) for n in self.dataset[val]):
                self.is_att_num[val] = True
                self.dataset[val] = pd.to_numeric(self.dataset[val], errors='coerce')
            else:
                self.is_att_num[val] = False

        if self.use_ranges_for_numeric:
            for attribute in self.dataset.columns:
                if self.is_att_num[attribute] and attribute != self.target_att:
                    min_att_value = self.dataset[attribute].min()
                    max_att_value = self.dataset[attribute].max()
                    self.att_range_dividers[attribute] = []
                    for i in range(1, self.avg_att_values_for_num):
                        divider = min_att_value + ((max_att_value - min_att_value) * i / self.avg_att_values_for_num)
                        self.att_range_dividers[attribute].append(divider)
                    self.att_range_dividers[attribute].append(max_att_value)

    def entropy(self, data: pd.DataFrame) -> float:
        entropy = 0.0
        for val in self.target_att_values:
            if len(data[data[self.target_att] == val]) > 0:
                fraction = len(data[data[self.target_att] == val]) / len(data)
                entropy += - fraction * log2(fraction)
        return entropy

    def find_average_attribute_values_number(self):
        sum_of_unique_attribute_values = 0.0
        counter = 0
        for key in self.dataset.columns:
            if not key == self.target_att:
                sum_of_unique_attribute_values += len(self.dataset[key].unique())
                counter += 1
        if counter > 0:
            return sum_of_unique_attribute_values/counter
        else:
            raise ValueError("Empty dataset!")

    def find_pivot(self, data: pd.DataFrame, attribute: str) -> float:
        # average value for now
        if not self.is_att_num[attribute]:
            raise ValueError("Wrong type of attribute for average function.")
        values_list = data[attribute].to_list()
        if not len(values_list) > 0:
            raise ValueError("Empty dataset.")
        values_sum = 0.0
        for val in values_list:
            values_sum += val
        return values_sum / len(values_list)

    def gain(self, data: pd.DataFrame, attribute: str) -> float:
        if attribute not in self.dataset.columns or attribute == self.target_att:
            raise ValueError("Wrong attribute for information gain!")
        info_gain = self.entropy(data)

        if self.is_att_num.get(attribute, None) is False:
            for val in data[attribute].unique():
                info_gain -= self.entropy(data[data[attribute] == val]) * len(data[data[attribute] == val]) / len(data)

        elif self.is_att_num.get(attribute, False) is True:
            if self.use_ranges_for_numeric:
                data[attribute] = pd.to_numeric(data[attribute], errors='coerce')
                temp_df = data[data[attribute] <= self.att_range_dividers[attribute][0]]
                info_gain -= self.entropy(temp_df) * len(temp_df) / len(data)

                for i in range(1, len(self.att_range_dividers[attribute]) - 1):
                    temp_data = data[data[attribute] <= self.att_range_dividers[attribute][i + 1]].copy()
                    temp_data = temp_data[temp_data[attribute] > self.att_range_dividers[attribute][i]].copy()
                    info_gain -= self.entropy(temp_data) * len(temp_data) / len(data)
                info_gain -= self.entropy(data[data[attribute] > self.att_range_dividers[attribute][-1]]) * len(
                    data[data[attribute] > self.att_range_dividers[attribute][-1]]) / len(data)
            else:
                pivot = self.find_pivot(data, attribute)
                info_gain -= self.entropy(data[data[attribute] >= pivot]) * len(data[data[attribute] >= pivot]) / len(data)
                info_gain -= self.entropy(data[data[attribute] < pivot]) * len(data[data[attribute] < pivot]) / len(data)
        return info_gain

    def find_maximum_gain(self, data: pd.DataFrame) -> str:
        best_attr = ""
        best_gain = -1.0
        for key in data.columns:
            if not key == self.target_att:
                current_gain = self.gain(data, key)
                if current_gain >= best_gain:
                    best_attr = key
                    best_gain = current_gain
        return best_attr

    def prepare_tree(self):
        all_attributes = list(self.dataset.columns)
        self.cntr = 0

        def build_tree(data: pd.DataFrame) -> {}:
            all_attributes = list(data.columns)

            def end_conditions(end_val: list, curr_dataset: pd.DataFrame):
                if len(all_attributes) == 1:
                    values_counter = {k: 0 for k in self.target_att_values}
                    if len(end_val) > 1:
                        self.cntr += 1
                    for value in end_val:
                        values_counter[value] += 1
                    best_val = self.target_att_values[0]
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

            if self.is_att_num[max_gain_att]:
                if self.use_ranges_for_numeric:
                    split_set = data[data[max_gain_att] <= self.att_range_dividers[max_gain_att][0]]
                    split_set = split_set.drop(columns=max_gain_att, axis=1)

                    if len(split_set) > 0:
                        end_values = list(split_set[self.target_att])
                        node[max_gain_att][0] = end_conditions(end_values, split_set)

                    for i in range(1, len(self.att_range_dividers[max_gain_att]) - 1):
                        split_set = data[data[max_gain_att] <= self.att_range_dividers[max_gain_att][i + 1]].copy()
                        split_set = split_set[split_set[max_gain_att] > self.att_range_dividers[max_gain_att][i]].copy()
                        split_set = split_set.drop(columns=max_gain_att, axis=1)

                        if len(split_set) > 0:
                            end_values = list(split_set[self.target_att])
                            node[max_gain_att][i] = end_conditions(end_values, split_set)

                    div = self.att_range_dividers[max_gain_att]
                    split_set = data[data[max_gain_att] < self.att_range_dividers[max_gain_att][-1]]
                    split_set = split_set.drop(columns=max_gain_att, axis=1)

                    if len(split_set) > 0:
                        end_values = list(split_set[self.target_att])
                        node[max_gain_att][len(self.att_range_dividers[max_gain_att]) - 1] = end_conditions(end_values, split_set)
                    div = 3
                else:
                    node[max_gain_att]["pivot"] = self.find_pivot(data, max_gain_att)

                    # less than pivot
                    split_set = data[data[max_gain_att] < node[max_gain_att]["pivot"]]
                    split_set = split_set.drop(columns=max_gain_att, axis=1)
                    if len(split_set) > 0:
                        end_values = list(split_set[self.target_att])
                        node[max_gain_att][0] = end_conditions(end_values, split_set)

                    # more than pivot
                    split_set = data[data[max_gain_att] >= node[max_gain_att]["pivot"]]
                    split_set = split_set.drop(columns=max_gain_att, axis=1)

                    end_values = list(split_set[self.target_att])
                    node[max_gain_att][1] = end_conditions(end_values, split_set)
            else:
                for val in data[max_gain_att].unique():
                    split_set = data[data[max_gain_att] == val]
                    split_set = split_set.drop(columns=max_gain_att, axis=1)

                    end_values = list(split_set[self.target_att])
                    node[max_gain_att][val] = end_conditions(end_values, split_set)

            return node

        msk = np.random.rand(len(self.dataset)) < 0.4
        window = self.dataset[msk]

        def get_misclassified() -> pd.DataFrame:
            predictions = self.predict(self.dataset)
            col_name = "predictions"
            self.dataset[col_name] = predictions
            mis = self.dataset[self.dataset[col_name] != self.dataset[self.target_att]]
            self.dataset = self.dataset.drop(col_name, axis=1)
            mis = mis.drop(col_name, axis=1)
            return mis

        nr_misses = 1
        self.tree = None
        if self.use_window:
            while nr_misses > 0 and len(window) < len(self.dataset):
                self.prepare_data()
                self.tree = build_tree(window)
                all_attributes = list(self.dataset.columns)
                misses = get_misclassified()
                nr_misses = len(misses)
                if nr_misses:
                    window = pd.concat([window, misses]).drop_duplicates().reset_index(drop=True)
        else:
            self.prepare_data()
            self.tree = build_tree(self.dataset)

    def predict(self, dataset: pd.DataFrame) -> []:
        predictions = []
        for index, row in dataset.iterrows():
            is_found = True
            cut_tree = self.tree
            while type(cut_tree) is dict:
                key_list = list(cut_tree.keys())
                attribute = key_list[0]
                if self.is_att_num.get(attribute, False):
                    if self.use_ranges_for_numeric:
                        if row[attribute] > self.att_range_dividers[attribute][-1]:
                            cut_tree = cut_tree[attribute][len(self.att_range_dividers[attribute]) - 1]
                        else:
                            for i in range(len(self.att_range_dividers[attribute])):
                                if row[attribute] <= self.att_range_dividers[attribute][i]:
                                    if i in cut_tree[attribute].keys():
                                        cut_tree = cut_tree[attribute][i]
                                        break
                                    is_found = False
                    else:
                        if row[attribute] >= cut_tree[attribute]["pivot"]:
                            cut_tree = cut_tree[attribute][1]
                        else:
                            cut_tree = cut_tree[attribute][0]
                else:
                    if row[attribute] not in cut_tree[attribute].keys():
                        is_found = False
                        break
                    cut_tree = cut_tree[attribute][row[attribute]]
            if is_found:
                predictions.append(cut_tree)
            else:
                predictions.append(None)
            # print(index)
        return predictions


def count_good(data_frame: pd.DataFrame, classification_attribute: str) -> int:
    count = 0
    for i, row in data_frame.iterrows():
        if row[classification_attribute] == row["pred"]:
            count += 1
    return count

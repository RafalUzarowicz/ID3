from numpy import log2
import pandas as pd
from src.loading_data import load_classic_dataset, load_data, divide


def entropy(dataset: pd.DataFrame):
    target_att = dataset.columns[-1]
    att_entropy = 0
    values = dataset[target_att].unique()
    for value in values:
        fraction = len(dataset[dataset[target_att] == value]) / len(dataset)
        att_entropy -= fraction * log2(fraction)
    return att_entropy


def attribute_entropy(attribute: str, dataset: pd.DataFrame):
    att_values = dataset[attribute].unique()
    dataset_len = len(dataset)
    att_entropy = 0
    for value in att_values:
        split_data = dataset[dataset[attribute] == value]
        ent = entropy(split_data)
        multiplier = len(split_data) / dataset_len
        att_entropy += ent * multiplier
    return att_entropy


def choose_biggest_entropy(dataset):
    set_entropy = entropy(dataset)

    att_entropy = {}
    for key in dataset.columns[:-1]:
        ent = attribute_entropy(key, dataset)
        att_entropy[key] = ent

    info_gain = {}
    for key, value in att_entropy.items():
        info_gain[key] = set_entropy - value
    biggest = -1
    biggest_att = ""
    for key, value in info_gain.items():
        if value > biggest:
            biggest = value
            biggest_att = key
    return biggest_att


def build_tree(dataset, tree=None):
    max_entropy_att = choose_biggest_entropy(dataset)
    att_values = dataset[max_entropy_att].unique()
    target_att = dataset.columns[-1]

    if tree is None:
        tree = {max_entropy_att: {}}

    for value in att_values:
        split_set = dataset[dataset[max_entropy_att] == value]
        split_set = split_set.drop(columns=max_entropy_att)
        end_values = split_set[target_att].unique()
        nr_val = len(end_values)
        if nr_val != 2:
            tree[max_entropy_att][value] = end_values[0]
        else:
            tree[max_entropy_att][value] = build_tree(split_set)
    return tree


def predict(tree: dict, dataset: pd.DataFrame):
    predictions = []
    for index, row in dataset.iterrows():
        key_list = list(tree.keys())
        attribute = key_list[0]
        cut_tree = tree[key_list[0]][row[attribute]]
        while type(cut_tree) is dict:
            key_list = list(cut_tree.keys())
            attribute = key_list[0]
            cut_tree = cut_tree[key_list[0]][row[attribute]]
        predictions.append(cut_tree)

    return predictions


def id3(dataset: pd.DataFrame):
    # choose portion as window
    # classify
    # chceck for correctnes
    # add misclassified items to window
    # repeat until all classified
    pass


def predict_for_multiple(dataframe):
    pass


def id3_for_multiple():
    pass


def count_good(data_frame: pd.DataFrame):
    count = 0
    for i, row in data_frame.iterrows():
        if row[data_frame.columns[-1]] == row[data_frame.columns[-2]]:
            count += 1
    return count


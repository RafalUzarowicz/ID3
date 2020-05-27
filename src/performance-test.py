"""
    author: Joanna Sokołowska - https://github.com/jsokolowska

 """

import numpy as np
import pandas as pd

from src import ID3DatasetLoader, ID3, count_good

# loader = ID3DatasetLoader()
# loader.load_example_dataset()
# data = loader.get_dataset()
# id3 = ID3(data, "Walc")
#
# predictions = id3.predict(data)
# print("Results")
# print(predictions)
#
# loader = ID3DatasetLoader()
# loader.load_example_dataset()
# data = loader.get_dataset()
# id3 = ID3(data, "Walc")
# predictions = id3.predict(data)
# print("Results")
# print(predictions)
id3_data_loader = ID3DatasetLoader()
id3_data_loader.load_example_dataset()
data = id3_data_loader.get_dataset()
idetrzy = ID3(data, "Walc", True, False)

print(idetrzy.find_average_attribute_values_number())
print(idetrzy)
predictions = idetrzy.predict(data)
data["pred"] = predictions
res = count_good(data, idetrzy.classification_attribute)
if res == len(data):
    print("Yupi ya yey!")
else:
    print("Res: " + str(res) + " out of " + str(len(data)))




#
#
# def noise_level_test(target_att, step=5, repeats=25, file_name=None):
#     # if file_name is not None:
#     #     dataset = load_data_from_file(file_name)
#     # else:
#     #     dataset = load_example_dataset("both")
#     #
#     # noise_levels = [a * step for a in range(1, 100 // step)]
#     #
#     # for noise_lvl in noise_levels:
#     #     for i in range(repeats):
#     #         data_cpy = dataset.copy(True)
#     #         corrupt_data(noise_lvl, data_cpy, target_att)
#     # corrupt data(noiselvl)
#     # divide dataset
#     # do id3
#     # chceck how correct that was
#     # remember
#     pass
#
#
# def corrupt_data(noise_lvl, dataset, target_att):
#     pass
#
#
# def predict_alcohol_consumption():
#     dataset = load_example_dataset("both")
#     msk = np.random.random(len(dataset)) < 0.2
#     dataset_lst = [dataset[msk]]
#     dataset = dataset[~msk]
#     msk = np.random.random()
#     # load_data
#     # cross-validate
#     # results
#     # confusion matrix
#     pass
#
#
# def examine_dataset():
#     pass

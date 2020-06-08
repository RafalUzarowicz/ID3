from src.id3 import *
from src.loading_data import *

# id3_data_loader = ID3DatasetLoader()
# id3_data_loader.load_example_dataset(which="mat")
# data = id3_data_loader.get_dataset()
# id3 = ID3(dataset=data, target_att="Walc", use_ranges_for_numeric=False)
#
# # predictions = id3.predict(data)
# # print("Results")
# # print(predictions)
#
# # id3_data_loader.load_classic_dataset()
# # # id3_data_loader.load_classic_with_numbers_dataset()
# # data = id3_data_loader.get_dataset()
# # id3 = ID3(data, "result")
# print(id3.find_average_attribute_values_number())
# print(id3)
# id3_data_loader.load_example_dataset(which="por")
# data = id3_data_loader.get_dataset()
# predictions = id3.predict(data)
# data["pred"] = predictions
# res = count_good(data, id3.target_att)
# if res == len(data):
#     print("Yupi ya yey!")
# else:
#     print("Res: " + str(res) + " out of " + str(len(data)))

from src.id3 import *
from src.loading_data import *

def main():
    id3_data_loader = ID3DatasetLoader()
    id3_data_loader.load_classic_dataset()
    data = id3_data_loader.get_dataset()
    idetrzy = ID3(data, "result")
    print(idetrzy)
    predictions = idetrzy.predict(data)
    data["pred"] = predictions
    res = count_good(data, idetrzy.classification_attribute)
    if res == len(data):
        print("Yupi ya yey!")
    else:
        print("Res: " + str(res) + " out of " + str(len(data)))

    # dane = load_classic_dataset()
    # dane = load_example_dataset()
    # d1, d2 = divide(dane, 0.8)
    # print(d1)
    # print()
    # print(d2)
    # idetrzy = ID3(dane, "result")
    # idetrzy = ID3(dane, "Walc")

if __name__ == "__main__":
    main()
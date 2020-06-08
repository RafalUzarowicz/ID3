"""
    authors:
    Rafał Uzarowicz - https://github.com/RafalUzarowicz
    Joanna Sokołowska - https://github.com/jsokolowska
"""
import argparse
from pathlib import Path
from src.id3 import ID3
from src.loading_data import ID3DatasetLoader

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description='''\
ID3 algorithm - tree-based classification
------------------------------------------------------------------''')
parser.add_argument('filepath_training', help="path to file with training data")
parser.add_argument('filepath_predict', help="path to file with data to predict")
parser.add_argument('target_att', help="attribute to classify with")

parser.add_argument('-t', '--file_type', type=str, default="csv",
                    help="chooses data files types, "
                         "available types: csv json xlsx")
parser.add_argument('-r', '--ranges_for_numeric', action='store_true', default=False,
                    help="enables using ranges in tree creation,"
                    "if program is run without this flag, pivot dividing is used for numeric attributes")
parser.add_argument('-w', '--window', action='store_true', default=False,
                    help="enables using windows in tree creation")
parser.add_argument('--numeric_attributes', metavar='N', type=str, nargs='+', default=None,
                    help='attributes that will be processed as numeric, '
                    "if nothing is passed all attributes with all int or float values will be numeric")
args = parser.parse_args()

if not Path(args.filepath_training).is_file():
    print("File with training data not found.\n")
elif not Path(args.filepath_predict).is_file():
    print("File with data for prediction not found.\n")
else:
    is_loaded = True
    loader = ID3DatasetLoader()
    if args.file_type == "csv":
        loader.load_csv_dataset(args.filepath_training)
    elif args.file_type == "json":
        loader.load_json_dataset(args.filepath_training)
    elif args.file_type == "xlsx":
        loader.load_excel_dataset(args.filepath_training)
    else:
        is_loaded = False
        print("File type not supported.\n")

    if is_loaded:
        data = loader.get_dataset()
        if args.numeric_attributes is not None:
            numeric = list(args.numeric_attributes)
        else:
            numeric = None
        id3 = ID3(data, args.target_att, args.ranges_for_numeric, args.window, numeric)
        predictions = id3.predict(data)
        data["predictions"] = predictions
        if args.file_type == "csv":
            data.to_csv("id3_results.csv")
        elif args.file_type == "json":
            data.to_json("id3_results.json")
        elif args.file_type == "excel":
            data.to_excel("id3_results.xlsx")

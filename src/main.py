"""
    authors:
    Joanna Sokołowska - https://github.com/jsokolowska
    Rafał Uzarowicz - https://github.com/RafalUzarowicz
"""
import argparse
from pathlib import Path
from src.id3 import ID3
from src.loading_data import ID3DatasetLoader

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description='''\
ID3 algorithm - tree-base classification
------------------------------------------------------------------''')
parser.add_argument('filepath_training', help="path to file with training data", required=True)
parser.add_argument('filepath_predict', help="path to file with data to predict", required=True)
parser.add_argument('target_att', help="attribute to classify with", required=True)

parser.add_argument('-t', '--file_type', action='store_true', default="csv",
                    help="chooses data files types."
                         "available types: csv json excel sql table")
parser.add_argument('-r', '--use_ranges_for_numeric', action='store_true', default=False,
                    help="enables using ranges in tree creation"
                    "if this value is false pivot dividing is used for numeric attributes")
parser.add_argument('-w', '--use_window', action='store_true', default=False,
                    help="enables using windows in tree creation")
parser.add_argument('numeric_attributes', metavar='N', type=str, nargs='+',
                    help='attributes that will be changed to numeric'
                    "if nothing is passed all attributes with all int or float values will be numeric")
args = parser.parse_args()

if not Path(args.filepath_training).is_file():
    print("File with training data not found.\n")
elif not Path(args.filepath_predict).is_file():
    print("File with data for prediction not found.\n")
else:
    is_loaded = True
    loader = ID3DatasetLoader()
    if args.file_type is "csv":
        loader.load_csv_dataset(args.filepath_training)
    elif args.file_type is "json":
        loader.load_json_dataset(args.filepath_training)
    elif args.file_type is "excel":
        loader.load_excel_dataset(args.filepath_training)
    elif args.file_type is "sql":
        loader.load_sql_table_dataset(args.filepath_training)
    elif args.file_type is "table":
        loader.load_table_dataset(args.filepath_training)
    else:
        is_loaded = False
        print("File type not supported.\n")

    if is_loaded:
        data = loader.get_dataset()
        id3 = ID3(data, args.target_att, args.use_ranges_for_numeric, args.use_window, list(args.numeric_attributes))
        predictions = id3.predict(data)
        print(predictions)

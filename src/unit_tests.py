import unittest
from src.id3 import *
from src.loading_data import *


class ID3Test(unittest.TestCase):

    def test_entropy(self):
        loader = ID3DatasetLoader()
        loader.load_classic_dataset()
        data = loader.get_dataset()
        id3_algo = ID3(data, "result")

        entropy_windy_true = id3_algo.entropy(data[data["windy"] == "true"])
        entropy_windy_false = id3_algo.entropy(data[data["windy"] == "false"])

        self.assertAlmostEqual(entropy_windy_true, 1.0, places=8)
        self.assertAlmostEqual(entropy_windy_false, 0.5 - (3 / 4) * log2(3 / 4), places=8)

    def test_gain(self):
        loader = ID3DatasetLoader()
        loader.load_classic_dataset()
        data = loader.get_dataset()
        id3_algo = ID3(data, "result")

        gain_windy = id3_algo.gain(data, "windy")

        self.assertAlmostEqual(gain_windy,
                               id3_algo.entropy(data) - 6 / 14 - (8 / 14) * 0.5 + (8 / 14) * (3 / 4) * log2(3 / 4),
                               places=7)

    def test_find_maximum_gain(self):
        loader = ID3DatasetLoader()
        loader.load_classic_dataset()
        data = loader.get_dataset()
        id3_algo = ID3(data, "result")

        gain_max_attr = id3_algo.find_maximum_gain(data)

        self.assertEqual(gain_max_attr, "outlook")

    def test_find_pivot(self):
        loader = ID3DatasetLoader()
        loader.load_classic_with_numbers_dataset()
        data = loader.get_dataset()
        id3_algo = ID3(data, "result")

        pivot_found_int = id3_algo.find_pivot(data, "number_int")
        pivot_found_float = id3_algo.find_pivot(data, "number_float")

        self.assertAlmostEqual(pivot_found_int, 1.85714285714, places=8)
        self.assertAlmostEqual(pivot_found_float, 4.64285714285, places=8)

    def test_find_average_attribute_values_number(self):
        loader = ID3DatasetLoader()
        loader.load_classic_dataset()
        data = loader.get_dataset()
        id3_algo = ID3(data, "result")

        avg_num = id3_algo.find_average_attribute_values_number()

        self.assertAlmostEqual(avg_num, 2.5, places=8)

    def test_count_good(self):
        loader = ID3DatasetLoader()
        loader.load_classic_dataset()
        data = loader.get_dataset()
        id3_algo = ID3(data, "result")

        predictions = id3_algo.predict(data)
        data["pred"] = predictions
        res = count_good(data, id3_algo.target_att)

        self.assertEqual(res, 14)

    def test_id3(self):
        loader = ID3DatasetLoader()
        loader.load_classic_dataset()
        data = loader.get_dataset()
        id3_algo = ID3(data, "result")

        predictions = id3_algo.predict(data)
        data["pred"] = predictions
        res = count_good(data, id3_algo.target_att)

        self.assertEqual(res, 14)


if __name__ == '__main__':
    unittest.main()

import sys
from typing import Self
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from vec.vec import Vec as Vector
from w2v import get_word_vector, load_model

class TestVector(unittest.TestCase):

    def test_mean_word(self):
        model = load_model("../glove50/glove_50_fast.wordvectors")
        raw_vec = get_word_vector(model, "india")
        v = Vector([float(x) for x in raw_vec])

        expected_mean = sum(raw_vec) / len(raw_vec)
        self.assertAlmostEqual(v.mean(), expected_mean, places=5)
        print(expected_mean, v.mean())

    def test_demean(self):
        model = load_model("../glove50/glove_50_fast.wordvectors")
        raw_vec = get_word_vector(model, "india")
        v = Vector([float(x) for x in raw_vec])

        self.assertAlmostEqual(v.demean(), 0.0, places=5)
        print("Demean:", v.demean())

    def test_standard_deviation(self):
        model = load_model("../glove50/glove_50_fast.wordvectors")
        raw_vec = get_word_vector(model, "india")
        v = Vector([float(x) for x in raw_vec])

        expected_std_dev = (sum((x - v.mean()) ** 2 for x in raw_vec) / len(raw_vec)) ** 0.5
        self.assertAlmostEqual(v.standard_deviation(), expected_std_dev, places=5)
        print("Std Dev:", expected_std_dev, v.standard_deviation())

    
if __name__ == "__main__":
    unittest.main()
import os
import unittest
import sys

sys.path.append('../')

from Analysis.cat_typicality_analysis import *
from test_functions import *
from Analysis.process_data import *
from pandas._testing import assert_frame_equal

os.chdir("..")


class Test(unittest.TestCase):
    """Summary
    """

    # @unittest.skip
    def test_same_outputs(self):
        output_2020_study_results(base_folder=output_folder)
        study_info = StudyInfo("2020 study")

        results_csv = output_folder + study_info.stats_folder + "/" + "results.csv"
        new_results_df = pd.read_csv(results_csv)
        original_results_df = pd.read_csv(get_original_csv(results_csv))

        disag_csv = output_folder + study_info.stats_folder + "/" + "disagreements.csv"
        new_disag_df = pd.read_csv(disag_csv)
        original_disag_df = pd.read_csv(get_original_csv(disag_csv))

        assert_frame_equal(new_results_df, original_results_df)
        assert_frame_equal(new_disag_df, original_disag_df)

    def test_configs(self):
        c1 = SimpleConfiguration('test_scene', 'fig1', 'gr1')
        c2 = SimpleConfiguration('test_scene', 'fig1', 'gr1')
        c3 = SimpleConfiguration('test_scene', 'fig3', 'gr1')
        c4 = SimpleConfiguration('test_scene', 'fig1', 'gr3')

        self.assertTrue(c1.configuration_match(c2))
        self.assertTrue(c2.configuration_match(c1))
        self.assertFalse(c1.configuration_match(c3))
        self.assertFalse(c3.configuration_match(c1))
        self.assertFalse(c3.configuration_match(c4))
        self.assertFalse(c4.configuration_match(c3))

        assert type(c1.configuration_match(c2)) == type(True)


if __name__ == "__main__":
    unittest.main()

import unittest
import sys
sys.path.append('../')

from Analysis.preprocess_features import Features
from Analysis.process_data import *


class Test(unittest.TestCase):
    """Summary
    """

    # @unittest.skip
    def test_typ_p_values(self):
        study_info = StudyInfo("test study")
        # Begin by loading users
        userdata = UserData(study_info)

        ## typicality data
        typ_data = TypicalityData(userdata)
        c1 = SimpleConfiguration("compsvo13v", "book", "board")
        c2 = SimpleConfiguration("compsvi3", "pear", "mug")
        values = typ_data.calculate_pvalue_c1_better_than_c2("on", c1, c2)

        self.assertEqual(values, [5, 1, 4, 0.96875])

    # @unittest.skip
    def test_svmod_p_values(self):
        study_info = StudyInfo("test study")
        # Begin by loading users
        userdata = UserData(study_info)

        ## typicality data
        svmod_data = ModSemanticData(userdata)
        c1 = SimpleConfiguration("compsvula", "pencil", "lamp")
        c2 = SimpleConfiguration("compsvul", "bowl", "lamp")
        values = svmod_data.calculate_pvalue_c1_better_than_c2("on", c1, c2)
        print(values)
        self.assertEqual(values, [4, 0, 0, 6, 0.004761904761904759])

    def test_preprocess_data(self):
        study_info = StudyInfo("test study")
        f = study_info.feature_processor
        print(f.means)
        self.assertAlmostEqual(f.means['support'], 0.2717375845)
        self.assertAlmostEqual(f.means['contact_proportion'], 0.06424559126607)


if __name__ == "__main__":
    unittest.main()

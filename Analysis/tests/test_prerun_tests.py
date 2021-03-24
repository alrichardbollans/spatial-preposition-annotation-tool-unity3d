import os
import unittest
import sys
sys.path.append('../')

from Analysis.basic_model_testing import preposition_list
from Analysis.classes import Constraint
from Analysis.compile_instances import *

from test_functions import *
import pandas as pd
from pandas._testing import assert_frame_equal

os.chdir('..')


# These tests don't change any files, just check that the values are correct

class MyTestCase(unittest.TestCase):
    # @unittest.skip
    def test_sv_outputs(self):
        study_info = StudyInfo("2019 study")

        svcollection = SemanticCollection(study_info)

        for preposition in svcollection.get_used_prepositions():
            print(preposition)

            new_df, original_df = generate_dataframes_to_compare(
                svcollection.study_info.config_ratio_csv(svcollection.filetag, preposition))

            assert_frame_equal(new_df, original_df)

    def test_comp_outputs(self):
        study_info = StudyInfo("2019 study")
        compcollection = ComparativeCollection(study_info)

        ## Compare constraints in csv with generated constraints
        constraints = Constraint.read_from_csv(study_info.constraint_csv)

        out = dict()
        for preposition in StudyInfo.preposition_list:
            # print(preposition)
            preposition_constraints = []
            for my_scene in study_info.scene_list:

                grounds = my_scene.selectable_objects

                for grd in grounds:

                    c = Comparison(my_scene.name, preposition, grd, study_info)
                    Cons = c.generate_constraints(compcollection.instance_list)
                    for con in Cons:
                        preposition_constraints.append(con)
            out[preposition] = preposition_constraints
        for p in preposition_list:
            self.assertEqual(len(out[p]), len(constraints[p]))
            index = 0
            for c1 in out[p]:
                c2 = constraints[p][index]
                print(c1)
                print(c2)
                self.assertTrue(Constraint.constraint_name_match(c1, c2))
                self.assertTrue(Constraint.constraint_feature_value_match(c1, c2))
                index += 1


    # @unittest.skip
    def test_clean_outputs_process_data(self):
        study_info = StudyInfo("2019 study")

        new_clean_sem_csv = study_info.data_folder + "/" + StudyInfo.sem_annotations_name
        new_clean_comp_csv = study_info.data_folder + "/" + StudyInfo.comp_annotations_name
        new_clean_user_csv = study_info.clean_user_csv

        new_clean_sem_dataset, original_clean_sem_dataset = generate_dataframes_to_compare(new_clean_sem_csv)
        new_clean_user_dataset, original_clean_user_dataset = generate_dataframes_to_compare(
            new_clean_user_csv)
        # Doesn't like reading the full csv as different column lengths, so need to specifywhich columns to use.
        new_clean_comp_dataset, original_clean_comp_dataset = generate_dataframes_to_compare(
            new_clean_comp_csv, columns_to_use=[0, 1, 2, 3, 4, 5, 6, 7])

        assert_frame_equal(new_clean_sem_dataset, original_clean_sem_dataset)
        assert_frame_equal(new_clean_comp_dataset, original_clean_comp_dataset)
        assert_frame_equal(new_clean_user_dataset, original_clean_user_dataset)

    def test_feature_values(self):
        study_info = StudyInfo("2019 study")

        input_feature_csv = study_info.input_feature_csv
        output_path = study_info.feature_output_csv
        means_output_path = study_info.feature_output_folder + "/feature_means.csv"
        std_output_path = study_info.feature_output_folder + "/feature_stds.csv"
        human_readable_path = study_info.human_readable_feature_output_csv

        new_df, original_df = generate_dataframes_to_compare(input_feature_csv)
        assert_frame_equal(new_df, original_df)
        new_df, original_df = generate_dataframes_to_compare(output_path)
        assert_frame_equal(new_df, original_df)
        # new_df, original_df = generate_dataframes_to_compare(means_output_path)
        #
        # assert_frame_equal(new_df, original_df)
        # new_df, original_df = generate_dataframes_to_compare(std_output_path)
        #
        # assert_frame_equal(new_df, original_df)
        new_df, original_df = generate_dataframes_to_compare(human_readable_path)
        print(new_df)
        print(original_df)
        assert_frame_equal(new_df, original_df)

    def test_specs(self):

        self.assertEqual(pd.__version__, '1.0.3')


if __name__ == '__main__':
    unittest.main()

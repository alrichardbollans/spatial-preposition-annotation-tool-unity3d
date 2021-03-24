import os
import unittest

from polysemy_analysis import *

from Analysis.data_import import StudyInfo
from test_functions import *

from pandas._testing import assert_frame_equal

os.chdir("..")


class Test(unittest.TestCase):
    """Summary
    """

    # @unittest.skip
    def test_polyseme_rank_info(self):
        '''
        When this info is outputted it uses preserve_empty_polysemes = True.
        :return:
        '''
        study_info = StudyInfo("2019 study")

        all_scenes = study_info.scene_name_list
        generated_polyseme_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info,
                                                           preserve_empty_polysemes=True)

        # Check ranks
        generated_polyseme_models.non_shared.output_polyseme_info()
        model_name = GeneratePolysemeModels.distinct_model_name
        for preposition in polysemous_preposition_list:
            new_rank_csv = study_info.polyseme_data_folder + model_name + "/ranks/" + preposition + " -ranks.csv"
            new_rank_df, original_rank_df = generate_dataframes_to_compare(new_rank_csv)

            cols = new_rank_df.columns.tolist()
            redordered_orig_df = original_rank_df[cols]
            assert_frame_equal(new_rank_df, redordered_orig_df)

    # @unittest.skip
    def test_initial_model(self):
        cluster_numbers = KMeansPolysemyModel.cluster_numbers
        self.assertEqual(cluster_numbers["on"], 8)
        self.assertEqual(cluster_numbers["in"], 4)
        self.assertEqual(cluster_numbers["over"], 4)
        self.assertEqual(cluster_numbers["under"], 4)

        study_info = StudyInfo("2019 study")

        all_scenes = study_info.scene_name_list
        generated_polyseme_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info)

        p_models = generated_polyseme_models.models

        all_csv = study_info.polysemy_score_folder + "all_test.csv"
        original_dataframe = pd.read_csv(get_original_csv(all_csv), index_col=0)
        print(original_dataframe)

        t = TestModels(p_models, "all")

        new_dframe = t.score_dataframe
        print(new_dframe)

        # reindex original as it contains shared aswell but new doesn't
        original_reindexed = dropcolumns_reindexlike(original_dataframe, new_dframe)

        assert_frame_equal(new_dframe, original_reindexed)

        # first check basic parametres
        for model in p_models:
            self.assertEqual(len(model.all_feature_keys), 13)
            if hasattr(model, "preposition_model_dict"):
                for p in model.test_prepositions:
                    self.assertEqual(len(model.preposition_model_dict[p].feature_keys), 10)

        # Check typicalities
        for model in p_models:

            for preposition in polysemous_preposition_list:
                model.output_typicalities(preposition)

        for preposition in polysemous_preposition_list:
            # Remove Kmeans column as it is not deterministic
            new_typicality_csv = generated_polyseme_models.study_info.base_polysemy_folder + "config typicalities/typicality-" + preposition + ".csv"
            new_typicality_df, original_typicality_df = generate_dataframes_to_compare(new_typicality_csv,
                                                                                       columns_to_use=[0, 1, 2, 3, 4,
                                                                                                       5, 6])
            print(new_typicality_df)
            print(original_typicality_df)
            assert_frame_equal(new_typicality_df, original_typicality_df)

    # @unittest.skip
    def test_k_fold(self):
        study_info = StudyInfo("2019 study")

        m = MultipleRunsPolysemyModels(study_info, number_runs=10, k=10, compare="y")

        self.assertIsInstance(m.Generate_Models_all_scenes, GeneratePolysemeModels)
        self.assertIsInstance(m.Generate_Models_all_scenes.features_to_remove, list)
        self.assertEqual(m.Generate_Models_all_scenes.features_to_remove, Configuration.ground_property_features.copy())

        self.assertIsInstance(m.model_name_list, list)

        test_folds = m.get_validation_scene_split()

        while not m.folds_check(test_folds):
            test_folds = m.get_validation_scene_split()

        self.assertEqual(len(test_folds), 10)

        for f1 in test_folds:
            self.assertNotEqual(len(f1), 0)
            for f2 in test_folds:
                if f1 != f2:
                    f1_set = set(f1)
                    f2_set = set(f2)
                    self.assertFalse(f1_set & f2_set)


if __name__ == "__main__":

    if len(polysemous_preposition_list) > 4:
        print("polysemous_preposition_list contains non-polysemous prepositions")
        print("Changing for test")
        polysemous_preposition_list = ['in', 'on', 'under', 'over']
    unittest.main()

# This is a non-exhaustive file for testing extra bits included in the thesis which haven't been formally archived.

# Things to move across
# Non-polysemous prepositions analysis
# Refined model analysis
import copy
import random

from sklearn.model_selection import train_test_split

from Analysis.basic_model_testing import MultipleRunsGeneric
from basic_model_testing import TestModels
from data_import import StudyInfo
from polysemy_analysis import DistinctPrototypePolysemyModel, preposition_list, SalientFeature, \
    polysemous_preposition_list, GeneratePolysemeModels, KMeansPolysemyModel, MultipleRunsPolysemyModels

extra_thesis_folder = "extra thesis results/"


class DataPartitionPolysemyModel(DistinctPrototypePolysemyModel):
    # THis model helps to check if arbitrailiy partitioning the data improves the baseline prototpe model
    name = "Partition Model"

    def __init__(self, train_scenes, test_scenes, study_info_, test_prepositions=preposition_list,
                 preserve_empty_polysemes=False,
                 baseline_model=None,
                 features_to_remove=None):
        DistinctPrototypePolysemyModel.__init__(self, DataPartitionPolysemyModel.name, train_scenes, test_scenes,
                                                study_info_, test_prepositions=test_prepositions,
                                                preserve_empty_polysemes=preserve_empty_polysemes,
                                                baseline_model=baseline_model, features_to_remove=features_to_remove)

    def refine_ideal_meaning(self, preposition, original_salient_features):
        """
        Refines the ideal meaning by finding median feature values of good instances.
        Outputs new list of polysemes for the model
        :param preposition:
        :param original_salient_features:
        :return:
        """
        if preposition in self.test_prepositions:
            # Find value of feature such that half of preposition instances are greater and half are less than value
            number_of_features = len(original_salient_features)

            candidate_features = []

            # Get new salient features
            for feature in self.study_info.all_feature_keys:
                if feature not in self.features_to_remove:
                    if not any(x.name == feature for x in original_salient_features):
                        candidate_features.append(feature)

            new_features = random.choices(candidate_features, k=number_of_features)

            new_salient_features = []
            for f in new_features:
                median = self.baseline_model.preposition_model_dict[preposition].goodAllFeatures[f].median()

                new_f = SalientFeature(f, median, "g")
                new_salient_features.append(new_f)

            new_polysemes = self.generate_polysemes(preposition, new_salient_features)
            return new_polysemes
        else:
            return None


class DistinctPrototypeMedianPolysemyModel(DistinctPrototypePolysemyModel):

    def __init__(self, name, train_scenes, test_scenes, study_info_, test_prepositions=preposition_list,
                 preserve_empty_polysemes=False,
                 baseline_model=None,
                 features_to_remove=None):
        DistinctPrototypePolysemyModel.__init__(self, name, train_scenes, test_scenes, study_info_,
                                                test_prepositions=test_prepositions,
                                                preserve_empty_polysemes=preserve_empty_polysemes,
                                                baseline_model=baseline_model, features_to_remove=features_to_remove)

    def refine_ideal_meaning(self, preposition, original_salient_features):
        """
        Refines the ideal meaning by finding median feature values of good instances.
        Outputs new list of polysemes for the model
        :param preposition:
        :param original_salient_features:
        :return:
        """

        # Find value of feature such that half of preposition instances are greater and half are less than value
        if preposition in self.test_prepositions:
            new_salient_features = []
            for f in original_salient_features:
                new_f = copy.deepcopy(f)

                median = self.baseline_model.preposition_model_dict[preposition].goodAllFeatures[new_f.name].median()
                new_f.value = median

                new_salient_features.append(new_f)

            new_polysemes = self.generate_polysemes(preposition, new_salient_features)
            return new_polysemes
        else:
            return None


class DistinctPrototypeRefinedPolysemyModel(DistinctPrototypePolysemyModel):

    def __init__(self, name, train_scenes, test_scenes, study_info_, test_prepositions=preposition_list,
                 preserve_empty_polysemes=False,
                 baseline_model=None,
                 features_to_remove=None):
        DistinctPrototypePolysemyModel.__init__(self, name, train_scenes, test_scenes, study_info_,
                                                test_prepositions=test_prepositions,
                                                preserve_empty_polysemes=preserve_empty_polysemes,
                                                baseline_model=baseline_model, features_to_remove=features_to_remove)

    def refine_ideal_meaning(self, preposition, original_salient_features):
        """
        Refines the ideal meaning by testing a validation set.
        Outputs new list of polysemes for the model
        :param preposition:
        :param original_salient_features:
        :return:
        """
        if preposition in self.test_prepositions:
            new_salient_features = []

            # Each of the salient features are proportions so we use these values
            # This makes generation non-deterministic for these models
            train_scenes, validation_scenes = train_test_split(self.train_scenes, test_size=0.5)


            for f in original_salient_features:
                g_values_to_try = [0.5, 0.6, 0.7, 0.8, 0.9]
                l_values_to_try = [0.1, 0.2, 0.3, 0.4, 0.5]

                if f.name == "horizontal_distance":
                    g_values_to_try = [0.05, 0.1, 0.15, 0.2]
                    l_values_to_try = g_values_to_try
                if f.name == "contact_proportion":
                    g_values_to_try = [0.1, 0.2, 0.3, 0.4, 0.5]
                    l_values_to_try = [0.1, 0.2, 0.3, 0.4, 0.5]
                if f.gorl == "l":
                    values_to_try = l_values_to_try.copy()
                else:
                    values_to_try = g_values_to_try.copy()

                best_value = None
                best_score = 0
                for v in values_to_try:

                    # Convert to standardised values
                    v = self.feature_processer.convert_normal_value_to_standardised(f.name, v)

                    score1 = self.test_ideal_feature_value(train_scenes, validation_scenes, preposition,
                                                           original_salient_features, f.name, v)
                    score2 = self.test_ideal_feature_value(validation_scenes, train_scenes, preposition,
                                                           original_salient_features, f.name, v)

                    total = score1 + score2
                    if total > best_score:
                        best_score = total
                        best_value = v

                if best_value is None:
                    raise ValueError("best_value unassigned")

                # The original feature is updated, which is better for training the next feature
                f.value = best_value
                new_salient_features.append(f)


            new_polysemes = self.generate_polysemes(preposition, new_salient_features)



            return new_polysemes
        else:
            return None

    def test_ideal_feature_value(self, train_scenes, validation_scenes, preposition, original_salient_features, feature,
                                 value):
        """
        Generates new polysemes for the model from the given feature and value,
        then tests on given test scenes.
        :param train_scenes:
        :param preposition:
        :param original_salient_features:
        :param feature:
        :param value:
        :return:
        """

        # First update the salient features
        new_salient_features = []
        for f in original_salient_features:
            new_f = copy.deepcopy(f)
            if new_f.name == feature:
                new_f.value = value
            new_salient_features.append(new_f)

        # Create new polysemes
        new_polysemes = self.generate_polysemes(preposition, new_salient_features, train_scenes=train_scenes)
        # The polyseme dict is updated here so that the model score can be calculated

        self.polyseme_dict[preposition] = new_polysemes

        all_constraints = self.constraint_dict[preposition]
        # Constraints to test on
        test_constraints = []

        for c in all_constraints:
            if c.scene in validation_scenes:
                test_constraints.append(c)

        # Get score for preposition
        score_two = self.weighted_score(test_constraints)

        return score_two


class SharedPrototypeRefinedPolysemyModel(DistinctPrototypeRefinedPolysemyModel):

    def __init__(self, name, train_scenes, test_scenes, study_info_, preserve_empty_polysemes=False,
                 baseline_model=None,
                 features_to_remove=None):
        DistinctPrototypeRefinedPolysemyModel.__init__(self, name, train_scenes, test_scenes, study_info_,
                                                       preserve_empty_polysemes=preserve_empty_polysemes,
                                                       baseline_model=baseline_model,
                                                       features_to_remove=features_to_remove)

    def modify_polysemes(self, polyseme_list):
        for poly in polyseme_list:
            poly.prototype = self.baseline_model.preposition_model_dict[poly.preposition].prototype
        return polyseme_list


class GeneratePartitionModels(GeneratePolysemeModels):

    def __init__(self, train_scenes, test_scenes, study_info_, test_prepositions=preposition_list,
                 preserve_empty_polysemes=False):
        GeneratePolysemeModels.__init__(self, train_scenes, test_scenes, study_info_,
                                        test_prepositions=test_prepositions, preserve_empty_polysemes=False)

        self.partition_model = DataPartitionPolysemyModel(train_scenes,
                                                          test_scenes, study_info_,
                                                          baseline_model=self.baseline_model,
                                                          features_to_remove=self.features_to_remove)
        self.models.append(self.partition_model)
        self.model_name_list.append(self.partition_model.name)


class GenerateAdditionalModels(GeneratePolysemeModels):
    refined_distinct_model_name = "Refined Distinct Model"
    distinct_model_name = "Distinct Prototype"
    shared_model_name = "Shared Prototype"

    distinct_refined_model_name = "Refined Distinct Model"
    shared_refined_model_name = "Refined Shared Model"

    distinct_median_model_name = "Median Distinct Model"
    shared_median_model_name = "Median Shared Model"

    baseline_model_name = "Baseline Model"
    cluster_model_name = KMeansPolysemyModel.name

    # main model we are testing
    # name of the model we want to compare with other models, and use to test particular features
    our_model_name = refined_distinct_model_name

    def __init__(self, train_scenes, test_scenes, study_info_, test_prepositions=preposition_list,
                 preserve_empty_polysemes=False):
        """Summary

        Args:
            train_scenes (TYPE): Description
            test_scenes (TYPE): Description
            study_info_ (TYPE): Description
            preserve_empty_polysemes (bool, optional): Description
            :param study_info_:

        Deleted Parameters:
            constraint_dict (None, optional): Description
        """
        GeneratePolysemeModels.__init__(self, train_scenes, test_scenes, study_info_,
                                        test_prepositions=test_prepositions,
                                        preserve_empty_polysemes=preserve_empty_polysemes)

        self.refined = DistinctPrototypeRefinedPolysemyModel(GenerateAdditionalModels.distinct_refined_model_name,
                                                             self.train_scenes,
                                                             self.test_scenes, self.study_info,
                                                             test_prepositions=self.test_prepositions,
                                                             preserve_empty_polysemes=self.preserve_empty_polysemes,
                                                             baseline_model=self.baseline_model,
                                                             features_to_remove=self.features_to_remove)

        # self.shared_refined = SharedPrototypeRefinedPolysemyModel(GenerateAdditionalModels.shared_refined_model_name,
        #                                                           self.train_scenes,
        #                                                           self.test_scenes, self.study_info,
        #                                                           preserve_empty_polysemes=self.preserve_empty_polysemes,
        #                                                           baseline_model=self.baseline_model,
        #                                                           features_to_remove=self.features_to_remove)

        self.median = DistinctPrototypeMedianPolysemyModel(GenerateAdditionalModels.distinct_median_model_name,
                                                           self.train_scenes,
                                                           self.test_scenes, self.study_info,
                                                           test_prepositions=self.test_prepositions,
                                                           preserve_empty_polysemes=self.preserve_empty_polysemes,
                                                           baseline_model=self.baseline_model,
                                                           features_to_remove=self.features_to_remove)

        # self.shared_median = copy.deepcopy(self.median)
        # self.shared_median.name = GenerateAdditionalModels.shared_median_model_name
        # self.shared_median.polyseme_dict = self.shared_median.get_shared_prototype_polyseme_dict(
        #     self.shared_median.polyseme_dict)

        self.models = [self.non_shared, self.baseline_model, self.cluster_model, self.refined, self.median]

        self.model_name_list = []
        for m in self.models:
            self.model_name_list.append(m.name)


def test_partition_model(runs, k, study):
    # First test on all scenes
    all_scenes = study.scene_name_list
    g_models = GeneratePartitionModels(all_scenes, all_scenes, study)

    models = g_models.models

    t = TestModels(models, "all")
    all_dataframe = t.score_dataframe.copy()

    all_dataframe.to_csv(extra_thesis_folder + "partition/partition_test.csv")
    print(all_dataframe)

    # Output definitions for reference
    d = g_models.partition_model.polyseme_dict
    for preposition in d:
        for polyseme in d[preposition]:
            filename = extra_thesis_folder + "partition/definitions/" + polyseme.preposition + "-" + polyseme.polyseme_name + ".csv"
            polyseme.output_definition(filename=filename)

    # Test kfold

    m = MultipleRunsGeneric(GeneratePartitionModels, extra_thesis_folder + "partition",
                            extra_thesis_folder + "partition",
                            study, number_runs=runs, k=k, compare="y")
    print(("Test Model k = " + str(k)))
    m.validation()
    m.output()
    print(m.average_dataframe)


def test_additional_models(runs, k, study_info_):
    """Summary

    Args:
        runs (TYPE): Description
        k (TYPE): Description
        study_info_ (TYPE): Description
        :param study_info_:
    """
    all_scenes = study_info_.scene_name_list
    p_models = GenerateAdditionalModels(all_scenes, all_scenes, study_info_)

    models = p_models.models

    t = TestModels(models, "all")
    all_dataframe = t.score_dataframe.copy()

    all_dataframe.to_csv(extra_thesis_folder + "refined models/all_test.csv")
    print(all_dataframe)

    m = MultipleRunsPolysemyModels(GenerateAdditionalModels, extra_thesis_folder + "refined models",
                                   extra_thesis_folder + "refined models",
                                   study_info_, number_runs=runs, k=k, compare="y")
    print(("Test Model k = " + str(k)))
    m.validation()
    m.output()
    print(m.average_dataframe)


def test_model_all_prepositions(runs, k, study_info_):
    """Summary

    Args:
        runs (TYPE): Description
        k (TYPE): Description
        study_info_ (TYPE): Description
        :param study_info_:
    """
    # First test on all scenes:

    all_scenes = study_info_.scene_name_list
    p_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info_, test_prepositions=preposition_list)

    models = p_models.models

    t = TestModels(models, "all")
    all_dataframe = t.score_dataframe.copy()

    all_dataframe.to_csv(extra_thesis_folder + "all prepositions/all_test.csv")
    print(all_dataframe)

    # Then k-fold

    m = MultipleRunsPolysemyModels(GeneratePolysemeModels, extra_thesis_folder + "all prepositions",
                                   extra_thesis_folder + "all prepositions",
                                   study_info_, number_runs=runs, k=k, compare="y", test_prepositions=preposition_list)

    print(("Test Model k = " + str(k)))
    m.validation()
    m.output()
    print((m.average_dataframe))


def output_unsatisfied_constraints():
    study_info_ = StudyInfo("2019 study")

    scene_list = study_info_.scene_name_list

    additional_models = GenerateAdditionalModels(scene_list, scene_list, study_info_)

    for model in additional_models.models:
        model.output_unsatisfied_constraints()


def output_all_polyseme_info(study_info_):
    """Summary
    :param study_info_:

    Args:
        study_info_ (TYPE): Description
    """
    print("outputting all polyseme info")
    all_scenes = study_info_.scene_name_list
    generated_polyseme_models = GenerateAdditionalModels(all_scenes, all_scenes, study_info_,
                                                         preserve_empty_polysemes=True)
    generated_polyseme_models.refined.output_polyseme_info()


if __name__ == '__main__':
    # output_unsatisfied_constraints()

    # plot_sr_typicality()

    if polysemous_preposition_list != preposition_list:
        study_info = StudyInfo("2019 study")
        # output_all_polyseme_info(study_info)

        # test_partition_model(10, 10, study_info)
        # test_model_all_prepositions(10, 10, study_info)
        #
        test_additional_models(10, 10, study_info)
    else:
        print("Edit poly preposition list")

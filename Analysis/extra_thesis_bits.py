# This is a non-exhaustive file for testing extra bits included in the thesis which haven't been formally archived.

# Things to move across
# Non-polysemous prepositions analysis
# Refined model analysis
import random, os

from polysemy_analysis import *


class SelectionRatioModel(Model):
    name = "Selection Ratio Model"

    def __init__(self, test_scenes, study_info):

        Model.__init__(self, self.name, test_scenes, study_info)

    def check_sr_exists(self, scene, figure, ground):

        # SR csv
        sv_filetag = SemanticCollection.filetag
        config_ratio_csv = self.study_info.config_ratio_csv(sv_filetag, 'in')
        sr_dataset = pd.read_csv(config_ratio_csv)

        counter = 0
        for index, row in sr_dataset.iterrows():
            if row[GeneratePrepositionModelParameters.scene_feature_name] == scene and row[
                GeneratePrepositionModelParameters.fig_feature_name] == figure and \
                    row[GeneratePrepositionModelParameters.ground_feature_name] == ground:
                counter += 1
                break
        if counter > 1:
            raise ValueError('Too many configurations found')
        elif counter == 0:
            return False
        elif counter == 1:
            return True

    def get_typicality(self, preposition, value_array, scene=None, figure=None, ground=None, study=None):

        sv_filetag = SemanticCollection.filetag
        config_ratio_csv = self.study_info.config_ratio_csv(sv_filetag, preposition)
        sr_dataset = pd.read_csv(config_ratio_csv)

        for index, row in sr_dataset.iterrows():
            if row[GeneratePrepositionModelParameters.scene_feature_name] == scene and row[
                GeneratePrepositionModelParameters.fig_feature_name] == figure and \
                    row[GeneratePrepositionModelParameters.ground_feature_name] == ground:
                sr = row[GeneratePrepositionModelParameters.ratio_feature_name]
                break
        return sr

    def get_test_constraints(self, preposition):
        allConstraints = self.constraint_dict[preposition]
        # Constraints to test on
        testConstraints = []

        for c in allConstraints:
            if c.scene in self.test_scenes:
                if self.check_sr_exists(c.scene, c.f1, c.ground):
                    if self.check_sr_exists(c.scene, c.f2, c.ground):
                        testConstraints.append(c)

        print('number of constraints:')
        print(preposition)
        print(len(testConstraints))
        return testConstraints


def initial_test(study_info_):
    """Summary

    Args:
        study_info_ (TYPE): Description
    """
    scene_list = study_info_.scene_name_list
    m = SelectionRatioModel(scene_list, study_info_)

    m.get_score()

    revised_constraint_dict = dict()
    for p in preposition_list:
        revised_constraint_dict[p] = m.get_test_constraints(p)

    features_to_remove = Configuration.ground_property_features.copy()

    preposition_models_dict = dict()

    # Get parameters for each preposition
    for p in preposition_list:
        M = GeneratePrepositionModelParameters(study_info_, p, scene_list,
                                               features_to_remove=features_to_remove)
        M.work_out_models()
        preposition_models_dict[p] = M

    p = PrototypeModel(preposition_models_dict, scene_list, study_info_, constraint_dict=revised_constraint_dict)
    p.get_score()

    print(p.scores)

    print(m.scores)

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

        new_salient_features = []
        for f in original_salient_features:
            new_f = copy.deepcopy(f)

            median = self.baseline_model.preposition_model_dict[preposition].goodAllFeatures[new_f.name].median()
            new_f.value = median

            new_salient_features.append(new_f)

        new_polysemes = self.generate_polysemes(preposition, new_salient_features)
        return new_polysemes


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
        Refines the ideal meaning by testing a validation test.
        Outputs new list of polysemes for the model
        :param preposition:
        :param original_salient_features:
        :return:
        """
        new_salient_features = []
        if preposition in polysemous_preposition_list:
            # Each of the salient features are proportions so we use these values
            # This makes generation non-deterministic for these models
            train_scenes, validation_scenes = train_test_split(self.train_scenes, test_size=0.5)
            g_values_to_try = [0.5, 0.6, 0.7, 0.8, 0.9]
            l_values_to_try = [0.1, 0.2, 0.2, 0.4, 0.5]

            for f in original_salient_features:
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

                best_value = 0
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

                if best_value == 0:
                    raise ValueError("best_value unassigned")

                # The original feature is updated, which is better for training the next feature
                f.value = best_value
                new_salient_features.append(f)
        new_polysemes = self.generate_polysemes(preposition, new_salient_features)
        return new_polysemes

    def test_ideal_feature_value(self, train_scenes, validation_scenes, preposition, original_salient_features, feature,
                                 value):
        """
        Generates new polysemes for the model from the given feature and value,
        then tests on given test scenes.
        :param train_scenes:
        :param test_scenes:
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

        allConstraints = self.constraint_dict[preposition]
        # Constraints to test on
        testConstraints = []

        for c in allConstraints:
            if c.scene in validation_scenes:
                testConstraints.append(c)

        # Get score for preposition
        score_two = self.weighted_score(testConstraints)

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
    # main model we are testing
    # name of the model we want to compare with other models, and use to test particular features

    refined_distinct_model_name = "Refined Distinct Model"
    distinct_model_name = "Distinct Prototype"
    shared_model_name = "Shared Prototype"

    distinct_refined_model_name = "Refined Distinct Model"
    shared_refined_model_name = "Refined Shared Model"

    distinct_median_model_name = "Median Distinct Model"
    shared_median_model_name = "Median Shared Model"

    baseline_model_name = "Baseline Model"
    cluster_model_name = KMeansPolysemyModel.name

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
                                        test_prepositions=test_prepositions, preserve_empty_polysemes=False)

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


class MultipleRunsGeneric(MultipleRuns):
    def __init__(self, model_generator, scores_tables_folder, scores_plots_folder, study_info_,
                 test_prepositions=preposition_list, number_runs=None,
                 k=None, compare=None):
        self.study_info = study_info_

        MultipleRuns.__init__(self, model_generator, self.study_info, test_prepositions=test_prepositions,
                              number_runs=number_runs, test_size=None, k=k,
                              compare=compare, features_to_test=None)

        self.scores_tables_folder = scores_tables_folder
        self.scores_plots_folder = scores_plots_folder
        self.get_file_strings()

    def folds_check(self, folds):
        """Summary

        Args:
            folds (TYPE): Description

        Returns:
            TYPE: Description
        """

        for f in folds:

            # Check all folds have some constraints to test
            for preposition in self.test_prepositions:

                allConstraints = self.constraint_dict[preposition]

                Constraints = []

                for c in allConstraints:
                    if c.scene in f:
                        Constraints.append(c)
                if len(Constraints) == 0:
                    return False

            if KMeansPolysemyModel.name in self.Generate_Models_all_scenes.model_name_list:
                # And also check that there are enough training samples for the K-Means model
                # in scenes not in fold
                # (samples must be greater than number of clusters..)
                scenes_not_in_fold = []
                for sc in self.study_info.scene_name_list:
                    if sc not in f:
                        scenes_not_in_fold.append(sc)
                for preposition in self.test_prepositions:
                    # Add some features to remove to ignore print out
                    prep_model = GeneratePrepositionModelParameters(self.study_info, preposition, scenes_not_in_fold,
                                                                    features_to_remove=Configuration.ground_property_features)
                    if len(prep_model.affFeatures.index) < KMeansPolysemyModel.cluster_numbers[preposition]:
                        return False

        return True


def test_partitionmodel_on_all_scenes(study_info_):
    """Summary
    :param study_info_:

    Args:
        study_info_ (TYPE): Description
    """
    print("test on all scenes")

    all_scenes = study_info_.scene_name_list
    g_models = GeneratePartitionModels(all_scenes, all_scenes, study_info_)

    models = g_models.models

    t = TestModels(models, "all")
    all_dataframe = t.score_dataframe.copy()

    all_dataframe.to_csv(study_info_.polysemy_score_folder + "partition_test.csv")
    print(all_dataframe)

    # Output definitions for reference
    d = g_models.partition_model.polyseme_dict
    for preposition in d:
        for polyseme in d[preposition]:
            polyseme.output_definition()


def test_partition_model(runs, k, study):
    m = MultipleRunsGeneric(GeneratePartitionModels, "extra thesis results/partition", "extra thesis results/partition",
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
    m = MultipleRunsGeneric(GenerateAdditionalModels, "extra thesis results/refined models",
                            "extra thesis results/refined models",
                            study_info_, number_runs=runs, k=k, compare="y")
    print(("Test Model k = " + str(k)))
    m.validation()
    m.output()
    print((m.average_dataframe))


def test_model_all_prepositions(runs, k, study_info_):
    """Summary

    Args:
        runs (TYPE): Description
        k (TYPE): Description
        study_info_ (TYPE): Description
        :param study_info_:
    """
    m = MultipleRunsPolysemyModels(study_info_, test_prepositions=preposition_list, number_runs=runs, k=k, compare="y")
    print(("Test Model k = " + str(k)))
    m.validation()
    m.output()
    print((m.average_dataframe))




if __name__ == '__main__':

    study_info = StudyInfo("2019 study")

    # K-fold for all prepositions
    if polysemous_preposition_list != preposition_list:
        # test_partitionmodel_on_all_scenes(study_info)
        test_model_all_prepositions(10, 10, study_info)
        test_partition_model(10, 10, study_info)
        test_additional_models(10, 10, study_info)
    else:
        print("Edit poly preposition list")

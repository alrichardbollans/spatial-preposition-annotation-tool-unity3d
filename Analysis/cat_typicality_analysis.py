# Compare each pair of configurations checking which better fits the category
# and which is more typical. test the significance of the differences.

# Compare how models perform on new typicality data, when trained on relational
# features only and with object-specific features.

# Check typicality agreement(think this is already done).
import csv
from itertools import combinations
import pandas as pd
import numpy as np

# first in  2019 study
from basic_model_testing import Model, GeneratePrepositionModelParameters, PrototypeModel, preposition_list
from compile_instances import SemanticCollection
from data_import import Configuration, StudyInfo
from polysemy_analysis import DistinctPrototypePolysemyModel, GeneratePolysemeModels

from process_data import ModSemanticData, TypicalityData, UserData


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
        preposition_models_dict[p] = M

    p = PrototypeModel(preposition_models_dict, scene_list, study_info_, constraint_dict=revised_constraint_dict)
    p.get_score()

    print(p.scores)

    print(m.scores)


def output_2020_study_results():
    model_study_info = StudyInfo("2019 study")
    study_info = StudyInfo("2020 study")
    # Begin by loading users
    userdata = UserData(study_info)

    ## data
    typ_data = TypicalityData(userdata)
    svmod_data = ModSemanticData(userdata)

    # Typicality model
    scene_list = model_study_info.scene_name_list
    # First generate baseline model
    preposition_models_dict = dict()

    features_to_remove = Configuration.ground_property_features.copy()
    # Get parameters for each preposition
    for p in preposition_list:
        M = GeneratePrepositionModelParameters(model_study_info, p, scene_list,
                                               features_to_remove=features_to_remove)
        preposition_models_dict[p] = M


    baseline_model = PrototypeModel(preposition_models_dict, scene_list, model_study_info)
    typ_model = DistinctPrototypePolysemyModel(GeneratePolysemeModels.distinct_model_name, scene_list,
                                               scene_list, model_study_info, baseline_model=baseline_model,
                                               features_to_remove=features_to_remove)

    # Object-specific functions
    scene_object_functions = {"sv_modtypa1": "", "sv_modtypa2": "Fixed ground. Mobile Figure",
                              "sv_modtypi1": "",
                              "sv_modtypi2": "Ground is container",
                              "sv_modtypi3": "",
                              "sv_modtypi4": "Ground is container",
                              "sv_modtypo1": "Ground is flat container",
                              "sv_modtypo2": "Ground is smaller than figure",
                              "sv_modtypo3": "Ground is convex container",
                              "sv_modtypo4": "Ground is larger than figure",
                              "sv_modtypov1": "Intermediary object between figure and ground",
                              "sv_modtypov2": "Figure is cover and ground is container",
                              "sv_modtypov3": "",
                              "sv_modtypov4": "Functional interaction",
                              "sv_modtypu1": "Ground illuminates figure",
                              "sv_modtypu2": "Intermediary object between figure and ground",
                              "sv_modtypu3": "",
                              "sv_modtypu4": "Functional interaction",

                              }
    # significance level to use
    sig_level = 0.1
    simple_config_list = typ_data.config_list

    # Get config list of tested configurations
    config_list = []
    for simple_config in simple_config_list:
        c = Configuration(simple_config.scene, simple_config.figure, simple_config.ground, study_info)
        config_list.append(c)

    config_pairs = list(combinations(config_list, 2))
    with open(
            study_info.stats_folder + "/"
            + "results.csv",
            "w",
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Preposition",
                "Configuration 1",
                "Configuration 2",
                "Typicality Scores",
                "Object-Specific Features",
                "Better Category Member",
                "More Typical Configuration"
            ]
        )

        for preposition in StudyInfo.preposition_list:

            print("Outputting results for:" + str(preposition))

            for pair in config_pairs:
                c1 = pair[0]
                c2 = pair[1]

                if not (c1.configuration_match(c2)):
                    # Only want to output categorisation info on related scenes
                    if typ_data.is_test_config(c1, preposition) and typ_data.is_test_config(c2, preposition):
                        value1_array = np.array(c1.row)
                        c1_typ = typ_model.get_typicality(preposition, value1_array, c1.scene, c1.figure, c1.ground,
                                                          study_info)
                        value2_array = np.array(c2.row)
                        c2_typ = typ_model.get_typicality(preposition, value2_array, c2.scene, c2.figure, c2.ground,
                                                          study_info)

                        typ_string = "[" + str(round(c1_typ, 4)) + "],[" + str(round(c2_typ, 4)) + "]"

                        role_string = "[" + scene_object_functions[c1.scene] + "],[" + scene_object_functions[
                            c2.scene] + "]"

                        sv_stat = svmod_data.calculate_pvalue_c1_better_than_c2(preposition, c1, c2)
                        number_of_tests1 = sv_stat[0] + sv_stat[1]
                        if number_of_tests1 != 0:

                            c1_sr = sv_stat[0] / number_of_tests1
                        else:
                            c1_sr = 0

                        number_of_tests2 = sv_stat[2] + sv_stat[3]

                        if number_of_tests2 != 0:
                            c2_sr = sv_stat[2] / number_of_tests2
                        else:
                            c2_sr = 0

                        p_value = 1
                        if c1_sr == c2_sr:
                            better_config = "None"

                        elif c1_sr > c2_sr:
                            better_config = c1.string_for_cattyp_table()
                            p_value = sv_stat[4]
                        else:
                            sv_stat = svmod_data.calculate_pvalue_c1_better_than_c2(preposition, c2, c1)
                            p_value = sv_stat[4]
                            better_config = c2.string_for_cattyp_table()
                        if p_value <= sig_level:
                            better_config += "*"

                        typ_stat = typ_data.calculate_pvalue_c1_better_than_c2(preposition, c1, c2)
                        c1_selected_over_c2 = typ_stat[1]
                        c2_selected_over_c1 = typ_stat[2]

                        typ_p_value = 1
                        if c1_selected_over_c2 == c2_selected_over_c1:
                            typ_config = "None"
                        elif c1_selected_over_c2 > c2_selected_over_c1:
                            typ_config = c1.string_for_cattyp_table()
                            typ_p_value = typ_stat[3]
                        else:
                            typ_config = c2.string_for_cattyp_table()
                            typ_stat = typ_data.calculate_pvalue_c1_better_than_c2(preposition, c2, c1)
                            typ_p_value = typ_stat[3]
                        if typ_p_value <= sig_level:
                            typ_config += "*"

                        to_write = (
                            [preposition, c1.string_for_cattyp_table(), c2.string_for_cattyp_table(), typ_string,
                             role_string, better_config, typ_config]
                        )
                        writer.writerow(to_write)

    with open(
            study_info.stats_folder + "/"
            + "monotonicity-preserving-examples.csv",
            "w",
    ) as monocsvfile:
        new_writer = csv.writer(monocsvfile)
        new_writer.writerow(
            [
                "Preposition",
                "Configuration 1",
                "Configuration 2",
                "Configurations Labelled",
                "More Typical Configuration"
            ]
        )

        for preposition in StudyInfo.preposition_list:

            print("Outputting results for:" + str(preposition))

            for pair in config_pairs:
                c1 = pair[0]
                c2 = pair[1]

                if not (c1.configuration_match(c2)):
                    # Only want to output categorisation info on related scenes
                    if typ_data.is_test_config(c1, preposition) and typ_data.is_test_config(c2, preposition):

                        sv_stat = svmod_data.calculate_pvalue_c1_better_than_c2(preposition, c1, c2)
                        number_of_tests1 = sv_stat[0] + sv_stat[1]
                        if number_of_tests1 != 0:

                            c1_sr = sv_stat[0] / number_of_tests1
                        else:
                            c1_sr = 0

                        number_of_tests2 = sv_stat[2] + sv_stat[3]

                        if number_of_tests2 != 0:
                            c2_sr = sv_stat[2] / number_of_tests2
                        else:
                            c2_sr = 0

                        same_categorisation = False
                        configs_labelled = ""
                        if c1_sr == c2_sr:
                            if c1_sr == 1:
                                configs_labelled = "Always"
                                same_categorisation = True
                            if c1_sr == 0:
                                configs_labelled = "Never"
                                same_categorisation = True

                        more_typical_config = False
                        typ_stat = typ_data.calculate_pvalue_c1_better_than_c2(preposition, c1, c2)
                        c1_selected_over_c2 = typ_stat[1]
                        c2_selected_over_c1 = typ_stat[2]

                        typ_p_value = 1
                        if c1_selected_over_c2 == c2_selected_over_c1:
                            typ_config = "None"
                        elif c1_selected_over_c2 > c2_selected_over_c1:
                            typ_config = c1.string_for_cattyp_table()
                            typ_p_value = typ_stat[3]
                        else:
                            typ_config = c2.string_for_cattyp_table()
                            typ_stat = typ_data.calculate_pvalue_c1_better_than_c2(preposition, c2, c1)
                            typ_p_value = typ_stat[3]
                        if typ_p_value <= sig_level:
                            typ_config += "*"
                            more_typical_config = True

                        if same_categorisation and more_typical_config:
                            to_write = (
                                [preposition, c1.string_for_cattyp_table(), c2.string_for_cattyp_table(),
                                 configs_labelled,
                                 typ_config]
                            )
                            new_writer.writerow(to_write)


if __name__ == '__main__':
    # study_info_ = StudyInfo("2019 study")
    #
    # initial_test(study_info_)

    output_2020_study_results()

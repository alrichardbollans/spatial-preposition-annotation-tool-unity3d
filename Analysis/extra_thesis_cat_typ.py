import csv
from itertools import combinations

import pandas as pd
import numpy as np
# Modules for plotting
import matplotlib.pyplot as plt

from scipy import stats

from basic_model_testing import Model, GeneratePrepositionModelParameters
from data_import import StudyInfo, Configuration
from compile_instances import InstanceCollection, SemanticCollection
from extra_thesis_polysemy import GenerateAdditionalModels
from polysemy_analysis import sv_filetag
from process_data import UserData, ModSemanticData, SemanticData, ComparativeData

preposition_list = StudyInfo.preposition_list


class SelectionRatioModel(Model):
    name = "Selection Ratio Model"

    def __init__(self, test_scenes, study_info):

        Model.__init__(self, self.name, test_scenes, study_info)
        # Note that if the sr is equal, the constraint will be unsatisfied.

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
        all_constraints = self.constraint_dict[preposition]
        # Constraints to test on
        test_constraints = []

        for c in all_constraints:
            if c.scene in self.test_scenes:
                if self.check_sr_exists(c.scene, c.f1, c.ground):
                    if self.check_sr_exists(c.scene, c.f2, c.ground):
                        test_constraints.append(c)

        print('number of constraints:')
        print(preposition)
        print(len(test_constraints))
        return test_constraints


class ConservativeSelectionRatioModel(SelectionRatioModel):
    # This class tests on more conservative set of constraints
    name = "Conservative Selection Ratio Model"

    def __init__(self, test_scenes, study_info):

        SelectionRatioModel.__init__(self, test_scenes, study_info)

    def get_test_constraints(self, preposition):
        all_constraints = self.constraint_dict[preposition]
        # Constraints to test on
        test_constraints = []

        for c in all_constraints:
            if c.scene in self.test_scenes:
                if self.check_sr_exists(c.scene, c.f1, c.ground):
                    if self.check_sr_exists(c.scene, c.f2, c.ground):

                        lhs = self.get_typicality_lhs(c)
                        rhs = self.get_typicality_rhs(c)
                        if lhs != rhs:
                            test_constraints.append(c)

        print('number of constraints:')
        print(preposition)
        print(len(test_constraints))
        return test_constraints


def output_unsatisfied_constraints():
    study_info = StudyInfo("2019 study")

    scene_list = study_info.scene_name_list

    # sr_model = ConservativeSelectionRatioModel(scene_list, study_info)
    #
    # sr_model.output_unsatisfied_constraints()

    sr_model = SelectionRatioModel(scene_list, study_info)

    sr_model.output_unsatisfied_constraints()


def plot_sr_typicality():
    model_study_info = StudyInfo("2019 study")

    scene_feature_name = GeneratePrepositionModelParameters.scene_feature_name
    fig_feature_name = GeneratePrepositionModelParameters.fig_feature_name
    gr_feature_name = GeneratePrepositionModelParameters.ground_feature_name

    all_scenes = model_study_info.scene_name_list
    g_models = GenerateAdditionalModels(all_scenes, all_scenes, model_study_info)

    ref_model = g_models.refined
    non_shared_model = g_models.non_shared

    plot_folder = 'extra thesis results/sr_typ_plots/'

    correlations = dict()
    p_values = dict()

    for preposition in preposition_list:
        config_ratio_csv = model_study_info.config_ratio_csv(sv_filetag, preposition)
        dataset = pd.read_csv(config_ratio_csv)

        ratio_feature_name = InstanceCollection.ratio_feature_name

        sr_list = dataset[ratio_feature_name].values.copy()

        sr_reshaped = sr_list.reshape(-1, 1)

        # Lets do both poly model and refined polymodel

        Xref = []
        Xpoly = []
        for index, row in dataset.iterrows():
            scene = row[scene_feature_name]
            fig = row[fig_feature_name]
            gr = row[gr_feature_name]
            c = Configuration(scene, fig, gr, model_study_info)
            # Typicality is calculated for each configuration
            # To check whether a configuration fits a particular polyseme we need to include
            value_array = np.array(c.row)
            typicalityref = ref_model.get_typicality(preposition, value_array, scene=c.scene, figure=c.figure,
                                                     ground=c.ground, study=model_study_info)
            Xref.append(typicalityref)

            typicalitypoly = non_shared_model.get_typicality(preposition, value_array, scene=c.scene, figure=c.figure,
                                                             ground=c.ground, study=model_study_info)
            Xpoly.append(typicalitypoly)

        fig, ax = plt.subplots()

        ax.set_xlabel("Typicality calculated by Refined Model")

        ax.set_ylabel("Selection Ratio")

        # Plot data point scatter
        ax.plot(Xref, sr_reshaped, 'k.')

        plt.savefig(plot_folder + preposition + "_xref_scatter.pdf", bbox_inches='tight')

        fig, ax = plt.subplots()

        ax.set_xlabel("Typicality calculated by Polysemy Model")

        ax.set_ylabel("Selection Ratio")

        # Plot data point scatter
        ax.plot(Xpoly, sr_reshaped, 'k.')

        plt.savefig(plot_folder + preposition + "_xnon_shared_scatter.pdf", bbox_inches='tight')

        # output csv to read
        scene_column = dataset[scene_feature_name].values
        fig_column = dataset[fig_feature_name].values
        gr_column = dataset[gr_feature_name].values

        sr_typ_df = pd.DataFrame(
            {scene_feature_name: scene_column, fig_feature_name: fig_column, gr_feature_name: gr_column,
             ratio_feature_name: sr_list, ref_model.name: Xref, non_shared_model.name: Xpoly})

        sr_typ_df.to_csv(plot_folder + preposition + "_xref_scatter.csv")

        # Calculate spearman correlation coefficient
        # Does not assume the datasets are normally distributed
        print(preposition)
        ref_correlation, ref_p_value = stats.spearmanr(Xref, sr_list)
        nonshared_correlation, nonshared_pvalue = stats.spearmanr(Xpoly, sr_list)

        correlations[preposition] = [ref_correlation, nonshared_correlation]
        p_values[preposition] = [ref_p_value, nonshared_pvalue]
    correlation_df = pd.DataFrame(correlations, index=[ref_model.name, non_shared_model.name])
    correlation_df.to_csv(plot_folder + "correlations.csv")

    pvalue_df = pd.DataFrame(p_values, index=[ref_model.name, non_shared_model.name])
    pvalue_df.to_csv(plot_folder + "pvalues.csv")


def compare_2019_cat_typ():
    results_folder = "extra thesis results/2019 study cat typ comparison"
    # 2019 study data allows us to generate a model of typicality
    model_study_info = StudyInfo("2019 study")
    study_info = StudyInfo("2019 study")
    # Begin by loading users
    userdata = UserData(study_info)

    # data
    sv_data = SemanticData(userdata)
    comparative_data = ComparativeData(userdata)

    # Typicality model
    scene_list = model_study_info.scene_name_list

    # significance level to use
    sig_level = 0.1
    simple_config_list = sv_data.config_list
    count_number_sign_pairs = 0  # NUmber of pairs where one of the configs is significantly better than the other

    # Get config list of tested configurations
    config_list = []
    for simple_config in simple_config_list:
        c = Configuration(simple_config.scene, simple_config.figure, simple_config.ground, study_info)
        config_list.append(c)

    config_pairs = list(combinations(config_list, 2))
    with open(
            results_folder + "/"
            + "results.csv",
            "w",
    ) as results_csvfile, open(
        results_folder + "/"
        + "disagreements.csv",
        "w",
    ) as disagcsvfile, open(
        results_folder + "/"
        + "monotonicity-preserving-examples.csv",
        "w",
    ) as monocsvfile:
        results_writer = csv.writer(results_csvfile)
        disag_writer = csv.writer(disagcsvfile)  # Collect disagreements
        mono_writer = csv.writer(monocsvfile)  # Monotone examples
        heading = [
            "Preposition", "Scene", "Ground",
            "Figure 1",
            "Figure 2",
            "Better Category Member",
            "More Typical Configuration"
        ]
        results_writer.writerow(heading)
        disag_writer.writerow(heading)
        mono_writer.writerow(
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
                    # If they are configrs being compared in comp task
                    if c1.scene == c2.scene and c1.ground ==c2.ground:
                        sv_stat = sv_data.calculate_pvalue_c1_better_than_c2(preposition, c1, c2)
                        times_c1_labelled = sv_stat[0]
                        times_c1_not_labelled = sv_stat[1]
                        number_of_tests1 = times_c1_labelled + times_c1_not_labelled
                        # If c1 tested at least once
                        if number_of_tests1 != 0:

                            c1_sr = times_c1_labelled / number_of_tests1

                            number_of_tests2 = sv_stat[2] + sv_stat[3]

                            # if c2 tested at least once
                            if number_of_tests2 != 0:
                                c2_sr = sv_stat[2] / number_of_tests2

                                if preposition == "in" and c1.scene=="compsvo79" and c1.ground == "table":
                                    print("####")
                                    print(times_c1_labelled)
                                    print(times_c1_not_labelled)
                                    print(c1.figure)
                                    print(c1_sr)
                                    print(c2.figure)
                                    print(c2_sr)

                                # Work out if monotone example
                                same_categorisation = False
                                configs_labelled = ""
                                if c1_sr == c2_sr:
                                    if c1_sr == 1:
                                        configs_labelled = "Always"
                                        same_categorisation = True
                                    if c1_sr == 0:
                                        configs_labelled = "Never"
                                        same_categorisation = True

                                # Calculate better config and p value
                                more_typical_config = False
                                p_value = 1
                                if c1_sr == c2_sr:
                                    better_config = "None"

                                elif c1_sr > c2_sr:
                                    better_config = c1.string_for_cattyp_table()
                                    p_value = sv_stat[4]
                                else:
                                    sv_stat = sv_data.calculate_pvalue_c1_better_than_c2(preposition, c2, c1)
                                    p_value = sv_stat[4]
                                    better_config = c2.string_for_cattyp_table()

                                typ_stat = comparative_data.calculate_pvalue_c1_better_than_c2(preposition, c1, c2)
                                typ_number_comparisons = typ_stat[0]

                                # If at least one comparison test
                                if typ_number_comparisons != 0:
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
                                        typ_stat = comparative_data.calculate_pvalue_c1_better_than_c2(preposition, c2, c1)
                                        typ_p_value = typ_stat[3]

                                    disagreement = False

                                    if better_config != "None" and typ_config != "None" and better_config != typ_config:
                                        disagreement = True

                                    if p_value <= sig_level:
                                        better_config += "*"
                                    if typ_p_value <= sig_level:
                                        typ_config += "*"
                                        more_typical_config = True

                                    if p_value <= sig_level or typ_p_value <= sig_level:
                                        count_number_sign_pairs += 1

                                    to_write = (
                                        [preposition,c1.scene,c1.ground, c1.figure, c2.figure,
                                         better_config,
                                         typ_config]
                                    )

                                    results_writer.writerow(to_write)
                                    if disagreement:
                                        disag_writer.writerow(to_write)

                                    if same_categorisation and more_typical_config:
                                        mono_write = (
                                            [preposition, c1.string_for_cattyp_table(), c2.string_for_cattyp_table(),
                                             configs_labelled,
                                             typ_config]
                                        )
                                        mono_writer.writerow(mono_write)

        results_writer.writerow(["Number of significant pairs", str(count_number_sign_pairs)])


if __name__ == '__main__':
    compare_2019_cat_typ()
    # output_unsatisfied_constraints()

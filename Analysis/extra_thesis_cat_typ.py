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

    sr_model = ConservativeSelectionRatioModel(scene_list, study_info)

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

        ax.set_xlabel("Typicality")

        ax.set_ylabel("Selection Ratio")

        # Plot data point scatter
        ax.plot(Xref, sr_reshaped, 'k.')

        plt.savefig(plot_folder + preposition + "_xref_scatter.pdf", bbox_inches='tight')

        fig, ax = plt.subplots()

        ax.set_xlabel("Typicality")

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


if __name__ == '__main__':
    output_unsatisfied_constraints()

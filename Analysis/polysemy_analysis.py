"""Summary
First run compile_instances.py
Attributes:
    comp_filetag (str): Description
    non_polysemous_prepositions (list): Description
    polysemous_preposition_list (list): Description
    preposition_list (TYPE): Description
    sv_filetag (str): Description

"""
#

# Standard imports
import copy
import pandas as pd
import numpy as np
import math
import itertools

# Ml modules

from sklearn.cluster import KMeans
from scipy.spatial.distance import minkowski
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Modules for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from basic_model_testing import TestModels, GeneratePrepositionModelParameters, Model, Features, MultipleRuns, \
    SemanticMethods, PrototypeModel
from data_import import Configuration, StudyInfo
from compile_instances import SemanticCollection, ComparativeCollection

# Useful global variables
sv_filetag = SemanticCollection.filetag  # Tag for sv task files
comp_filetag = ComparativeCollection.filetag  # Tag for comp task files
preposition_list = StudyInfo.preposition_list
polysemous_preposition_list = ['in', 'on', 'under', 'over']  # list of prepositions which exist in the data
non_polysemous_prepositions = ["inside", "above", "below", "on top of", 'against']


class Cluster:
    """Summary
    
    Attributes:
        hr_means (TYPE): Description
        hr_means_csv (TYPE): Description
        instances (TYPE): Description
        instances_csv (TYPE): Description
        label (TYPE): Description
        mean_csv (TYPE): Description
        mean_series (TYPE): Description
        means (TYPE): Description
        preposition (TYPE): Description
        study_info (TYPE): Description
    """

    def __init__(self, study_info_, preposition, instances, label, alg_typ=None):
        """Summary
        
        Args:
            study_info_ (TYPE): Description
            preposition (TYPE): Description
            instances (TYPE): Description
            label (TYPE): Description
            alg_typ (None, optional): Description
        """
        self.study_info = study_info_
        self.preposition = preposition
        self.label = label
        self.instances = instances
        self.mean_series = self.instances.mean()
        # Convert series to appropriate dataframe
        self.means = pd.DataFrame({'values': self.mean_series.values})
        self.means = self.means.set_index(self.mean_series.index)
        self.means = self.means.transpose()

        feature_processer = Features(self.study_info.name)
        self.hr_means = feature_processer.convert_standard_df_to_normal(self.means)

        if alg_typ == "kmeans":
            base_folder = self.study_info.kmeans_folder
        elif alg_typ == "hry":
            base_folder = self.study_info.hry_folder
        else:
            print("Error: No cluster type given")

        self.mean_csv = base_folder + "cluster means/clusters-" + preposition + str(self.label) + ".csv"
        self.instances_csv = base_folder + "cluster instances/instances-" + preposition + str(self.label) + ".csv"

        self.hr_means_csv = base_folder + "cluster means/human readable/instances-" + preposition + str(
            self.label) + ".csv"

    def output(self):
        """Summary
        """
        # self.mean_series["cardinality"] = len(self.instances.index)
        self.means.to_csv(self.mean_csv)
        self.instances.to_csv(self.instances_csv)

        self.hr_means.to_csv(self.hr_means_csv)


class Clustering:
    """Summary
    
    Attributes:
        all_scenes (TYPE): Description
        cluster_centres_csv (TYPE): Description
        dendrogram_pdf (TYPE): Description
        elbow_pdf (TYPE): Description
        good_dataset (TYPE): Description
        good_instance_csv (TYPE): Description
        good_instance_features (TYPE): Description
        good_instances_all_features (TYPE): Description
        hr_good_instance_csv (TYPE): Description
        hr_good_instances (TYPE): Description
        initial_inertia_csv (TYPE): Description
        good_instances_to_cluster (TYPE): Description
        km_instances_to_cluster (TYPE): Description
        models (TYPE): Description
        possible_instances_all_features (TYPE): Description
        possible_instances_features (TYPE): Description
        preposition (TYPE): Description
        relation_weights (TYPE): Description
        sample_weights (TYPE): Description
        study_info (TYPE): Description
        typical_instances (TYPE): Description
    """

    # Number of clusters created by inspecting HAC dendograms
    cluster_numbers = {'on': 3, 'in': 2, 'against': 3, 'under': 3, 'over': 2}

    def __init__(self, study_info_, preposition):
        """Summary
        
        Args:
            study_info_ (TYPE): Description
            preposition (TYPE): Description
        """
        self.study_info = study_info_

        self.all_scenes = self.study_info.scene_name_list
        self.preposition = preposition
        self.models = GeneratePrepositionModelParameters(self.study_info, preposition, self.all_scenes,
                                                         features_to_remove=Configuration.ground_property_features.copy())

        # All selected instances
        self.possible_instances_all_features = self.models.affAllFeatures
        self.possible_instances_features = self.models.remove_unused_features(self.possible_instances_all_features)
        # Dataset containing 'good' instances
        self.good_dataset = self.models.train_dataset[
            (self.models.train_dataset.iloc[:, self.models.ratio_index] >= 0.5)]
        # Reindex df for later readability
        self.good_dataset = self.good_dataset.reset_index(drop=True)
        # All 'good' instances
        self.good_instances_all_features = self.models.remove_nonfeatures(self.good_dataset)

        self.good_instance_features = self.models.remove_unused_features(self.good_instances_all_features)

        self.typical_instances = self.models.typical_features

        # self.models.all_feature_weights()
        # self.feature_weights= self.models.read_all_feature_weights()
        self.relation_weights = self.models.read_regression_weights()

        self.cluster_centres_csv = self.study_info.kmeans_folder + "cluster centres/clusters-" + preposition + ".csv"
        self.dendrogram_pdf = self.study_info.hry_folder + "figures/dendrogram/dendrogram-" + preposition + ".pdf"
        self.elbow_pdf = self.study_info.kmeans_folder + "figures/elbow/" + preposition + ".pdf"
        self.initial_inertia_csv = self.study_info.kmeans_folder + "initial inertia/initial_inertias.csv"
        # The dataframe we use for clustering
        self.good_instances_to_cluster = self.good_instance_features.copy()
        self.km_instances_to_cluster = self.possible_instances_features.copy()
        # Samples are weighted by selection ratio
        self.sample_weights = self.models.aff_dataset[
            self.models.ratio_feature_name]  # self.good_dataset[self.models.ratio_feature_name]
        # Output good instances to read
        self.good_instance_csv = self.study_info.cluster_data_folder + "good preposition instances/good instances - " + self.preposition + ".csv"
        self.good_instances_to_cluster.to_csv(self.good_instance_csv)

        self.feature_processer = Features(self.study_info.name)
        self.hr_good_instance_csv = self.study_info.cluster_data_folder + "good preposition instances/human readable/good instances - " + self.preposition + ".csv"

        self.hr_good_instances = self.feature_processer.convert_standard_df_to_normal(self.good_instances_to_cluster)

        self.hr_good_instances.to_csv(self.hr_good_instance_csv)

    def custom_metric(self, u, v):
        """Summary
        
        Args:
            u (TYPE): Description
            v (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # weighted euclidean distance. Also weight by instances somehow?
        return minkowski(u, v, p=2, w=self.relation_weights.values)

    def work_out_hierarchy_model(self):
        """Summary
        """
        instances = self.good_instances_to_cluster
        print(instances)
        Z = linkage(instances, method='single', optimal_ordering=True)  # ,metric=self.custom_metric

        fig = plt.figure()
        fig.canvas.set_window_title(self.preposition)
        # fig.suptitle("Dendrogram for '"+ self.preposition+"'", fontsize=20)

        # factor = self.color_threshold[self.preposition]
        if self.preposition == "on":
            thresh = 0.65 * max(Z[:, 2])
        else:
            thresh = 0.7 * max(Z[:, 2])  # Default threshold
        dendrogram(Z, color_threshold=thresh)

        plt.savefig(self.dendrogram_pdf, bbox_inches='tight')

        # Form flat clusters based on threshold
        clusters = fcluster(Z, criterion='distance', t=thresh)

        done_clusters = []
        for c in clusters:
            if c not in done_clusters:
                done_clusters.append(c)
                cluster_instances_index = []
                for i in range(len(clusters)):
                    if c == clusters[i]:
                        cluster_instances_index.append(i)
                cluster_instances = instances.iloc[cluster_instances_index, :]
                cluster = Cluster(self.study_info, self.preposition, cluster_instances, c, alg_typ="hry")
                cluster.output()
        print(("Number of clusters: " + str(len(done_clusters))))

    # if self.preposition == "on":
    # 	print(instances.iloc[25,:])
    # 	print(instances.iloc[26,:])
    # 	print("#")
    # 	print(instances.iloc[20,:])
    # 	print(instances.iloc[21,:])
    # 	print(instances.iloc[12,:])
    # 	plt.show()

    def work_out_kmeans_model(self, k):
        """Summary
        
        Args:
            k (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        number_clusters = k

        # nparray = nparray.reshape(-1,1)
        # set random state to make randomness deterministic
        km = KMeans(
            n_clusters=number_clusters, random_state=0

        )
        km.fit(self.km_instances_to_cluster, sample_weight=self.sample_weights)

        return km

    def output_cluster_info(self, km):
        """Summary
        
        Args:
            km (TYPE): Description
        """
        out = dict()

        for i in range(len(km.cluster_centers_)):
            out["cluster_" + str(i)] = km.cluster_centers_[i]

        df = pd.DataFrame(out, self.study_info.all_feature_keys)
        print(self.preposition)
        print(df)

        df.to_csv(self.cluster_centres_csv)

        k = self.cluster_numbers[self.preposition]
        for i in range(0, k):
            instances = self.km_instances_to_cluster[km.labels_ == i]
            cluster = Cluster(self.study_info, self.preposition, instances, i, alg_typ="kmeans")
            cluster.output()

    def output_expected_kmeans_model(self):
        """Summary
        """
        k = self.cluster_numbers[self.preposition]
        kmodel = self.work_out_kmeans_model(k)

        self.output_cluster_info(kmodel)

    def check_inertia_calculation(self):
        """Summary
        """
        # Checks inertia calculation agrees with KMeans method
        km = self.work_out_kmeans_model(4)
        print("Our inertia")
        i = self.calculate_inertia_from_centres(km.cluster_centers_)
        print(i)
        print("proper inertia")
        print((km.inertia_))

    def distance_to_centre_squared(self, point, centre):
        """Summary
        
        Args:
            point (TYPE): Description
            centre (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # Subtract arrays point wise
        point = np.subtract(point, centre)
        # Square pointwise
        point = np.square(point)
        # Dot product pointwise by weights
        summ = np.sum(point)

        # Square root to get distance
        distance = math.sqrt(summ)

        d2 = distance * distance

        return d2

    def calculate_inertia_from_centres(self, centres):
        """Summary
        
        Args:
            centres (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        total_sum = 0
        # As per the kmeans source code, sample weights are scaled so average weight is 1
        weight_scaling_factor = len(self.sample_weights) / self.sample_weights.sum()

        for index, row in self.km_instances_to_cluster.iterrows():

            sample = row.values
            distance = -1
            closest_centre = 0
            for centre in centres:
                d = self.distance_to_centre_squared(sample, centre)
                if distance == -1:
                    distance = d
                    closest_centre = centre
                elif d < distance:
                    distance = d
                    closest_centre = centre

            weight = self.sample_weights[index]
            normalised_weight = weight * weight_scaling_factor
            weighted_distance = distance * normalised_weight
            total_sum += weighted_distance

        return total_sum

    def calculate_polysemes_inertia(self, polysemes):
        """Summary
        
        Args:
            polysemes (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        init = []
        centres = []
        for polyseme in polysemes:
            df = pd.read_csv(polyseme.mean_csv, index_col=0, names=["feature", "value"])

            centres.append(df["value"].values)

        i = self.calculate_inertia_from_centres(centres)
        print(self.preposition)
        print(("Number of clusters:" + str(len(centres))))
        return i

    def plot_elbow_polyseme_inertia(self):
        """Summary
        """
        all_scenes = self.study_info.scene_name_list
        generated_polyseme_models = GeneratePolysemeModels(all_scenes, all_scenes,
                                                           self.study_info, preserve_empty_polysemes=True)

        polysemes = generated_polyseme_models.non_shared.polyseme_dict[self.preposition]
        polysemes_inertia = self.calculate_polysemes_inertia(polysemes)

        inertias = []
        K = list(range(1, 10))
        for k in K:
            kmeanModel = self.work_out_kmeans_model(k)

            inertias.append(kmeanModel.inertia_)

        fig, axes = plt.subplots()

        axes.plot([len(polysemes)], [polysemes_inertia], markersize=15, markeredgewidth=3, linestyle='None',
                  marker=(5, 2), label="Polysemes")

        # Plot the elbow
        axes.plot(K, inertias, 'bx-', label="K-Means")

        # plt.annotate('This is awesome!',
        #              xy=(len(polysemes), polysemes_inertia),
        #              xycoords='data',
        #              textcoords='offset points',
        #              arrowprops=dict(arrowstyle="->"))
        # axes.annotate('Polysemy Inertia', xy=(len(polysemes), polysemes_inertia),  xycoords='data',
        #             xytext=(len(polysemes)-3, polysemes_inertia+15),

        #             horizontalalignment='left', verticalalignment='bottom',
        #             )
        axes.set_xlabel('Number of Clusters')
        axes.set_ylabel('Inertia')
        axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # axes.set_title('Inertia from Kmeans Clusters - ' + self.preposition, pad=10)

        plt.legend(loc='upper right')

        plt.savefig(self.elbow_pdf, bbox_inches='tight')

    # plt.close()

    def output_initial_inertia(self):
        """Summary
        """
        kmeanModel = self.work_out_kmeans_model(1)
        inertia = kmeanModel.inertia_
        normalised_inertia = inertia / len(self.km_instances_to_cluster.index)
        new_csv = False
        try:

            in_df = pd.read_csv(self.initial_inertia_csv, index_col=0)

        except Exception as e:
            in_df = pd.DataFrame(columns=['preposition', 'inertia', 'divided by number of instances'])
            # print("unsusccefully read")
            new_csv = True

        finally:

            row_index_in_df = in_df[in_df['preposition'] == self.preposition].index.tolist()

            if len(row_index_in_df) == 0:
                in_df = in_df.append({'preposition': self.preposition, 'inertia': inertia,
                                      'divided by number of instances': normalised_inertia}, ignore_index=True)
            else:

                in_df.at[row_index_in_df[0], 'inertia'] = inertia
                in_df.at[row_index_in_df[0], 'divided by number of instances'] = normalised_inertia

            in_df.to_csv(self.initial_inertia_csv)


class ClusterInModel:
    """Summary
    
    Attributes:
        centre (TYPE): Description
        preposition (TYPE): Description
        rank (TYPE): Description
        weights (TYPE): Description
    """

    def __init__(self, preposition, centre, weights, rank):
        """Summary
        
        Args:
            preposition (TYPE): Description
            centre (TYPE): Description
            weights (TYPE): Description
            rank (TYPE): Description
        """
        self.preposition = preposition
        self.centre = centre
        self.weights = weights
        self.rank = rank


class Polyseme:
    """Summary
    Polyseme is defined by feature values: a configuration can be a polyseme instance if certain
    conditions on the feature values are satisfied.
    This class uses GeneratePrepositionModelParameters to find feature weights and prototypes for the polyseme.
    Attributes:
        annotation_csv (TYPE): Description
        eq_feature_dict (TYPE): Description
        greater_feature_dict (TYPE): Description
        less_feature_dict (TYPE): Description
        mean_csv (TYPE): Description
        number_of_instances (TYPE): Description
        plot_folder (TYPE): Description
        polyseme_name (TYPE): Description
        preposition (TYPE): Description
        preposition_models (TYPE): Description
        prototype (TYPE): Description
        prototype_csv (TYPE): Description
        rank (int): Description
        regression_weights_csv (TYPE): Description
        study_info (TYPE): Description
        train_scenes (TYPE): Description
        weights (TYPE): Description
    """

    def __init__(self, study_info_, preposition, polyseme_name, train_scenes, eq_feature_dict=None,
                 greater_feature_dict=None, less_feature_dict=None, features_to_remove=None):
        """Summary
        
        Args:
            study_info_ (TYPE): Description
            preposition (TYPE): Description
            polyseme_name (TYPE): Description
            train_scenes (TYPE): Description
            eq_feature_dict (None, optional): Description
            greater_feature_dict (None, optional): Description
            less_feature_dict (None, optional): Description

            :param study_info_:
        """
        self.study_info = study_info_
        self.polyseme_name = polyseme_name
        self.preposition = preposition
        self.train_scenes = train_scenes
        self.features_to_remove = features_to_remove

        # Dictionary containing distinguishing features and their values
        self.eq_feature_dict = eq_feature_dict
        self.greater_feature_dict = greater_feature_dict
        self.less_feature_dict = less_feature_dict

        self.annotation_csv = self.study_info.polyseme_data_folder + 'annotations/' + self.preposition + "-" + self.polyseme_name + ' .csv'
        self.prototype_csv = self.study_info.polyseme_data_folder + 'prototypes/' + self.preposition + "-" + self.polyseme_name + ' .csv'
        self.mean_csv = self.study_info.polyseme_data_folder + 'means/' + self.preposition + "-" + self.polyseme_name + ' .csv'

        self.regression_weights_csv = self.study_info.polyseme_data_folder + 'regression weights/' + self.preposition + "-" + self.polyseme_name + ' .csv'
        self.plot_folder = self.study_info.polyseme_data_folder + 'plots/'

        self.preposition_models = GeneratePrepositionModelParameters(self.study_info, self.preposition,
                                                                     self.train_scenes,
                                                                     features_to_remove=self.features_to_remove,
                                                                     polyseme=self)

        # Assign a rank/hierarchy to polysemes

        self.rank = self.get_rank()
        # Number of configurations fitting polysemes which were labelled as preposition by any participant
        self.number_of_instances = self.get_number_of_instances()

        self.weights = self.preposition_models.regression_weights
        self.prototype = self.preposition_models.prototype

    def potential_instance(self, scene, figure, ground):
        """Summary
        Checks if configuration could be a possible polyseme instance.
        Args:
            scene (TYPE): Description
            figure (TYPE): Description
            ground (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # boolean checks whether the configuration could be an instance

        r = Configuration(scene, figure, ground, self.study_info)

        r.load_from_csv()
        if self.eq_feature_dict != None:
            for feature in self.eq_feature_dict:
                value = round(r.set_of_features[feature], 6)
                condition = round(self.eq_feature_dict[feature], 6)
                if value != condition:
                    return False

        if self.greater_feature_dict != None:
            for feature in self.greater_feature_dict:

                if r.set_of_features[feature] < self.greater_feature_dict[feature]:
                    return False
        if self.less_feature_dict != None:
            for feature in self.less_feature_dict:
                if r.set_of_features[feature] > self.less_feature_dict[feature]:
                    return False
        return True

    def get_number_of_instances(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        return len(self.preposition_models.aff_dataset.index)

    def get_rank(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        ratio_feature_name = self.preposition_models.ratio_feature_name

        mean = self.preposition_models.train_possible_instances_dataset.mean(axis=0)[ratio_feature_name]

        self.rank = mean
        if np.isnan(self.rank):
            self.rank = 0

        return self.rank

    def plot(self):
        """Summary
        """
        self.preposition_models.plot_models()

    def output_prototype_weight(self):
        """Summary
        """
        pf = pd.DataFrame(self.prototype, self.study_info.all_feature_keys)

        pf.to_csv(self.prototype_csv)

        wf = pd.DataFrame(self.weights, self.study_info.all_feature_keys)

        wf.to_csv(self.regression_weights_csv)

    def output_definition(self):
        """Summary
        """
        out = dict()
        out["eq_feature_dict"] = []
        out["greater_feature_dict"] = []
        out["less_feature_dict"] = []
        for feature in self.study_info.all_feature_keys:

            if self.eq_feature_dict != None:
                if feature in self.eq_feature_dict:

                    out["eq_feature_dict"].append(round(self.eq_feature_dict[feature], 6))
                else:
                    out["eq_feature_dict"].append("None")
            else:
                out["eq_feature_dict"].append("None")

            if self.greater_feature_dict != None:
                if feature in self.greater_feature_dict:

                    out["greater_feature_dict"].append(round(self.greater_feature_dict[feature], 6))
                else:
                    out["greater_feature_dict"].append("None")
            else:
                out["greater_feature_dict"].append("None")

            if self.less_feature_dict != None:
                if feature in self.less_feature_dict:

                    out["less_feature_dict"].append(round(self.less_feature_dict[feature], 6))
                else:
                    out["less_feature_dict"].append("None")
            else:
                out["less_feature_dict"].append("None")

        wf = pd.DataFrame(out, self.study_info.all_feature_keys)  # ["equality", "greater than", "less than"])

        wf.to_csv(
            self.study_info.polyseme_data_folder + "/definitions/" + self.preposition + "-" + self.polyseme_name + ".csv")


class PolysemyModel(Model):
    """Summary
    
    Attributes:
        test_prepositions (TYPE): Description
    """

    # Puts together preposition models and has various functions for testing
    def __init__(self, name, test_scenes, study_info_):

        """Summary
        
        Args:
            name (TYPE): Description
            train_scenes (TYPE): Description
            test_scenes (TYPE): Description
            study_info_ (TYPE): Description
            weight_dict (None, optional): Description
            polyseme_dict (None, optional): Description
            cluster_dict (None, optional): Description
        

        """

        Model.__init__(self, name, test_scenes, study_info_)
        self.test_prepositions = polysemous_preposition_list

    def get_typicality(self, preposition, point, scene=None, figure=None, ground=None):
        print("This shouldn't be called")

    def weighted_score(self, preposition, Constraints):
        """Summary

        Args:
            preposition (TYPE): Description
            Constraints (TYPE): Description

        Returns:
            TYPE: Description
        """
        # Calculates how well W and P satisfy the constraints, accounting for constraint weight
        counter = 0

        for c in Constraints:
            lhs = self.get_typicality(preposition, c.lhs_values)
            rhs = self.get_typicality(preposition, c.rhs_values)
            if c.is_satisfied(lhs, rhs):
                counter += c.weight

        return counter


class PrototypePolysemyModel(PolysemyModel):

    def __init__(self, name, train_scenes, test_scenes, study_info_, preserve_empty_polysemes=False,
                 baseline_model=None, features_to_remove=None):
        PolysemyModel.__init__(self, name, test_scenes, study_info_)
        self.baseline_model = baseline_model
        self.train_scenes = train_scenes
        self.features_to_remove = features_to_remove
        # When empty polysemes are preserved their values are generated as normal
        # e.g. rank,numebr of instances  = 0. THis is useful for outputting data on the polysemes
        # When empty polysemes are not preserved, empty polysemes are assigned values from the baseline model.

        self.preserve_empty_polysemes = preserve_empty_polysemes
        # Dictionary of polysemes for each preposition
        # Non-shared polysemes don't share the prototype
        self.non_shared_dict = self.get_non_shared_prototype_polyseme_dict()
        # Shared polysemes do share the prototype
        self.shared_dict = self.get_shared_prototype_polyseme_dict()
        if name == "Distinct Prototype":

            self.polyseme_dict = self.non_shared_dict
        elif name == "Shared Prototype":
            self.polyseme_dict = self.shared_dict
        else:
            raise ValueError("Incorrectly assigned model name")

    def get_shared_prototype_polyseme_dict(self):
        """Summary
        Gets polyseme dictionary from existing dictionary but makes each polyseme share the prototype.
        Returns:
            TYPE: Description
        """
        out = dict()
        old_dict = self.non_shared_dict

        for preposition in old_dict:
            out[preposition] = []
            for polyseme in old_dict[preposition]:
                new_pol = copy.deepcopy(polyseme)
                new_pol.prototype = self.baseline_model.preposition_model_dict[preposition].prototype

                out[preposition].append(new_pol)

        return out

    def generate_polysemes(self, preposition, salient_features):
        """Summary
        Generates polysemes based on ideal meaning discussion. Uses salient features and their threshold values.
        Args:
            preposition (TYPE): Description
            salient_features (TYPE): Description

        Returns:
            TYPE: Description
        """

        polysemes = []

        g_dict = dict()
        l_dict = dict()
        for f in salient_features:
            if f.gorl == "l":
                l_dict[f.feature] = f.value
            else:
                g_dict[f.feature] = f.value

        # Canon

        p1 = Polyseme(self.study_info, preposition, "canon", self.train_scenes, greater_feature_dict=g_dict,
                      less_feature_dict=l_dict, features_to_remove=self.features_to_remove)
        polysemes.append(p1)

        # Nearly canon
        x = len(salient_features) - 1
        while x >= 0:

            name_count = 0
            for pair in list(itertools.combinations(salient_features, x)):
                name_count += 1
                g_feature_dict = dict()
                l_feature_dict = dict()

                for f in salient_features:

                    if f not in pair:

                        if f.gorl == "l":
                            g_feature_dict[f.feature] = f.value
                        else:
                            l_feature_dict[f.feature] = f.value
                    if f in pair:

                        if f.gorl == "l":
                            l_feature_dict[f.feature] = f.value
                        else:
                            g_feature_dict[f.feature] = f.value
                if x == 0:
                    p_name = "far" + str(name_count)
                elif x == len(salient_features) - 1:
                    p_name = "near" + str(name_count)
                else:
                    p_name = "not far" + str(name_count)
                ply = Polyseme(self.study_info, preposition, p_name, self.train_scenes,
                               greater_feature_dict=g_feature_dict, less_feature_dict=l_feature_dict,
                               features_to_remove=self.features_to_remove)
                polysemes.append(ply)
            x = x - 1
        return polysemes

    def get_non_shared_prototype_polyseme_dict(self):
        """Summary

        Returns:
            TYPE: Description
        """
        out = dict()

        contact03 = self.feature_processer.convert_normal_value_to_standardised("contact_proportion", 0.3)

        above09 = self.feature_processer.convert_normal_value_to_standardised("above_proportion", 0.9)
        above07 = self.feature_processer.convert_normal_value_to_standardised("above_proportion", 0.7)

        sup09 = self.feature_processer.convert_normal_value_to_standardised("support", 0.9)
        b07 = self.feature_processer.convert_normal_value_to_standardised("bbox_overlap_proportion", 0.7)
        lc075 = self.feature_processer.convert_normal_value_to_standardised("location_control", 0.75)
        gf09 = self.feature_processer.convert_normal_value_to_standardised("g_covers_f", 0.9)
        bl09 = self.feature_processer.convert_normal_value_to_standardised("below_proportion", 0.9)
        fg09 = self.feature_processer.convert_normal_value_to_standardised("f_covers_g", 0.9)

        # On

        f1 = SalientFeature("above_proportion", above09, "g")
        f2 = SalientFeature("support", sup09, "g")
        f3 = SalientFeature("contact_proportion", contact03, "g")
        on_salient_features = [f1, f2, f3]
        out["on"] = self.generate_polysemes("on", on_salient_features)

        # In
        f1 = SalientFeature("bbox_overlap_proportion", b07, "g")
        f2 = SalientFeature("location_control", lc075, "g")

        in_salient_features = [f1, f2]

        out["in"] = self.generate_polysemes("in", in_salient_features)

        # Under
        f1 = SalientFeature("g_covers_f", gf09, "g")
        f2 = SalientFeature("below_proportion", bl09, "g")

        under_salient_features = [f1, f2]

        out["under"] = self.generate_polysemes("under", under_salient_features)

        # Over
        f1 = SalientFeature("f_covers_g", fg09, "g")
        f2 = SalientFeature("above_proportion", above07, "g")

        over_salient_features = [f1, f2]

        out["over"] = self.generate_polysemes("over", over_salient_features)

        if self.preserve_empty_polysemes:
            pass
        else:
            for prep in out:

                for poly in out[prep]:

                    if poly.number_of_instances == 0:
                        # In the case there are no training instances (rank=0)
                        # Set the general parameters

                        poly.weights = self.baseline_model.preposition_model_dict[prep].regression_weights
                        poly.prototype = self.baseline_model.preposition_model_dict[prep].prototype

                        ratio_feature_name = GeneratePrepositionModelParameters.ratio_feature_name

                        poly.rank = self.baseline_model.preposition_model_dict[prep].aff_dataset.mean(axis=0)[
                            ratio_feature_name]

                        if np.isnan(poly.rank):
                            poly.rank = 0

        return out

    def get_possible_polysemes(self, preposition, scene, figure, ground):
        """Summary

        Args:
            preposition (TYPE): Description
            scene (TYPE): Description
            figure (TYPE): Description
            ground (TYPE): Description

        Returns:
            TYPE: Description
        """
        out = []

        for polyseme in self.polyseme_dict[preposition]:
            if polyseme.potential_instance(scene, figure, ground):
                out.append(polyseme)
        return out

    def get_typicality(self, preposition, point, scene=None, figure=None, ground=None):
        '''
        Finds similarity to possible polysemes and multiplies by polyseme rank.
        :param preposition:
        :param point:
        :param scene:
        :param figure:
        :param ground:
        :return:
        '''
        out = 0
        pps = self.get_possible_polysemes(preposition, scene, figure, ground)
        if len(pps) == 0:
            print("Error: No polyseme given for:")
            print(preposition)
            print(scene)
            print(figure)
            print(ground)

        for polyseme in pps:

            prototype_array = polyseme.prototype
            weight_array = polyseme.weights
            new = SemanticMethods.semantic_similarity(weight_array, point, prototype_array)

            new = new * polyseme.rank

            if new > out:
                out = new

        return out

    def weighted_score(self, preposition, Constraints):
        """Summary

        Args:
            preposition (TYPE): Description
            Constraints (TYPE): Description

        Returns:
            TYPE: Description
        """
        # Calculates how well W and P satisfy the constraints, accounting for constraint weight
        counter = 0

        for c in Constraints:
            lhs = self.get_typicality(preposition, c.lhs_values, c.scene, c.f1, c.ground)
            rhs = self.get_typicality(preposition, c.rhs_values, c.scene, c.f2, c.ground)
            if c.is_satisfied(lhs, rhs):
                counter += c.weight

        return counter


class KMeansPolysemyModel(PolysemyModel):
    name = "KMeans Model"
    cluster_numbers = {'on': 8, 'in': 4, 'under': 4, 'over': 4}

    def __init__(self, preposition_model_dict, test_scenes, study_info_):
        PolysemyModel.__init__(self, KMeansPolysemyModel.name, test_scenes, study_info_)

        self.preposition_model_dict = preposition_model_dict
        self.cluster_dict = self.get_cluster_dict()

    def get_cluster_dict(self):
        """Summary

        Returns:
            TYPE: Description
        """
        # Number of non-empty polysemes from polysemy model for all scenes
        # Actually none are empty, even for on

        out = dict()

        for preposition in polysemous_preposition_list:
            out[preposition] = []

            p_model_parameters = self.preposition_model_dict[preposition]

            # All selected instances
            possible_instances = p_model_parameters.affFeatures

            ratio_feature_name = p_model_parameters.ratio_feature_name
            sample_weights = p_model_parameters.aff_dataset[ratio_feature_name]

            # Issue that sometimes there's more samples than clusters
            # Set random state to make randomness deterministic for repeatability
            km = KMeans(
                n_clusters=self.cluster_numbers[preposition], random_state=0

            )
            km.fit(possible_instances, sample_weight=sample_weights)

            # Work out cluster ranks
            # Group configurations by their closest cluster centre.
            # THen find the average selectionratio for each group.

            weights_used_features = p_model_parameters.regression_weights_used_features

            cluster_ratio_sums = []
            cluster_number_of_instances = []
            for i in range(len(km.cluster_centers_)):
                cluster_ratio_sums.append(0)
                cluster_number_of_instances.append(0)

            for index, row in p_model_parameters.feature_dataframe.iterrows():
                # For each configuration add ratio to totals of closest centre

                # Note dropping columns from dataset preserves row order i.e.
                # row order of feature_dataframe = train_datset
                ratio_of_instance = p_model_parameters.train_dataset.at[index, ratio_feature_name]

                v = row.values
                # Convert values to np array
                v = np.array(v)

                sem_distance = -1
                chosen_centre = 0
                chosen_index = -1
                # Get closest centre
                for i in range(len(km.cluster_centers_)):

                    centre = km.cluster_centers_[i]

                    distance = SemanticMethods.semantic_distance(weights_used_features, v, centre)

                    if sem_distance == -1:
                        sem_distance = distance
                        chosen_centre = centre
                        chosen_index = i
                    elif distance < sem_distance:
                        sem_distance = distance
                        chosen_centre = centre
                        chosen_index = i
                # Update sums

                cluster_ratio_sums[chosen_index] += ratio_of_instance
                cluster_number_of_instances[chosen_index] += 1

            # Add clusters to dictionary.
            for i in range(len(km.cluster_centers_)):
                if cluster_number_of_instances[i] != 0:
                    rank = cluster_ratio_sums[i] / cluster_number_of_instances[i]
                else:
                    rank = 0

                new_c = ClusterInModel(preposition, km.cluster_centers_[i], weights_used_features, rank)
                out[preposition].append(new_c)

        return out

    def get_typicality(self, preposition, point, scene=None, figure=None, ground=None):
        """Summary
        # Finds most similar cluster centre to point. Multiplies similarity to that cluster by cluster rank
        Args:
            preposition (TYPE): Description
            point (TYPE): Description

        Returns:
            TYPE: Description
        """

        clusters = self.cluster_dict[preposition]
        # Weight array uses weights assigned to baseline model
        # Same weights for all clusters for given preposition
        weight_array = clusters[0].weights
        closest_centre_typicality = 0
        closest_cluster = 0
        # Unused features must be removed here as weight and prototype array don't account for them.
        new_point = self.preposition_model_dict[preposition].remove_unused_features_from_array(point)
        for cluster in clusters:
            prototype_array = cluster.centre

            new = SemanticMethods.semantic_similarity(weight_array, new_point, prototype_array)
            if new > closest_centre_typicality:
                closest_centre_typicality = new
                closest_cluster = cluster

        out = closest_centre_typicality * closest_cluster.rank

        return out


class SalientFeature:
    """Summary
    
    Attributes:
        feature (TYPE): Description
        gorl (TYPE): Description
        value (TYPE): Description
    """

    def __init__(self, feature, value, gorl):
        """Summary
        
        Args:
            feature (TYPE): Description
            value (TYPE): Description
            gorl (TYPE): Description
        """
        self.feature = feature
        self.value = value
        self.gorl = gorl


class GeneratePolysemeModels:
    """Summary
    
    Attributes:
        baseline_model (TYPE): Description
        cluster_model (TYPE): Description
        models (TYPE): Description
        non_shared (TYPE): Description
        preserve_empty_polysemes (TYPE): Description
        shared (TYPE): Description
        study_info (TYPE): Description
        test_scenes (TYPE): Description
        train_scenes (TYPE): Description
    """

    our_model_name = "Distinct Prototype"

    other_model_name = "Shared Prototype"
    baseline_model_name = "Baseline Model"
    cluster_model_name = KMeansPolysemyModel.name

    # List of all model names
    model_name_list = [our_model_name, other_model_name, baseline_model_name, cluster_model_name]

    # List of model names except ours
    other_name_list = [other_model_name, baseline_model_name, cluster_model_name]

    def __init__(self, train_scenes, test_scenes, study_info_, preserve_empty_polysemes=False):
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

        self.study_info = study_info_
        # Scenes used to train models
        self.train_scenes = train_scenes
        # Scenes used to test models
        self.test_scenes = test_scenes
        self.features_to_remove = Configuration.ground_property_features.copy()
        # When empty polysemes are preserved their values are generated as normal
        # e.g. rank,numebr of instances  = 0. THis is useful for outputting data on the polysemes
        # When empty polysemes are not preserved, empty polysemes are assigned values from the baseline model.

        self.preserve_empty_polysemes = preserve_empty_polysemes

        # First generate baseline model
        preposition_models_dict = dict()

        # Get parameters for each preposition
        for p in preposition_list:
            M = GeneratePrepositionModelParameters(self.study_info, p, self.train_scenes,
                                                   features_to_remove=self.features_to_remove)
            preposition_models_dict[p] = M

        self.preposition_parameters_dict = preposition_models_dict
        self.baseline_model = PrototypeModel(preposition_models_dict, self.test_scenes, self.study_info)
        self.baseline_model.test_prepositions = polysemous_preposition_list
        self.baseline_model.name = self.baseline_model_name

        self.cluster_model = KMeansPolysemyModel(self.preposition_parameters_dict, self.test_scenes, self.study_info)

        self.non_shared = PrototypePolysemyModel(GeneratePolysemeModels.our_model_name, self.train_scenes,
                                                 self.test_scenes, self.study_info,
                                                 preserve_empty_polysemes=self.preserve_empty_polysemes,
                                                 baseline_model=self.baseline_model,
                                                 features_to_remove=self.features_to_remove)
        self.shared = PrototypePolysemyModel(GeneratePolysemeModels.other_model_name, self.train_scenes,
                                             self.test_scenes, self.study_info,
                                             preserve_empty_polysemes=self.preserve_empty_polysemes,
                                             baseline_model=self.baseline_model,
                                             features_to_remove=self.features_to_remove)

        self.models = [self.non_shared, self.shared, self.baseline_model, self.cluster_model]

    def output_polyseme_info(self):
        """Summary
        """
        d = self.non_shared.polyseme_dict

        for preposition in d:
            out = dict()
            print(("Outputting:" + preposition))
            for polyseme in d[preposition]:
                polyseme.output_prototype_weight()
                polyseme.output_definition()
                polyseme.plot()

                polyseme.preposition_models.aff_dataset.to_csv(polyseme.annotation_csv)
                polyseme.preposition_models.affFeatures.mean().to_csv(polyseme.mean_csv)

                out[preposition + "-" + polyseme.polyseme_name] = [polyseme.get_number_of_instances(), polyseme.rank]

            number_df = pd.DataFrame(out, ["Number", "Rank"])
            number_df.to_csv(self.study_info.polyseme_data_folder + "/ranks/" + preposition + " -ranks.csv")


class MultipleRunsPolysemyModels(MultipleRuns):
    """Summary
    
    Attributes:
        all_csv (TYPE): Description
        all_plot (TYPE): Description
        average_csv (TYPE): Description
        average_plot_pdf (TYPE): Description
        average_plot_title (TYPE): Description
        comparison_csv (TYPE): Description
        file_tag (TYPE): Description
        km_comparison_csv (TYPE): Description
        scores_plots_folder (TYPE): Description
        scores_tables_folder (TYPE): Description
        study_info (TYPE): Description
    """

    def __init__(self, study_info_, number_runs=None, k=None, compare=None):
        """Summary
        
        Args:
            study_info_ (TYPE): Description
            number_runs (None, optional): Description
            k (None, optional): Description
            compare (None, optional): Description
        
        Deleted Parameters:
            study_info (TYPE): Description
            constraint_dict (TYPE): Description
            features_to_test (None, optional): Description
        """
        self.study_info = study_info_

        MultipleRuns.__init__(self, GeneratePolysemeModels, self.study_info, number_runs=number_runs, test_size=None,
                              k=k,
                              compare=compare, features_to_test=None)

        self.scores_tables_folder = self.study_info.polysemy_score_folder + "tables"
        self.scores_plots_folder = self.study_info.polysemy_score_folder + "plots"

        self.all_csv = self.study_info.polysemy_score_folder + "all_test.csv"
        self.all_plot = self.study_info.polysemy_score_folder + "ScoresUsingAllData.pdf"

        if self.k is not None:
            self.file_tag = str(self.k) + "fold"
            self.average_plot_title = "Scores Using Repeated K-Fold Validation. K = " + str(self.k) + " N = " + str(
                self.number_runs)

            self.average_plot_pdf = self.scores_plots_folder + "/average" + self.file_tag + ".pdf"
            self.average_csv = self.scores_tables_folder + "/averagemodel scores " + self.file_tag + ".csv"
            self.comparison_csv = self.scores_tables_folder + "/repeatedcomparisons " + self.file_tag + ".csv"
            self.km_comparison_csv = self.scores_tables_folder + "/km_repeatedcomparisons " + self.file_tag + ".csv"

    # Overides inherited method
    def output(self):
        """Summary
        """
        # Handle outputting here so we're not always outputting
        self.average_dataframe = self.dataframe_dict["all_features"]
        # Reorder columns for output

        new_column_order = self.Generate_Models_all_scenes.model_name_list
        reordered_df = self.average_dataframe[new_column_order]
        reordered_df.to_csv(self.average_csv)

        self.plot_dataframe_bar_chart(self.average_dataframe, self.average_plot_pdf, "Preposition", "Score",
                                      self.average_plot_title)
        if self.compare is not None:
            # Output to csv
            self.comparison_df.to_csv(self.comparison_csv)

            self.km_comparison_df.to_csv(self.km_comparison_csv)


def output_all_polyseme_info(study_info_):
    """Summary
    :param study_info_:
    
    Args:
        study_info_ (TYPE): Description
    """
    print("outputting all polyseme info")
    all_scenes = study_info_.scene_name_list
    generated_polyseme_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info_,
                                                       preserve_empty_polysemes=True)
    generated_polyseme_models.output_polyseme_info()


def test_on_all_scenes(study_info_):
    """Summary
    :param study_info_:
    
    Args:
        study_info_ (TYPE): Description
    """
    print("test on all scenes")

    all_scenes = study_info_.scene_name_list
    generated_polyseme_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info_)

    p_models = generated_polyseme_models.models

    t = TestModels(p_models, "all")
    all_dataframe = t.score_dataframe.copy()

    # all_dataframe =all_dataframe.drop(non_polysemous_prepositions)

    all_dataframe.to_csv(study_info_.score_folder + "all_test.csv")
    print(all_dataframe)


def test_model(runs, k, study_info_):
    """Summary
    
    Args:
        runs (TYPE): Description
        k (TYPE): Description
        study_info_ (TYPE): Description
        :param study_info_:
    """
    m = MultipleRunsPolysemyModels(study_info_, number_runs=runs, k=k, compare="y")
    print(("Test Model k = " + str(k)))
    m.validation()
    m.output()
    print((m.average_dataframe))


def test_models(study_info_):
    """Summary
    
    Args:
        study_info_ (TYPE): Description
    """
    mpl.rcParams['font.size'] = 40
    mpl.rcParams['legend.fontsize'] = 37
    mpl.rcParams['axes.titlesize'] = 'medium'
    mpl.rcParams['axes.labelsize'] = 'medium'
    mpl.rcParams['ytick.labelsize'] = 'small'

    test_on_all_scenes(study_info_)
    # test_model(2,2)
    test_model(10, 10, study_info_)


def output_typicality(study_info_):
    """Summary
    :param study_info_:
    
    Args:
        study_info_ (TYPE): Description
    """
    print("outputting typicalities")

    all_scenes = study_info_.scene_name_list
    generated_polyseme_models = GeneratePolysemeModels(all_scenes, all_scenes, study_info_)
    p_models = generated_polyseme_models.models
    for model in p_models:

        for preposition in polysemous_preposition_list:
            model.output_typicalities(preposition)


def output_clustering_info(study_info_):
    """Summary
    :param study_info_:
    
    Args:
        study_info_ (TYPE): Description
    """
    mpl.rcParams['font.size'] = 15
    mpl.rcParams['legend.fontsize'] = 12
    for preposition in polysemous_preposition_list:
        c = Clustering(study_info_, preposition)

        c.plot_elbow_polyseme_inertia()
        c.output_initial_inertia()
        c.output_expected_kmeans_model()


def work_out_all_hry_clusters(study_info_):
    """Summary
    :param study_info_:
    
    Args:
        study_info_ (TYPE): Description
    """
    print("Working out hry clusters")
    for preposition in polysemous_preposition_list:
        print(preposition)
        c = Clustering(study_info_, preposition)
        c.work_out_hierarchy_model()


def main(study_info_):
    """Un/comment functions to run tests and outputs
    
    Args:
        study_info_ (TYPE): Description
        :param study_info_:
    
    Deleted Parameters:
        constraint_dict (TYPE): Description
    """
    # output_all_polyseme_info(study_info_)
    # Clustering

    work_out_all_hry_clusters(study_info_)

    # Polysemes and performance

    # output_typicality(study_info_)
    # test_models(study_info_)

    mpl.rcParams['axes.titlesize'] = 'large'
    mpl.rcParams['axes.labelsize'] = 'large'
    output_clustering_info(study_info_)


if __name__ == '__main__':
    study_info = StudyInfo("2019 study")

    main(study_info)

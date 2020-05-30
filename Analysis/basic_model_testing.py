"""Summary

Attributes:
    comp_filetag (TYPE): Description
    preposition_list (TYPE): Description
    sv_filetag (TYPE): Description
"""
# First run compile_instances.py

# Input: Configuration selection info from compile_instances. Constraints from Comp task
# Looks at how features relate to categorisation and likelihood of selection
# Outputs prototypes and plots
# Can be run after compile_instances

# Standard imports
import csv
import pandas as pd
import numpy as np
import math

# Modules for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

# Modules for testing and model making
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.special import comb

# Local module imports
from preprocess_features import Features
from compile_instances import InstanceCollection, SemanticCollection, ComparativeCollection
from data_import import Configuration, StudyInfo
from classes import Constraint

# Useful global variables
sv_filetag = SemanticCollection.filetag  # Tag for sv task files
comp_filetag = ComparativeCollection.filetag  # Tag for comp task files
preposition_list = StudyInfo.preposition_list


def convert_index(i, number_of_columns):
    """Converts index to place in row/columns for plots
    
    Args:
        x (TYPE): Description
        number_of_columns (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    if i == 0 or i == 6 or i == 12:
        x_pos = 0
        y_pos = 0
    elif i < 6:
        x_pos = int(i / number_of_columns)
        y_pos = i % number_of_columns
    else:
        i = i - 6
        x_pos = int(i / number_of_columns)
        y_pos = i % number_of_columns

    return x_pos, y_pos


class SemanticMethods:
    """Class for reusable methods related to semantic measurements
    """

    @staticmethod
    def semantic_distance(weight_array, x, y, feature_keys, features_to_remove=None):
        """
        Parameters:
            weight_array: 1-D Array of feature weights
            x (TYPE): Description
            y: 1-D arrays. Points to compare
            feature_keys (TYPE): Description
            features_to_remove: Feature to remove from consideration
        
        Returns:
            distance: Float representing semantic distance from list_of_annotations to y

        
        Deleted Parameters:
            list_of_annotations: 1-D arrays. Points to compare
        """
        if features_to_remove is None:
            features_to_remove = []
        if len(feature_keys) != len(weight_array):
            print("Error: weight array and feature keys not same length")
            print(len(weight_array))
            print(len(feature_keys))

        for feature in features_to_remove:
            i = feature_keys.index(feature)
            weight_array[i] = 0

        point = np.subtract(x, y)  # Subtract arrays point wise

        point = np.square(point)  # Square pointwise

        summ = np.dot(point, weight_array)  # Dot product pointwise by weights

        distance = math.sqrt(summ)  # Square root to get distance

        return distance

    @staticmethod
    def semantic_similarity(weight_array, x, y, feature_keys, features_to_remove=None):
        """
        Calculates weighted semantic similarity between x and y.

        :param weight_array:
        :param x:
        :param y:
        :param feature_keys:
        :param features_to_remove:
        :return: float
        """

        # Get semantic distance
        distance = SemanticMethods.semantic_distance(weight_array, x, y, feature_keys,
                                                     features_to_remove=features_to_remove)
        # Get similarity
        out = math.exp(-distance)
        return out


class GeneratePrepositionModelParameters:
    """Summary

    Class with methods to generate model parameters for particular preposition
    
    Attributes:
        aff_dataset (TYPE): Description
        affFeatures (TYPE): Description
        affRelations (TYPE): Description
        all_features_regression_weight_csv (TYPE): Description
        allFeatures (TYPE): Description
        barycentre_csv (TYPE): Description
        barycentre_prototype (TYPE): Description
        categorisation_feature_name (TYPE): Description
        category_index (int): Description
        dataset (TYPE): Description
        exemplar_csv (TYPE): Description
        exemplar_mean (TYPE): Description
        all_feature_keys (TYPE): Description
        features_to_remove (TYPE): Description
        fig_feature_name (TYPE): Description
        ground_feature_name (TYPE): Description
        interval (TYPE): Description
        interval_predictions (TYPE): Description
        linear_regression_model (TYPE): Description
        neg_dataset (TYPE): Description
        neg_features (TYPE): Description
        poly_regression_model (TYPE): Description
        polyseme (TYPE): Description
        possible_instances_dataset (TYPE): Description
        preposition (TYPE): Description
        prototype (list): Description
        prototype_csv (TYPE): Description
        ratio_feature_name (TYPE): Description
        ratio_index (int): Description
        regression_weight_csv (TYPE): Description
        regression_weights (list): Description
        feature_dataframe (TYPE): Description
        feature_keys (TYPE): Description
        scene_feature_name (TYPE): Description
        scene_index (int): Description
        study_info (TYPE): Description
        train_dataset (TYPE): Description
        train_possible_intances_dataset (TYPE): Description
        train_scenes (TYPE): Description
        typical_dataset (TYPE): Description
        typical_features (TYPE): Description
    """

    # Given training scenes, works out models for individual preposition
    ratio_feature_name = InstanceCollection.ratio_feature_name
    categorisation_feature_name = InstanceCollection.categorisation_feature_name
    scene_feature_name = InstanceCollection.scene_feature_name
    fig_feature_name = InstanceCollection.fig_feature_name
    ground_feature_name = InstanceCollection.ground_feature_name

    ratio_index = -2
    category_index = -1
    scene_index = 0

    interval = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]).reshape(-1, 1)

    def __init__(self, study_info_, preposition, train_scenes, features_to_remove=None, polyseme=None):
        """Summary
        
        Args:
            study_info_ (TYPE): Description
            preposition (TYPE): Description
            train_scenes (TYPE): Description
            features_to_remove (None, optional): Description
            polyseme (None, optional): Description
        """

        if features_to_remove is None:
            features_to_remove = []

        self.study_info = study_info_

        self.all_feature_keys = self.study_info.all_feature_keys
        self.features_to_remove = features_to_remove

        self.feature_keys = []
        for f in self.all_feature_keys:
            if f not in self.features_to_remove:

                self.feature_keys.append(f)

        self.train_scenes = train_scenes

        # Given polyseme if being used to model polyseme
        self.polyseme = polyseme

        self.preposition = preposition
        # Use pandas dataframes for ease of importing etc..
        # Created in compile_instances write_config_ratios()
        # Row is created in dataset only if the configuration was tested
        # Set of values with selection information
        config_ratio_csv = self.study_info.config_ratio_csv(sv_filetag, preposition)
        dataset = pd.read_csv(config_ratio_csv)
        self.dataset = dataset

        self.possible_instances_dataset = self.dataset.copy()
        if self.polyseme is not None:
            # # Remove none polyseme preposition instances from dataset
            indexes_to_drop = []
            indexes_to_drop_pid = []
            for index, row in self.dataset.iterrows():
                if polyseme.potential_instance(row[self.scene_feature_name], row[self.fig_feature_name],
                                               row[self.ground_feature_name]):
                    pass
                elif row[self.categorisation_feature_name] == 0:
                    indexes_to_drop_pid.append(index)

                else:
                    indexes_to_drop.append(index)
                    indexes_to_drop_pid.append(index)

            # Dataset to train polyseme on (configurations not yet removed for training)
            self.dataset.drop(self.dataset.index[indexes_to_drop], inplace=True)
            # Dataset of configurations that fit polyseme conditions
            self.possible_instances_dataset.drop(self.possible_instances_dataset.index[indexes_to_drop_pid],
                                                 inplace=True)
            # Remove non-training instances
            self.train_possible_intances_dataset = self.remove_nontrainingscenes(
                self.possible_instances_dataset)  # self.possible_instances_dataset[(self.possible_instances_dataset.iloc[:,self.scene_index].isin(train_scenes))]

        # Remove rows from above where not training scene
        self.train_dataset = self.remove_nontrainingscenes(
            self.dataset)

        # Remove selection info columns and names to only have features
        self.allFeatures = self.remove_nonfeatures(self.train_dataset)

        # Feature dataframe for training the models.
        self.feature_dataframe = self.remove_unused_features(self.allFeatures)

        # Remove rows from above where not a preposition instance
        self.aff_dataset = self.train_dataset[(self.train_dataset.iloc[:, self.category_index] == 1)]

        # Remove selection info columns to only have features
        self.affFeatures = self.remove_nonfeatures(self.aff_dataset)
        

        # Typical instances are the best examples
        ratio_max = self.train_dataset[self.ratio_feature_name].max()
        self.typical_dataset = self.train_dataset[(self.train_dataset.iloc[:, self.ratio_index] == ratio_max)]
        # Remove selection info columns to only have features
        self.typical_features = self.remove_nonfeatures(self.typical_dataset)
        # Ratio dataset with non-instances - nobody selected the preposition
        self.neg_dataset = self.train_dataset[(self.train_dataset.iloc[:, self.category_index] == 0)]
        # Remove selection info columns to only have features
        self.neg_features = self.remove_nonfeatures(self.neg_dataset)

        # prototype calculated using regression. Stored as array
        self.prototype = []

        self.prototype_csv = self.study_info.model_info_folder + "/prototypes/" + preposition + ".csv"

        # regression weights calculated by linear regression. stored as array and dataframe
        self.poly_regression_model = None
        self.linear_regression_model = None

        self.regression_weights = []
        self.regression_weight_csv = self.study_info.model_info_folder + "/regression weights/" + preposition + ".csv"
        self.all_features_regression_weight_csv = self.study_info.model_info_folder + "/regression weights/allfeatures_" + preposition + ".csv"

        # Stores model predictions for later plotting
        self.interval_predictions = dict()

        # barycentre_prototype . stored as array
        self.barycentre_prototype = None

        self.barycentre_csv = self.study_info.model_info_folder + "/barycentre model/" + preposition + "-prototype.csv"

        # exemplar_mean . stored as array
        self.exemplar_mean = None

        self.exemplar_csv = self.study_info.model_info_folder + "/exemplar/" + preposition + "-exemplar_means.csv"

        self.work_out_models()

    def remove_nontrainingscenes(self, d):
        """Summary
        
        Args:
            d (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        copy_d = d.copy()
        return copy_d[(copy_d.iloc[:, self.scene_index].isin(self.train_scenes))]

    def remove_nonfeatures(self, d):
        """Summary
        Removes nonfeature columns from a dataframe
        Args:
            d (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # Remove selection info columns and names to only have features
        return d.drop(["Scene", "Figure", "Ground", self.ratio_feature_name, self.categorisation_feature_name], axis=1)

    def remove_unused_features(self, d):
        """Summary

        Args:
            d (TYPE): Description

        Returns:
            TYPE: Description
        """
        new_d = d

        if self.features_to_remove is not None:

            # Remove features to remove
            new_d = new_d.drop(self.features_to_remove, axis=1)
        return new_d

    def plot_features_ratio(self, no_columns, axes, feature, X, y_pred, Y):
        """Summary
        
        Args:
            no_columns (TYPE): Description
            axes (TYPE): Description
            feature (TYPE): Description
            X (TYPE): Description
            y_pred (TYPE): Description
            Y (TYPE): Description
        """
        # X = Selection Ratio
        # y_pred =  predicted Y values on unit interval
        # Y =  feature value

        index = self.feature_keys.index(feature)

        # Get position to  display, by index
        x_pos, y_pos = convert_index(index, no_columns)


        ax1 = axes[x_pos, y_pos]

        ax1.set_xlabel("Selection Ratio")
        ylabel = feature
        # Rename some features
        if ylabel == "contact_proportion":
            ylabel = "contact"
        if ylabel == "bbox_overlap_proportion":
            ylabel = "containment"

        ax1.set_ylabel(ylabel)
        # ax1.set_xbound(0,1)
        # ax1.set_ybound(0,1)
        # ax1.set_xlim(-0.1,1.1)
        # ax1.set_ylim(-0.1,1.1)
        ax1.grid(True)

        # Plot data point scatter
        ax1.plot(X, Y, 'k.')
        # Plot regression line
        ax1.plot(X, y_pred, color='red', linewidth=2)
        # Plot barycentre and exemplar values
        end = [1]
        end = np.array(end).reshape(-1, 1)

        index_for_prototypes = self.all_feature_keys.index(feature)
        if self.barycentre_prototype is not None:
            b = self.barycentre_prototype[index_for_prototypes]
            b = np.array([b]).reshape(-1, 1)
            # Plot barycentre value
            ax1.plot(end, b, markersize=10, markeredgewidth=2, marker='+')
        if self.exemplar_mean is not None:
            ex = self.exemplar_mean[index_for_prototypes]
            ex = np.array([ex]).reshape(-1, 1)

            # Plot exemplar mean value
            ax1.plot(end, ex, markersize=10, markeredgewidth=2, marker=(5, 2))

    # if self.prototype is not None:
    # 	p = self.prototype[index]
    # 	p = np.array([p]).reshape(-1,1)

    # 	#Plot exemplar mean value
    # 	ax1.plot(p,end,markersize=10,markeredgewidth=2, marker=(4, 2))

    def work_out_models(self):
        """Summary
        """
        self.work_out_linear_regression_model()
        # self.work_out_polynomial_regression_model(3)
        self.work_out_barycentre_prototype()
        self.work_out_exemplar_mean()
        self.work_out_prototype_model()

    def work_out_barycentre_prototype(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        out = []
        X = self.affFeatures

        for feature in self.all_feature_keys:
            pro_value = X[feature].mean()

            out.append(pro_value)

        out = np.array(out)
        self.barycentre_prototype = out
        return self.barycentre_prototype

    def work_out_exemplar_mean(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        out = []
        X = self.typical_features

        for feature in self.all_feature_keys:
            pro_value = X[feature].mean()
            out.append(pro_value)

        out = np.array(out)
        self.exemplar_mean = out
        return self.exemplar_mean

    def get_plot_filename(self, file_no):
        """Summary
        
        Args:
            file_no (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        x = str(file_no)

        if self.polyseme is not None:
            filename = self.polyseme.plot_folder + self.preposition + "-" + self.polyseme.polyseme_name + x + ' .pdf'
        else:

            filename = self.study_info.model_info_folder + "/plots/" + self.preposition + x + ".pdf"
        return filename

    def plot_models(self):
        """Summary
        """
        # Plots simple linear regressions used to find prototypes
        no_rows = 3
        no_columns = 2

        fig, axes = plt.subplots(nrows=no_rows, ncols=no_columns, sharex=False, sharey=False)
        fig.tight_layout()
        fig.canvas.set_window_title('Ratio vs. Feature')
        plot_count = 0
        file_no = 1

        for feature in self.feature_keys:

            plot_count += 1

            r = plot_count % (no_columns * no_rows)

            # Reshape data first
            Y = self.train_dataset[feature].values.reshape(-1, 1)
            X = self.train_dataset[self.ratio_feature_name].values.reshape(-1, 1)
            # Get prediction of all points on interval
            y_pred = self.interval_predictions[feature]

            self.plot_features_ratio(no_columns, axes, feature, X, y_pred, Y)

            filename = self.get_plot_filename(file_no)

            # When the figure is full of plots, save figure
            if r == 0:
                plt.savefig(filename, bbox_inches='tight')
                file_no += 1

                # Clear plots for new figure
                fig, axes = plt.subplots(nrows=no_rows, ncols=no_columns, sharex=False, sharey=False)
                fig.tight_layout()
                fig.canvas.set_window_title('Ratio vs. Feature')
        # Save remaining plots
        filename = self.get_plot_filename(file_no)
        plt.savefig(filename, bbox_inches='tight')

    def work_out_feature_prototype(self, feature):
        """Summary
        
        Args:
            feature (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # First predict feature value given selection ratio of 1
        # Reshape data first
        X = self.train_dataset[self.ratio_feature_name].values.reshape(-1, 1)
        Y = self.train_dataset[feature].values.reshape(-1, 1)

        model1 = LinearRegression()

        # Fit model to data
        model1.fit(X, Y)
        y_pred = model1.predict(X)
        self.interval_predictions[feature] = y_pred

        # Get prototype for feature
        max_point = np.array([1]).reshape(-1, 1)

        feature_prototype = model1.predict(max_point)

        pro_value = feature_prototype[0][0]
        return pro_value

    def work_out_linear_regression_model(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        # Next get gradient when feature predicts selection ratio
        # Reshape data first

        X = self.feature_dataframe
        Y = self.train_dataset[self.ratio_feature_name].values.reshape(-1, 1)

        lin_model = LinearRegression()

        # Fit model to data
        lin_model.fit(X, Y)

        self.linear_regression_model = lin_model


        return lin_model

    def work_out_feature_weights(self):
        """Summary
        """
        if self.linear_regression_model is None:
            model2 = self.work_out_linear_regression_model()
        else:
            model2 = self.linear_regression_model

        X = self.feature_dataframe

        v = pd.DataFrame(model2.coef_, index=["coefficient"]).transpose()
        w = pd.DataFrame(X.columns, columns=["feature"])
        coeff_df = pd.concat([w, v], axis=1, join="inner")
        coeff_df = coeff_df.set_index("feature")

        weights = []

        for feature in self.all_feature_keys:
            if self.features_to_remove is not None:
                # If the feature is removed, append 0 instead
                if feature in self.features_to_remove:
                    weights.append(0)
                else:
                    w = abs(coeff_df.loc[feature, "coefficient"])
                    weights.append(w)
            else:
                w = abs(coeff_df.loc[feature, "coefficient"])
                weights.append(w)

        self.regression_weights = np.array(weights)

    def work_out_prototype_model(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        # Work out linear regression on each feature by comparing to the ratio of times selected

        # This step gives prototypes for later steps, which are saved to prototypes folder

        prototype = []

        for feature in self.all_feature_keys:
            # First predict feature value given selection ratio of 1
            pro_value = self.work_out_feature_prototype(feature)

            # Add to dictionary
            prototype.append(pro_value)

        self.work_out_feature_weights()

        self.prototype = np.array(prototype)

        return self.prototype

    def work_out_polynomial_regression_model(self, n):
        """Summary
        
        Args:
            n (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # Next get gradient when feature predicts selection ratio
        # Reshape data first
        X = self.feature_dataframe
        Y = self.train_dataset[self.ratio_feature_name].values.reshape(-1, 1)

        polynomial_features = PolynomialFeatures(degree=n)
        x_poly = polynomial_features.fit_transform(X)

        model2 = LinearRegression()

        # Fit model to data
        model2.fit(x_poly, Y)

        self.poly_regression_model = model2
        print((self.preposition))
        print(("Polynomial Score" + str(n)))
        print((model2.score(x_poly, Y)))

        return model2

    def output_models(self):
        """Summary
        """
        # Only called once when training scenes are all scenes, so these are the best model parameters
        wf = pd.DataFrame(self.regression_weights, self.all_feature_keys)

        wf.to_csv(self.regression_weight_csv)

        pf = pd.DataFrame(self.prototype, self.all_feature_keys)

        pf.to_csv(self.prototype_csv)

        epf = pd.DataFrame(self.barycentre_prototype, self.all_feature_keys)

        epf.to_csv(self.barycentre_csv)

        exf = pd.DataFrame(self.exemplar_mean, self.all_feature_keys)

        exf.to_csv(self.exemplar_csv)

    def read_all_feature_weights(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        # Read regression weights for all features
        wf = pd.read_csv(self.all_features_regression_weight_csv, index_col=0)

        return wf

    def read_regression_weights(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        # Read regression weights for relations
        wf = pd.read_csv(self.regression_weight_csv, index_col=0)

        return wf

    def read_calculations(self, classifier):
        """Summary
        
        Args:
            classifier (TYPE): Description
        """
        with open("figures/csv_tables" + self.name + classifier + ":" + self.preposition + ".csv") as csvfile:
            read = csv.reader(csvfile)  # .readlines())
            reader = list(read)

            for line in reader:
                if line[0] in self.feature_keys:
                    value = line[1]
                    setattr(self, classifier + ":" + line[0], value)


class Model:
    """Summary
    

    """

    # Puts together preposition models and has various functions for testing
    def __init__(self, name, test_scenes, study_info_):
        """Summary
        
        Args:

        """
        # if features_to_remove is None:
        #     features_to_remove = []

        self.study_info = study_info_
        self.feature_processer = Features(self.study_info.name)

        self.all_feature_keys = self.study_info.all_feature_keys

        self.name = name
        # Prepositions to test
        self.test_prepositions = preposition_list
        # Dictionary containing constraints to satisfy
        self.constraint_dict = Constraint.read_from_csv(self.study_info.constraint_csv)

        self.test_scenes = test_scenes

    def generate_arrays(self, salient_features):
        """
        Creates prototype and weight arrays from dictionary of salient features specifiying their (human-readable) typical values.

        :param salient_features:
        :returns prototype_array, weight_array
        """
        pro_array = []
        weight_array = []
        for feature in self.all_feature_keys:
            if feature not in salient_features:
                pro_array.append(0)
                weight_array.append(0)
            else:
                x = self.feature_processer.convert_normal_value_to_standardised(feature, salient_features[feature])
                pro_array.append(x)
                weight_array.append(1)

        pro_array = np.array(pro_array)
        weight_array = np.array(weight_array)

        return pro_array, weight_array

    def get_typicality(self, preposition, point):
        print("This shouldn't be called")

    def get_score(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        # Calculates scores on all constraints for particular model
        # Prototype dict is a dictionary of prototypes (1D arrays) for each preposition

        scores = []  # Scores for each preposition to be averaged later
        weight_totals = []  # Total constraint weights for each preposition
        totals = []  # Total number of constraints for each preposition
        average_score = 0
        weighted_average_score = 0
        total_weight_counter = 0
        for preposition in self.test_prepositions:

            allConstraints = self.constraint_dict[preposition]
            # Constraints to test on
            Constraints = []

            for c in allConstraints:
                if c.scene in self.test_scenes:
                    Constraints.append(c)

            # Constraint info
            weight_counter = 0
            counter = 0
            for c in Constraints:
                weight_counter += c.weight
                counter += 1
            total_weight_counter += weight_counter
            weight_totals.append(weight_counter)
            totals.append(counter)

            # Get score for preposition
            score_two = self.weighted_score(preposition, Constraints)

            weighted_average_score += score_two

            score = float(score_two) / weight_counter
            average_score += score

            # score =  round(score,3)
            scores.append(score)

        average_score = float(average_score) / len(self.test_prepositions)
        weighted_average_score = float(weighted_average_score) / total_weight_counter
        # average_score =  round(average_score,3)
        # weighted_average_score =  round(weighted_average_score,3)
        scores.append(average_score)
        scores.append(weighted_average_score)

        self.weight_totals = weight_totals
        self.totals = totals
        self.scores = scores

        return scores

    # First score based on number of satisfied constraints
    def unweighted_score(self, preposition, Constraints):
        """Summary
        
        Args:
            preposition (TYPE): Description
            Constraints (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # Calculates how well W and P satisfy the constraints, NOT accounting for constraint weight
        # W and P are 1D arrays

        total = 0
        counter = 0
        for c in Constraints:
            total += 1

            lhs = self.get_typicality(preposition, c.lhs)
            rhs = self.get_typicality(preposition, c.rhs)
            if c.is_satisfied(lhs, rhs):
                counter += 1

        return counter

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
        # else:
        # 	if len(self.test_scenes) == len(self.train_scenes):
        # 		if preposition == "under" and self.name == "Our Prototype":

        # 			print("#")
        # 			print(c.scene)
        # 			print("ground:" + c.ground)
        # 			print("Correct Figure: " + c.f1)
        # 			print("Incorrectly better figure:" + c.f2)
        # 			cp = Comparison(c.scene,preposition,c.ground)
        # 			print("possible_figures:" + str(cp.possible_figures))

        return counter


class PrototypeModel(Model):
    name = "Our Prototype"

    def __init__(self, preposition_model_dict, test_scenes, study_info_):
        self.preposition_model_dict = preposition_model_dict

        Model.__init__(self, PrototypeModel.name, test_scenes, study_info_)

    def get_typicality(self, preposition, point):
        p_model = self.preposition_model_dict[preposition]
        weight_array = p_model.regression_weights
        prototype_array = p_model.prototype

        out = SemanticMethods.semantic_similarity(weight_array, point, prototype_array, self.all_feature_keys,
                                                  features_to_remove=p_model.features_to_remove)

        return out


class CSModel(Model):
    name = "Conceptual Space"

    def __init__(self, preposition_model_dict, test_scenes, study_info_):
        self.preposition_model_dict = preposition_model_dict

        Model.__init__(self, CSModel.name, test_scenes, study_info_)

    def get_typicality(self, preposition, point):
        p_model = self.preposition_model_dict[preposition]
        weight_array = p_model.regression_weights
        prototype_array = p_model.barycentre_prototype
        out = SemanticMethods.semantic_similarity(weight_array, point, prototype_array, self.all_feature_keys,
                                                  features_to_remove=p_model.features_to_remove)

        return out


class ExemplarModel(Model):
    name = "Exemplar"

    def __init__(self, preposition_model_dict, test_scenes, study_info_):
        self.preposition_model_dict = preposition_model_dict

        Model.__init__(self, ExemplarModel.name, test_scenes, study_info_)

    def get_typicality(self, preposition, point):

        p_model = self.preposition_model_dict[preposition]
        weight_array = p_model.regression_weights
        ## Need to fix here downwards
        exemplars = p_model.typical_features
        none_instances = p_model.neg_features

        # Find average semantic similarity to points in exemplar model
        counter = 0
        semantic_similarity_sum = 0

        # Iterate over rows in exemplar dataframe
        for index, row in exemplars.iterrows():
            # Get row values
            exemplar_values = row.values

            # Convert values to np array
            exemplar_values = np.array(exemplar_values)
            counter += 1
            # Calculate similarity of current point to exemplar

            semantic_similarity_sum += SemanticMethods.semantic_similarity(weight_array, point, exemplar_values,
                                                                           self.all_feature_keys,
                                                                           features_to_remove=p_model.features_to_remove)

        if counter == 0:
            return 0
        else:
            average = float(semantic_similarity_sum) / counter

        return average


class ProximityModel(Model):
    name = "Proximity"

    def __init__(self, test_scenes, study_info_):

        Model.__init__(self, ProximityModel.name, test_scenes, study_info_)

        prototype_array = []
        weight_array = []

        for feature in self.all_feature_keys:
            if feature != "shortest_distance":
                prototype_array.append(0)
                weight_array.append(0)
            else:
                x = self.feature_processer.convert_normal_value_to_standardised(feature, 0)
                prototype_array.append(x)
                weight_array.append(1)

        self.prototype_array = np.array(prototype_array)
        self.weight_array = np.array(weight_array)

    def get_typicality(self, preposition, point):

        out = SemanticMethods.semantic_similarity(self.weight_array, point, self.prototype_array, self.all_feature_keys)

        return out


class SimpleModel(Model):
    name = "Simple"

    def __init__(self, test_scenes, study_info_):
        Model.__init__(self, SimpleModel.name, test_scenes, study_info_)

        self.prototype_dictionary = dict()
        self.weight_dictionary = dict()

        in_prototype, in_weights = self.generate_arrays({"bbox_overlap_proportion": 1})
        self.prototype_dictionary["in"] = self.prototype_dictionary["inside"] = in_prototype
        self.weight_dictionary["in"] = self.weight_dictionary["inside"] = in_weights

        on_prototype, on_weights = self.generate_arrays({"above_proportion": 1, "contact_proportion": 1})
        self.prototype_dictionary["on"] = self.prototype_dictionary["on top of"] = on_prototype
        self.weight_dictionary["on"] = self.weight_dictionary["on top of"] = on_weights

        above_prototype, above_weights = self.generate_arrays({"above_proportion": 1, "horizontal_distance": 0})
        self.prototype_dictionary["above"] = self.prototype_dictionary["over"] = above_prototype
        self.weight_dictionary["above"] = self.weight_dictionary["over"] = above_weights

        below_prototype, below_weights = self.generate_arrays({"below_proportion": 1, "horizontal_distance": 0})
        self.prototype_dictionary["below"] = self.prototype_dictionary["under"] = below_prototype
        self.weight_dictionary["below"] = self.weight_dictionary["under"] = below_weights

        against_prototype, against_weights = self.generate_arrays({"contact_proportion": 1, "horizontal_distance": 0})
        self.prototype_dictionary["against"] = against_prototype
        self.weight_dictionary["against"] = against_weights

    def get_typicality(self, preposition, point):
        prototype_array = self.prototype_dictionary[preposition]
        weight_array = self.weight_dictionary[preposition]

        out = SemanticMethods.semantic_similarity(weight_array, point, prototype_array, self.all_feature_keys)

        return out


class BestGuessModel(Model):
    name = "Best Guess"

    def __init__(self, test_scenes, study_info_):
        Model.__init__(self, BestGuessModel.name, test_scenes, study_info_)

        self.prototype_dictionary = dict()
        self.weight_dictionary = dict()

        inside_prototype, inside_weights = self.generate_arrays({"bbox_overlap_proportion": 1})
        self.prototype_dictionary["inside"] = inside_prototype
        self.weight_dictionary["inside"] = inside_weights

        in_prototype, in_weights = self.generate_arrays({"bbox_overlap_proportion": 1, "location_control": 1})
        self.prototype_dictionary["in"] = in_prototype
        self.weight_dictionary["in"] = in_weights

        on_prototype, on_weights = self.generate_arrays({"above_proportion": 1, "contact_proportion": 1, "support": 1})
        self.prototype_dictionary["on"] = on_prototype
        self.weight_dictionary["on"] = on_weights

        ontopof_prototype, ontopof_weights = self.generate_arrays({"above_proportion": 1, "contact_proportion": 1})
        self.prototype_dictionary["on top of"] = ontopof_prototype
        self.weight_dictionary["on top of"] = ontopof_weights

        above_prototype, above_weights = self.generate_arrays({"above_proportion": 1, "horizontal_distance": 0})
        self.prototype_dictionary["above"] = above_prototype
        self.weight_dictionary["above"] = above_weights

        over_prototype, over_weights = self.generate_arrays({"above_proportion": 1, "f_covers_g": 1})
        self.prototype_dictionary["over"] = over_prototype
        self.weight_dictionary["over"] = over_weights

        below_prototype, below_weights = self.generate_arrays({"below_proportion": 1, "horizontal_distance": 0})
        self.prototype_dictionary["below"] = below_prototype
        self.weight_dictionary["below"] = below_weights

        under_prototype, under_weights = self.generate_arrays({"below_proportion": 1, "g_covers_f": 1})
        self.prototype_dictionary["under"] = under_prototype
        self.weight_dictionary["under"] = under_weights

        against_prototype, against_weights = self.generate_arrays(
            {"contact_proportion": 1, "horizontal_distance": 0, "location_control": 0.5})
        self.prototype_dictionary["against"] = against_prototype
        self.weight_dictionary["against"] = against_weights

    def get_typicality(self, preposition, point):
        prototype_array = self.prototype_dictionary[preposition]
        weight_array = self.weight_dictionary[preposition]

        out = SemanticMethods.semantic_similarity(weight_array, point, prototype_array, self.all_feature_keys)

        return out


class RegressionModel(Model):
    # This needs fixing if using in future
    name = "Regression Model"

    def __init__(self, preposition_model_dict, test_scenes, study_info_):
        self.preposition_model_dict = preposition_model_dict

        Model.__init__(self, RegressionModel.name, test_scenes, study_info_)

    def get_typicality(self, preposition, point):
        p_model = self.preposition_model_dict[preposition]
        regression_model = p_model.regression_model

        point_array = np.array(point).reshape(1, -1)
        if self.regression_dimension is not None:
            # Must transform the point for polynomial regression
            polynomial_features = PolynomialFeatures(degree=self.regression_dimension)
            point_array = polynomial_features.fit_transform(point_array)

        t = self.regression_model_dict[preposition].predict(point_array)
        return t


class GenerateBasicModels:
    """Summary
    

    """

    # List of all model names
    model_name_list = [PrototypeModel.name, ExemplarModel.name, CSModel.name, BestGuessModel.name, SimpleModel.name,
                       ProximityModel.name]

    other_name_list = [ExemplarModel.name, CSModel.name, BestGuessModel.name, SimpleModel.name,
                       ProximityModel.name]

    # Generating models to test
    def __init__(self, train_scenes, test_scenes, study_info_, extra_features_to_remove=None, only_test_our_model=None):
        """Summary
        

        """
        self.study_info = study_info_

        # Scenes used to train models
        self.train_scenes = train_scenes
        # Scenes used to test models
        self.test_scenes = test_scenes
        # Features to remove from consideration (not used in training or testing)
        self.features_to_remove = Configuration.ground_property_features

        # Extra features may be removed to test on
        if extra_features_to_remove is not None:
            for f in extra_features_to_remove:
                self.features_to_remove.append(f)
        print(self.features_to_remove)
        preposition_models_dict = dict()

        # Get parameters for each preposition
        for p in preposition_list:
            M = GeneratePrepositionModelParameters(self.study_info, p, self.train_scenes,
                                                   features_to_remove=self.features_to_remove)
            preposition_models_dict[p] = M
        self.preposition_paramters_dict = preposition_models_dict
        our_model = PrototypeModel(preposition_models_dict, self.test_scenes, self.study_info)

        if only_test_our_model is None:

            exemplar_model = ExemplarModel(preposition_models_dict, self.test_scenes, self.study_info)
            cs_model = CSModel(preposition_models_dict, self.test_scenes, self.study_info)
            proximity_model = ProximityModel(self.test_scenes, self.study_info)
            simple_model = SimpleModel(self.test_scenes, self.study_info)
            best_guess_model = BestGuessModel(self.test_scenes, self.study_info)

            models = [our_model, exemplar_model, cs_model, proximity_model, simple_model, best_guess_model]

        else:

            models = [our_model]

        self.models = models


class TestModels:
    """Summary
    # Takes input set of models and outputs database of scores
    Attributes:
        model_name_list (list): Description
        models (TYPE): Description
        score_dataframe (TYPE): Description
        version_name (TYPE): Description
    """



    def __init__(self, models, version_name):
        """Summary
        
        Args:
            models (TYPE): Description
            version_name (TYPE): Description
        """
        self.version_name = version_name
        self.models = models
        self.model_name_list = []

        out = dict()

        for model in self.models:
            self.model_name_list.append(model.name)
            model.get_score()
            out[model.name] = model.scores

        # out["Total Constraint Weights"] = models[0].weight_totals + ["",""]

        df = pd.DataFrame(out, self.models[0].test_prepositions + ["Average", "Overall"])

        # Reorder columns
        new_column_order = self.model_name_list
        reordered_df = df[new_column_order]

        self.score_dataframe = reordered_df


class MultipleRuns:
    """Summary
    
    Attributes:
        all_csv (TYPE): Description
        all_dataframe (TYPE): Description
        all_plot (TYPE): Description
        average_csv (TYPE): Description
        average_dataframe (TYPE): Description
        average_plot_pdf (TYPE): Description
        average_plot_title (str): Description
        compare (TYPE): Description
        comparison_csv (TYPE): Description
        comparison_df (TYPE): Description
        constraint_dict (TYPE): Description
        count_cluster_model_wins (TYPE): Description
        count_other_model_beats_cluster (TYPE): Description
        count_other_model_wins (TYPE): Description
        count_our_model_wins (TYPE): Description
        count_with_feature_better (TYPE): Description
        count_without_feature_better (TYPE): Description
        dataframe_dict (TYPE): Description
        feature_comparison_df (TYPE): Description
        feature_removed_average_csv (TYPE): Description
        features_to_test (TYPE): Description
        file_tag (TYPE): Description
        Generate_Models_all_scenes (TYPE): Description
        k (TYPE): Description
        km_comparison_df (TYPE): Description
        model_generator (TYPE): Description
        number_runs (TYPE): Description
        run_count (int): Description
        scene_list (TYPE): Description
        scenes_used_for_testing (list): Description
        scenes_used_for_training (list): Description
        scores_plots_folder (TYPE): Description
        scores_tables_folder (TYPE): Description
        study_info (TYPE): Description
        test_prepositions (TYPE): Description
        test_size (TYPE): Description
        total_number_runs (TYPE): Description
    """

    # This class carries out multiple runs of model tests and outputs the results
    # Number of runs must be specified as well as either test_size for standard repeated sampling
    # or k for repeated k-fold sampling
    def __init__(self, model_generator, study_info_, number_runs=None, test_size=None, k=None,
                 compare=None,
                 features_to_test=None):
        """Summary
        
        Args:
            model_generator (TYPE): Description
            study_info_ (TYPE): Description
            number_runs (None, optional): Description
            test_size (None, optional): Description
            k (None, optional): Description
            compare (None, optional): Description
            features_to_test (None, optional): Description
        """
        self.study_info = study_info_
        self.model_generator = model_generator
        self.constraint_dict = Constraint.read_from_csv(self.study_info.constraint_csv)
        self.number_runs = number_runs
        self.test_size = test_size
        self.k = k
        self.compare = compare
        self.features_to_test = features_to_test

        self.run_count = 0
        # Dictionary of dataframes giving scores. Indexed by removed features.
        self.dataframe_dict = dict()

        self.scene_list = self.study_info.scene_name_list
        self.Generate_Models_all_scenes = self.generate_models(self.scene_list, self.scene_list)
        self.test_prepositions = self.Generate_Models_all_scenes.models[0].test_prepositions
        self.all_csv = self.study_info.name + "/scores/tables/all-model scores.csv"
        self.all_plot = self.study_info.name + "/scores/plots/ScoresUsingAllData.pdf"
        if self.features_to_test is None:

            self.scores_tables_folder = self.study_info.name + "/scores/tables/all features"
            self.scores_plots_folder = self.study_info.name + "/scores/plots/all features"
        else:
            self.scores_tables_folder = self.study_info.name + "/scores/tables/removed features"
            self.scores_plots_folder = self.study_info.name + "/scores/plots/removed features"
        if self.test_size is not None:
            self.file_tag = "rss" + str(self.test_size)
            self.average_plot_title = "Scores Using RRSS Validation"
        if self.k is not None:
            self.file_tag = str(self.k) + "fold"
            self.average_plot_title = "Scores Using Repeated K-Fold Validation. K = " + str(self.k) + " N = " + str(
                self.number_runs)

            self.average_plot_pdf = self.scores_plots_folder + "/average" + self.file_tag + ".pdf"
            self.average_csv = self.scores_tables_folder + "/averagemodel scores " + self.file_tag + ".csv"
            self.comparison_csv = self.scores_tables_folder + "/repeatedcomparisons " + self.file_tag + ".csv"

        if self.features_to_test is not None:
            self.feature_removed_average_csv = dict()
            for feature in self.features_to_test:
                self.feature_removed_average_csv[
                    feature] = self.scores_tables_folder + "/averagemodel scores " + self.file_tag + " " + feature + "removed.csv"

        self.prepare_comparison_dicts()

        if self.features_to_test is not None:
            for feature in self.features_to_test:
                self.count_without_feature_better[feature] = dict()
                self.count_with_feature_better[feature] = dict()
                for p in self.test_prepositions + ["Average", "Overall"]:
                    self.count_without_feature_better[feature][p] = 0
                    self.count_with_feature_better[feature][p] = 0
        # following lists help confirm all scenes get used for both training and testing
        self.scenes_used_for_testing = []
        self.scenes_used_for_training = []

    def prepare_comparison_dicts(self):
        """Summary
        """
        # Dealing with these values could be improved..
        # Counts to compare models
        self.count_cluster_model_wins = dict()
        self.count_other_model_beats_cluster = dict()
        # Counts to compare models
        self.count_our_model_wins = dict()
        self.count_other_model_wins = dict()
        # Counts to compare features
        self.count_without_feature_better = dict()
        self.count_with_feature_better = dict()

        # Prepare dicts
        for other_model in self.Generate_Models_all_scenes.other_name_list:
            self.count_our_model_wins[other_model] = 0
            self.count_other_model_wins[other_model] = 0
        # To compare kmeans cluster model
        if hasattr(self.Generate_Models_all_scenes, "cluster_model_name"):
            for other_model in self.Generate_Models_all_scenes.model_name_list:
                if other_model != self.Generate_Models_all_scenes.cluster_model_name:
                    self.count_cluster_model_wins[other_model] = 0
                    self.count_other_model_beats_cluster[other_model] = 0

    def generate_models(self, train_scenes, test_scenes, extra_feature_to_remove=None):
        """Summary
        Generates a list of models to test from given train and test scenes.

        """
        if self.features_to_test is not None:
            # Only test our model
            generate_models = self.model_generator(train_scenes, test_scenes, self.study_info,
                                                   extra_features_to_remove=extra_feature_to_remove,
                                                   only_test_our_model=True)

        else:
            # Test all models
            generate_models = self.model_generator(train_scenes, test_scenes, self.study_info,
                                                   extra_features_to_remove=extra_feature_to_remove)

        return generate_models

    def test_all_scenes(self):
        """Summary
        """
        generate_models = self.Generate_Models_all_scenes
        models = generate_models.models
        t = TestModels(models, "all")
        self.all_dataframe = t.score_dataframe

        self.all_dataframe.to_csv(self.all_csv)

        self.plot_dataframe_bar_chart(self.all_dataframe, self.all_plot, "Preposition", "Score",
                                      "Scores Using All Data")

    def single_validation_test(self, train_scenes, test_scenes):
        """Summary
        
        Args:
            train_scenes (TYPE): Description
            test_scenes (TYPE): Description
        """
        generate_models = self.generate_models(train_scenes, test_scenes)

        t = TestModels(generate_models.models, str(self.run_count))
        # Get generated scores
        dataset = t.score_dataframe

        # Add scores to total
        if "all_features" in self.dataframe_dict:
            self.dataframe_dict["all_features"] = self.dataframe_dict["all_features"].add(dataset)

        else:
            self.dataframe_dict["all_features"] = dataset

        # Get our score from dataframe
        our_score = dataset.at["Overall", PrototypeModel.name]

        # Compare Models
        if self.compare is not None:
            for other_model in generate_models.other_name_list:

                # Get score
                other_score = dataset.at["Overall", other_model]
                # Update counts
                if our_score > other_score:
                    self.count_our_model_wins[other_model] += 1

                if other_score > our_score:
                    self.count_other_model_wins[other_model] += 1
            if hasattr(generate_models, "cluster_model_name"):
                k_means_score = dataset.at["Overall", generate_models.cluster_model_name]
                for other_model in generate_models.model_name_list:
                    if other_model != self.Generate_Models_all_scenes.cluster_model_name:
                        # Get score
                        other_score = dataset.at["Overall", other_model]
                        # Update counts
                        if k_means_score > other_score:
                            self.count_cluster_model_wins[other_model] += 1

                        if other_score > k_means_score:
                            self.count_other_model_beats_cluster[other_model] += 1

        # Add scores to dataframe
        if self.features_to_test is not None:

            for feature in self.features_to_test:
                generate_models = self.generate_models(train_scenes, test_scenes, extra_feature_to_remove=feature)

                t = TestModels(generate_models.models, str(self.run_count))

                feature_dataset = t.score_dataframe
                # feature_dataset = feature_dataset.drop(["Total Constraint Weights"],axis=1)

                for p in self.test_prepositions + ["Average", "Overall"]:
                    without_feature_score = feature_dataset.at[p, generate_models.our_model_name]
                    with_feature_score = dataset.at[p, generate_models.our_model_name]

                    if without_feature_score > with_feature_score:
                        self.count_without_feature_better[feature][p] += 1
                    if with_feature_score > without_feature_score:
                        self.count_with_feature_better[feature][p] += 1

                # Add to totals
                if feature in self.dataframe_dict:
                    self.dataframe_dict[feature] = self.dataframe_dict[feature].add(feature_dataset)

                else:
                    self.dataframe_dict[feature] = feature_dataset

    def get_validation_scene_split(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        # Get train-test scenes
        if self.test_size is not None:
            train_scenes, test_scenes = train_test_split(self.scene_list, test_size=self.test_size)

            # Update scene lists
            for sc in train_scenes:
                if sc not in self.scenes_used_for_training:
                    self.scenes_used_for_training.append(sc)

            for sc in test_scenes:
                if sc not in self.scenes_used_for_testing:
                    self.scenes_used_for_testing.append(sc)
            return [train_scenes, test_scenes]
        if self.k is not None:
            # Create random folds for testing
            folds = []

            scenes_left = self.scene_list
            divisor = self.k
            while divisor > 1:
                t_size = float(1) / divisor
                train_scenes, test_scenes = train_test_split(scenes_left, test_size=t_size)
                folds.append(test_scenes)
                scenes_left = train_scenes
                divisor = divisor - 1
                if divisor == 1:
                    folds.append(train_scenes)

            return folds

    def folds_check(self, folds):
        """Summary
        
        Args:
            folds (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # Check all folds have some constraints to test
        for f in folds:
            for preposition in self.test_prepositions:

                allConstraints = self.constraint_dict[preposition]

                Constraints = []

                for c in allConstraints:
                    if c.scene in f:
                        Constraints.append(c)
                if len(Constraints) == 0:
                    return False
        return True

    def validation(self):
        """Summary
        """
        # Perform Repeated random sub-sampling validation
        # Either using k-fold or standard method
        for i in range(self.number_runs):
            self.run_count = i

            print(("Run Number:" + str(i + 1)))

            if self.test_size is not None:
                split = self.get_validation_scene_split()
                train_scenes = split[0]
                test_scenes = split[1]
                self.single_validation_test(train_scenes, test_scenes)
            if self.k is not None:
                # This handles the case where test_scenes do not produce any constraints
                while True:
                    folds = self.get_validation_scene_split()

                    if self.folds_check(folds):
                        for f in folds:
                            print(("Fold Number:" + str(folds.index(f))))
                            test_scenes = f
                            train_scenes = []
                            for s in self.scene_list:
                                if s not in test_scenes:
                                    train_scenes.append(s)
                            self.single_validation_test(train_scenes, test_scenes)
                        break
                    else:
                        pass
                    # print("Fold with no constraints to test. Retrying...")

        # First update value of number of runs to account for folds
        if self.k is not None:
            self.total_number_runs = self.number_runs * self.k
        else:
            self.total_number_runs = self.number_runs
        # Output comparison of models and p-value
        if self.compare is not None:
            other_model_p_value = dict()
            for other_model in self.Generate_Models_all_scenes.other_name_list:
                p_value = calculate_p_value(self.total_number_runs, self.count_our_model_wins[other_model])
                other_model_p_value[other_model] = p_value

            # Create dataframes to output
            p_value_df = pd.DataFrame(other_model_p_value, ["p_value"])
            our_model_win_count = pd.DataFrame(self.count_our_model_wins, ["Our model wins"])
            other_model_win_count = pd.DataFrame(self.count_other_model_wins, ["Other model wins"])
            # Append dataframes into one
            new_df = p_value_df.append([our_model_win_count, other_model_win_count], sort=False)
            self.comparison_df = new_df

            kmeans_other_model_p_value = dict()
            if hasattr(self.Generate_Models_all_scenes, "cluster_model_name"):

                for other_model in self.Generate_Models_all_scenes.model_name_list:
                    if other_model != self.Generate_Models_all_scenes.cluster_model_name:
                        p_value = calculate_p_value(self.total_number_runs, self.count_cluster_model_wins[other_model])
                        kmeans_other_model_p_value[other_model] = p_value
            # Create dataframes to output
            km_p_value_df = pd.DataFrame(kmeans_other_model_p_value, ["p_value"])
            cluster_model_win_count = pd.DataFrame(self.count_cluster_model_wins, ["Cluster model wins"])
            km_other_model_win_count = pd.DataFrame(self.count_other_model_beats_cluster, ["Other model wins"])
            # Append dataframes into one
            km_new_df = km_p_value_df.append([cluster_model_win_count, km_other_model_win_count], sort=False)
            self.km_comparison_df = km_new_df

        if self.features_to_test is not None:
            feature_p_value = dict()
            with_feature_better = dict()
            without_feature_better = dict()
            for feature in self.features_to_test:
                for p in self.test_prepositions + ["Average", "Overall"]:
                    p_value = calculate_p_value(self.total_number_runs, self.count_with_feature_better[feature][p])
                    feature_p_value[feature + ":" + p] = p_value
                    with_feature_better[feature + ":" + p] = self.count_with_feature_better[feature][p]
                    without_feature_better[feature + ":" + p] = self.count_without_feature_better[feature][p]

            # Create dataframes to output
            p_value_df = pd.DataFrame(feature_p_value, ["p_value"])
            win_count = pd.DataFrame(with_feature_better, ["With feature wins"])
            lose_count = pd.DataFrame(without_feature_better, ["Without feature wins"])
            # Append dataframes into one
            new_df = p_value_df.append([win_count, lose_count], sort=False)
            self.feature_comparison_df = new_df

        # Print some info
        print(("Total Runs:" + str(self.total_number_runs)))

        if self.test_size is not None:
            print("# Scenes used for testing")
            print((len(self.scenes_used_for_testing)))
            print("# Scenes used for training")
            print((len(self.scenes_used_for_training)))

        # Finalise by averaging scores in dataframe
        for key in self.dataframe_dict:
            self.dataframe_dict[key] = self.dataframe_dict[key].div(self.total_number_runs)

    def output(self):
        """Summary
        """
        # Handle outputting here so we're not always outputting
        self.average_dataframe = self.dataframe_dict["all_features"]
        # Reorder columns for output
        if self.features_to_test == None:
            new_column_order = self.Generate_Models_all_scenes.model_name_list
            reordered_df = self.average_dataframe[new_column_order]
            reordered_df.to_csv(self.average_csv)
        else:
            self.average_dataframe.to_csv(self.average_csv)

        self.plot_dataframe_bar_chart(self.average_dataframe, self.average_plot_pdf, "Preposition", "Score",
                                      self.average_plot_title)
        if self.compare is not None:
            # Output to csv
            self.comparison_df.to_csv(self.comparison_csv)
            self.km_comparison_df.to_csv(self.km_comparison_csv)
        if self.features_to_test is not None:

            # Output to csv
            self.feature_comparison_df.to_csv(self.comparison_csv)
            for feature in self.features_to_test:
                dff = self.dataframe_dict[feature]

                dff.to_csv(self.feature_removed_average_csv[feature])

            out = dict()

            for feature in self.features_to_test:
                print((self.dataframe_dict[feature]))
                out[feature] = self.dataframe_dict[feature][self.Generate_Models_all_scenes.our_model_name]
            out["None removed"] = self.average_dataframe[self.Generate_Models_all_scenes.our_model_name]
            df = pd.DataFrame(out, self.test_prepositions + ["Average", "Overall"])
            df.to_csv(self.scores_tables_folder + "/functional_feature_analysis.csv")

            output_file = self.scores_plots_folder + "/ScoresWithRemovedFeatures.pdf"
            x_label = "Preposition"
            y_label = "Score"
            plot_title = "Average Scores With Removed Features. K = " + str(self.k) + " N = " + str(self.number_runs)
            self.plot_dataframe_bar_chart(df, output_file, x_label, y_label, plot_title)
        # ax = df.plot(kind='bar', title ="Average Scores With Removed Features. K = "+str(self.k) + " N = " + str(self.number_runs),figsize=(15, 10), legend=True)

        # ax.set_xlabel("Preposition")
        # ax.set_ylabel("Score")
        # ax.set_yticks(np.arange(0, 1.05, 0.05))
        # ax.grid(True)
        # ax.set_axisbelow(True)

        # plt.legend(loc='upper center', bbox_to_anchor=(0.44, -0.35), ncol=3)

        # plt.savefig(self.scores_plots_folder+"/ScoresWithRemovedFeatures.pdf", bbox_inches='tight')

    def plot_dataframe_bar_chart(self, dataset, file_to_save, x_label, y_label, plot_title):
        """Summary
        
        Args:
            dataset (TYPE): Description
            file_to_save (TYPE): Description
            x_label (TYPE): Description
            y_label (TYPE): Description
            plot_title (TYPE): Description
        """
        if self.features_to_test == None:
            new_column_order = self.Generate_Models_all_scenes.model_name_list
            reordered_df = dataset[new_column_order]
        else:
            reordered_df = dataset

        ax = reordered_df.plot(kind='bar', width=0.85, title=plot_title, figsize=(20, 10), legend=True)

        ax.set_xlabel(x_label, labelpad=10)
        ax.set_ylabel(y_label)
        ax.set_yticks(np.arange(0, 1.01, 0.05))
        ax.set_ylim([0, 1])
        ax.set_title(plot_title, pad=10)
        ax.grid(True)
        ax.set_axisbelow(True)

        plt.legend(loc='upper center', bbox_to_anchor=(0.44, -0.42), ncol=3)

        # plt.show()
        plt.savefig(file_to_save, bbox_inches='tight')

    def plot_bar_from_csv(self, file, file_to_save, x_label, y_label, plot_title, columns_to_drop=None):
        """Summary
        
        Args:
            file (TYPE): Description
            file_to_save (TYPE): Description
            x_label (TYPE): Description
            y_label (TYPE): Description
            plot_title (TYPE): Description
            columns_to_drop (None, optional): Description
        """
        dataset = pd.read_csv(file, index_col=0)
        if columns_to_drop is not None:
            dataset = dataset.drop(columns_to_drop, axis=1)
        self.plot_dataframe_bar_chart(dataset, file_to_save, x_label, y_label, plot_title)


def plot_preposition_graphs(study_info):
    """Summary
    
    Args:
        study_info (TYPE): Description
    """
    scene_list = study_info.scene_name_list
    generated_models = GenerateBasicModels(scene_list,scene_list,study_info)
    for p in preposition_list:
        M = generated_models.preposition_paramters_dict[p]
        M.output_models()
        M.plot_models()


def calculate_p_value(N, x):
    """Summary
    
    Args:
        N (TYPE): Description
        x (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    total = 0
    for i in range(x, N + 1):
        v = comb(N, i) * (math.pow(0.5, N))

        total += v
    return total


def test_features(study_info_):
    """Summary
    
    Args:
        study_info_ (TYPE): Description
    """
    functional_features = ["location_control", "support"]
    m = MultipleRuns(GenerateBasicModels, study_info_, number_runs=100, k=2, features_to_test=functional_features)
    print("Test Features")
    m.validation()
    m.output()


def initial_test(study_info_):
    """Summary
    
    Args:
        study_info_ (TYPE): Description
    """
    m = MultipleRuns(GenerateBasicModels, study_info_)
    print("Test on all scenes")
    m.test_all_scenes()


def test_models(study_info_):
    """Summary
    
    Args:
        study_info_ (TYPE): Description
    """
    m = MultipleRuns(GenerateBasicModels, study_info_, number_runs=100, k=2, compare="y")
    print("Test Model k = 2")
    m.validation()
    m.output()

    m = MultipleRuns(GenerateBasicModels, study_info_, number_runs=100, k=3, compare="y")
    print("Test Model k = 3")
    m.validation()
    m.output()


def plot_all_csv(study_info_):
    """Summary
    
    Args:
        study_info_ (TYPE): Description
    """
    m = MultipleRuns(GenerateBasicModels, study_info_)
    file = m.all_csv
    out_file = m.all_plot

    # self.plot_dataframe_bar_chart(self.all_dataframe,self.all_plot,"Preposition","Score","Scores Using All Data")
    m.plot_bar_from_csv(file, out_file, "Preposition", "Score", "Scores Using All Data")


def plot_kfold_csv(k, study_info_):
    """Summary
    
    Args:
        k (TYPE): Description
        study_info_ (TYPE): Description
    """
    m = MultipleRuns(GenerateBasicModels, study_info_, number_runs=100, k=k)
    file = m.average_csv
    out_file = m.average_plot_pdf

    m.plot_bar_from_csv(file, out_file, "Preposition", "Score", m.average_plot_title)


def plot_feature_csv(k, study_info_):
    """Summary
    
    Args:
        k (TYPE): Description
        study_info_ (TYPE): Description
    """
    functional_features = ["location_control", "support"]
    m = MultipleRuns(GenerateBasicModels, study_info_, number_runs=100, k=k, features_to_test=functional_features)
    file = m.scores_tables_folder + "/functional_feature_analysis.csv"
    output_file = m.scores_plots_folder + "/ScoresWithRemovedFeatures.pdf"
    x_label = "Preposition"
    y_label = "Score"
    plot_title = "Average Scores With Removed Features. K = " + str(m.k) + " N = " + str(m.number_runs)

    m.plot_bar_from_csv(file, output_file, x_label, y_label, plot_title)


def main(study_info_):
    """Summary
    
    Args:
        study_info_ (TYPE): Description
    """
    plot_preposition_graphs(study_info)
    # Edit plot settings
    mpl.rcParams['font.size'] = 40
    mpl.rcParams['legend.fontsize'] = 37
    mpl.rcParams['axes.titlesize'] = 'medium'
    mpl.rcParams['axes.labelsize'] = 'medium'
    mpl.rcParams['ytick.labelsize'] = 'small'

    initial_test(study_info_)
    # test_models(study_info_)
    # test_features(study_info_)


# plot_all_csv()
# plot_kfold_csv(2)
# plot_feature_csv(2)


if __name__ == '__main__':
    study_info = StudyInfo("2019 study")

    main(study_info)

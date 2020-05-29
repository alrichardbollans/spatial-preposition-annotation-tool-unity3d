"""Summary

Deleted Attributes:
    preposition_list (list): Description
"""
import itertools
import ast

import csv
import numpy as np
import pandas as pd

from data_import import StudyInfo, Configuration


class Constraint:
    """Summary
    
    Attributes:
        csv_row (TYPE): Description
        f1 (TYPE): Description
        f2 (TYPE): Description
        ground (TYPE): Description
        lhs (TYPE): Description
        parity (TYPE): Description
        preposition (TYPE): Description
        rhs (TYPE): Description
        scene (TYPE): Description
        titles (list): Description
        weak (TYPE): Description
        weight (TYPE): Description
    
    Deleted Attributes:
        folder_path (str): Description
        output_path (TYPE): Description
    """

    # A constraint is a linear inequality
    # Left hand side is more typical than right hand side
    # LHS is values of first config
    # RHS is values of second config

    def __init__(self, scene, preposition, ground, f1, f2, weight, lhs, rhs):
        """Summary
        
        Args:
            scene (TYPE): Description
            preposition (TYPE): Description
            ground (TYPE): Description
            f1 (TYPE): Description
            f2 (TYPE): Description
            weight (TYPE): Description
            lhs (dict): Set of features for (f1,ground) configuration
            rhs (dict): Set of features for (f2,ground) configuration
        """
        self.scene = scene
        self.preposition = preposition
        self.ground = ground
        # f1,f2 are figures being compared
        # f1 should be more typical than f2 to satisfy constraint
        self.f1 = f1
        self.f2 = f2
        # lhs and rhs are coefficients for the problem
        # coefficients are ordered by Configuration.feature_keys/relation_keys
        # These are configuration values for the instances being compared
        self.lhs = lhs
        self.rhs = rhs

        # Weight given to constraint
        self.weight = weight

        # Modify the column headings so they are unique
        self.lhs_columns = []
        for heading in list(self.lhs.keys()):
            self.lhs_columns.append(heading + "_1")

        self.rhs_columns = []
        for heading in list(self.rhs.keys()):
            self.rhs_columns.append(heading + "_2")

        self.csv_columns = ['scene', 'preposition', 'ground', 'f1', 'f2',
                            'weight'] + self.lhs_columns + self.rhs_columns
        self.csv_row = [
                           self.scene,
                           self.preposition,
                           self.ground,
                           self.f1,
                           self.f2,
                           self.weight] + list(self.lhs.values()) + list(self.rhs.values())

    def __str__(self):
        return ''.join(str(e) for e in self.csv_row)

    def write_to_csv(self, csv_file):
        """Summary Writes constraint to csv file. Appends to file if constraint isn't already included. Replaces row
        if constraint already included.

        Args: csv_file (TYPE): Description

        """

        try:
            # First attempt to find csv.
            original_df = pd.read_csv(csv_file, index_col=0)

        except Exception as failed_to_read:
            # If no csv exists create a new one.
            original_df = pd.DataFrame(columns=self.csv_columns)

        # Gives rows which match this constraint
        row_check = (original_df['scene'] == self.scene) & (original_df['preposition'] == self.preposition) & (
                original_df['ground'] == self.ground) & (original_df['f1'] == self.f1) & (original_df['f2'] == self.f2)
        single_constraint_df = original_df.loc[row_check, :]

        if single_constraint_df.empty:
            # If this constraint isn't already in the csv, append it

            new_row = pd.Series(self.csv_row, index=self.csv_columns)

            new_df = original_df.append(new_row, ignore_index=True)

        else:
            # If the constraint is already there then edit the row.
            original_df.loc[row_check, :] = self.csv_row

            new_df = original_df.copy()

        new_df.to_csv(csv_file)

    @staticmethod
    def read_from_csv(csv_file):
        """Summary
        
        Returns:
            TYPE: Description
        
        Args:
            csv_file (TYPE): Description
        """
        # Outputs dictionary of constraints from csv
        out = dict()
        for p in StudyInfo.preposition_list:
            out[p] = []
        dataset = pd.read_csv(csv_file, index_col=0)
        for index, line in dataset.iterrows():
            lhs = {}
            rhs = {}
            for feature in dataset.columns:
                # Columns ending in _1 denote feature values for f1 and _2 for f2
                if feature.endswith("_1"):
                    # Remove _1 for dictionary
                    feature_key = feature[:-2]
                    lhs[feature_key] = line[feature]

                if feature.endswith("_2"):
                    feature_key = feature[:-2]
                    rhs[feature_key] = line[feature]

            c = Constraint(line["scene"], line["preposition"], line["ground"], line["f1"], line["f2"], line["weight"],
                           lhs, rhs)
            

            out[c.preposition].append(c)
        return out

    def is_satisfied(self, lhs_value, rhs_value):
        """Summary
        Simple method to help maintain correct order. Returns true if lhs_value > rhs_value. Else returns False.
        Args:
            lhs_value (TYPE): Description
            rhs_value (TYPE): Description
        
        Returns:
            TYPE: Description
        """

        # LHS is more typical than RHS
        if lhs_value > rhs_value:
            return True
        else:
            return False

    # Returns the value of RHS-LHS, with values from X
    def separation(self, X):
        """Summary
        
        Args:
            X (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        lhs_value = np.dot(self.lhs, X)
        rhs_value = np.dot(self.rhs, X)

        return rhs_value - lhs_value

    # def write_constraint(self):

    # def read_constraints(self):


class Comparison:
    """Summary
    
    Attributes:
        chance_agreement (TYPE): Description
        chance_agreement_with_none (TYPE): Description
        ground (TYPE): Description
        possible_figures (TYPE): Description
        preposition (TYPE): Description
        scene (TYPE): Description
        study_info (TYPE): Description
    
    Deleted Attributes:
        study (TYPE): Description
    """

    def __init__(self, scene, preposition, ground, study):
        """Summary
        
        Args:
            scene (TYPE): Description
            preposition (TYPE): Description
            ground (TYPE): Description
            study (TYPE): Description
        """
        self.study_info = study
        self.preposition = preposition
        self.scene = scene
        self.ground = ground
        # self.annotations = self.get_annotations(datalist)
        # self.figure_selection_number = self.get_choices()
        self.possible_figures = self.get_possible_figures()
        # Note the following doesn't account for the fact that users can select none
        self.chance_agreement = 1 / float(len(self.possible_figures))
        self.chance_agreement_with_none = 1 / float(len(self.possible_figures) + 1)

    def generate_constraints(self, instancelist):
        """Summary
        
        Args:
            instancelist (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # Takes an annotation list
        # Generates constraints and outputs a list
        C = []
        instances = self.get_instances(instancelist)

        if len(instances) > 0:

            figure_selection_number = self.get_choices(instancelist)

            pairs = list(itertools.combinations(self.possible_figures, 2))

            for p in pairs:

                f1 = p[0]

                f2 = p[1]

                x1 = figure_selection_number[f1]

                x2 = figure_selection_number[f2]

                c1 = Configuration(self.scene, f1, self.ground, self.study_info)
                c2 = Configuration(self.scene, f2, self.ground, self.study_info)

                c1_features = c1.set_of_features
                c2_features = c2.set_of_features

                if x1 == x2:
                    pass

                elif x1 < x2:
                    weight = x2 - x1
                    con = Constraint(self.scene, self.preposition, self.ground, f2, f1, weight, c2_features,
                                     c1_features)
                    C.append(con)

                else:
                    weight = x1 - x2
                    con = Constraint(self.scene, self.preposition, self.ground, f1, f2, weight, c1_features,
                                     c2_features)
                    C.append(con)

        return C

    def get_instances(self, instancelist):
        """Summary
        
        Args:
            instancelist (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        out = []
        for i in instancelist:
            # if i.figure != "none":
            if self.instance_match(i):
                out.append(i)
        return out

    def get_none_instances(self, instancelist):
        """Summary
        
        Args:
            instancelist (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        out = []
        for i in instancelist:
            if i.figure == "none":
                if self.instance_match(i):
                    out.append(i)
        return out

    def instance_match(self, i):
        """Summary
        
        Args:
            i (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        if (
                i.scene == self.scene
                and i.preposition == self.preposition
                and i.ground == self.ground
        ):  #
            return True
        else:
            return False

    def get_choices(self, instancelist):
        """Summary
        
        Args:
            instancelist (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        out = dict()
        for f in self.possible_figures:
            out[f] = 0
        for i in self.get_instances(instancelist):
            # print(i.figure)
            f = i.figure
            if f in out:
                out[f] += 1
            else:
                out[f] = 1
        return out

    def get_possible_figures(self):
        """Summary
        
        Returns:
            TYPE: Description
        """

        out = []

        for my_scene in self.study_info.scene_list:
            if my_scene.name == self.scene:
                for g_object in my_scene.selectable_objects:
                    if g_object != self.ground and g_object not in out:
                        out.append(g_object)

        return out


class Instance(Configuration):
    """Summary
    
    Attributes:
        id (TYPE): Description
        preposition (TYPE): Description
        task (TYPE): Description
        user (TYPE): Description
    """

    def __init__(self, ID, user, task, scene, preposition, figure, ground, study):
        """Summary
        
        Args:
            ID (TYPE): Description
            user (TYPE): Description
            task (TYPE): Description
            scene (TYPE): Description
            preposition (TYPE): Description
            figure (TYPE): Description
            ground (TYPE): Description
            study (TYPE): Description
        """
        Configuration.__init__(self, scene, figure, ground, study)
        self.task = task
        self.preposition = preposition
        self.id = ID
        self.user = user

    def __str__(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        return (
                "["
                + str(self.scene)
                + ","
                + str(self.figure)
                + ","
                + str(self.ground)
                + "]"
        )

    def print_info(self):
        """Summary
        """
        Configuration.print_info()
        print(("preposition = " + self.preposition))
        print(("annotation id = " + self.id))
        print(("user = " + self.user))


class CompInstance(Configuration):
    """Summary
    
    Attributes:
        id (TYPE): Description
        possible_figures (TYPE): Description
        preposition (TYPE): Description
        task (TYPE): Description
        user (TYPE): Description
    """

    # Figure is the selected figure
    # We can consider this as a usual preposition instance and plot it
    # /compare with selection instances
    def __init__(
            self, ID, user, task, scene, preposition, figure, ground, possible_figures, study
    ):
        """Summary
        
        Args:
            ID (TYPE): Description
            user (TYPE): Description
            task (TYPE): Description
            scene (TYPE): Description
            preposition (TYPE): Description
            figure (TYPE): Description
            ground (TYPE): Description
            possible_figures (TYPE): Description
            study (TYPE): Description
        """
        Configuration.__init__(self, scene, figure, ground, study)
        self.possible_figures = possible_figures
        self.task = task
        self.preposition = preposition
        self.id = ID
        self.user = user

    def print_info(self):
        """Summary
        """
        Configuration.print_info(self)
        print(("preposition = " + self.preposition))
        print(("annotation id = " + self.id))
        print(("user = " + self.user))

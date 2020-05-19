"""Summary

Attributes:
    preposition_list (list): Description
"""
import itertools
import ast

import os
import csv
import numpy as np
import pandas as pd


preposition_list = [
    "in",
    "inside",
    "against",
    "on",
    "on top of",
    "under",
    "below",
    "over",
    "above",
]


def clean_name(object_name):
    """Summary
    
    Args:
        object_name (string): Description
    
    Returns:
        string: Description
    """
    if "." in object_name:
        new_clean_name = object_name[: object_name.find(".")]
    elif "_" in object_name:
        new_clean_name = object_name[: object_name.find("_")]
    else:
        new_clean_name = object_name
    return new_clean_name.lower()


def remove_dot_unity(scene_name):
    """Summary
    
    Args:
        scene_name (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    new_clean_name = scene_name
    if ".unity" in scene_name:
        new_clean_name = scene_name[: scene_name.find(".unity")]
    return new_clean_name


def get_git_project_directory():
    """
    Returns:
        str: Path to base unity project folder
    """
    unity_folder_name = "Unity Projects"
    repo_name = "spatial-preposition-annotation-tool-unity3d"
    current_directory = os.getcwd()
    user_home = os.path.expanduser("~")

    if os.path.basename(current_directory) == repo_name:
        return current_directory
    if os.path.basename(os.path.dirname(current_directory)) == repo_name:
        return os.path.dirname(current_directory)
    return user_home + "/Dropbox/" + unity_folder_name + "/" + repo_name


class BasicInfo:

    """Summary
    
    Attributes:
        a_index (dict): Description
        abv (TYPE): Description
        analysis_folder_name (str): Description
        base_data_folder_name (str): Description
        base_feature_data_folder_name (str): Description
        base_stats_folder_name (str): Description
        comp_annotations_name (str): Description
        comp_task (str): Description
        comparative_abbreviations (TYPE): Description
        comparative_preposition_list (TYPE): Description
        data_folder (TYPE): Description
        feature_data_folder (TYPE): Description
        feature_output_csv (TYPE): Description
        feature_output_folder (TYPE): Description
        human_readable_feature_output_csv (TYPE): Description
        preposition_list (list): Description
        project_path (TYPE): Description
        raw_annotation_csv (TYPE): Description
        raw_user_csv (TYPE): Description
        sem_annotations_name (str): Description
        semantic_abbreviations (TYPE): Description
        semantic_preposition_list (TYPE): Description
        stats_folder (TYPE): Description
        sv_task (str): Description
        svmod_annotations_name (str): Description
        svmod_task (str): Description
        typ_a_index (dict): Description
        typ_annotations_name (str): Description
        typ_task (str): Description
        unity_project_folder_name (str): Description
    """

    # Class containing basic info related to data collection
    unity_project_folder_name = "Unity3D Annotation Environment"
    analysis_folder_name = "Analysis"
    base_feature_data_folder_name = "Scene Data"

    base_data_folder_name = "collected data"
    sem_annotations_name = "clean semantic annotation list.csv"
    svmod_annotations_name = "clean svmod annotation list.csv"
    comp_annotations_name = "clean comparative annotation list.csv"
    typ_annotations_name = "clean typicality annotation list.csv"

    base_stats_folder_name = "stats"

    # Prepositions Used
    preposition_list = [
        "in",
        "inside",
        "against",
        "on",
        "on top of",
        "under",
        "below",
        "over",
        "above",
    ]  # list of prepositions which exist in the data

    semantic_preposition_list = preposition_list

    comparative_preposition_list = preposition_list

    # Task abbreviations
    sv_task = "sv"
    comp_task = "comp"
    typ_task = "typ"
    svmod_task = "sv_mod"
    semantic_abbreviations = [sv_task, svmod_task]

    comparative_abbreviations = [comp_task]

    # Dictionary giving the index of each value in annotations

    a_index = {
        "id": 0,
        "userid": 1,
        "time": 2,
        "figure": 3,
        "ground": 4,
        "task": 5,
        "scene": 6,
        "preposition": 7,
        "prepositions": 8,
        "cam_rot": 9,
        "cam_loc": 10,
    }
    # indexs for typ task
    typ_a_index = {
        "id": 0,
        "userid": 1,
        "time": 2,
        "c1": 3,
        "c2": 4,
        "task": 5,
        "selection": 6,
        "preposition": 7,
    }

    project_path = get_git_project_directory()

    def __init__(self, study):
        """Summary
        
        Args:
            study (TYPE): Description
        """
        # Abbreviation for files and folders
        self.abv = study

        # paths and filenames

        self.feature_data_folder = (
            self.project_path
            + "/"
            + self.analysis_folder_name
            + "/"
            + self.base_feature_data_folder_name
            + "/"
            + self.abv
        )

        self.feature_output_folder = "feature values" + "/" + self.abv
        # Path for outputting feature values
        self.feature_output_csv = (
            self.feature_output_folder + "/standardised_values.csv"
        )
        # Path for outputting human-readable feature values
        self.human_readable_feature_output_csv = (
            self.feature_output_folder + "/human_readable_values.csv"
        )

        self.data_folder = self.base_data_folder_name + "/" + self.abv
        self.raw_user_csv = self.data_folder + "/" + "userlist.csv"

        self.raw_annotation_csv = self.data_folder + "/" + "annotationlist.csv"

        self.stats_folder = self.base_stats_folder_name + "/" + self.abv

    # @staticmethod
    # def get_scene_list():
    #     """Summary

    #     Returns:
    #         TYPE: Description
    #     """
    #     scene_list = []

    #     s = SceneInfo()
    #     for scene in s.scene_list:
    #         scene_list.append(scene.name)
    #     return scene_list


class Relationship:
    """class which is used to read feature values
    
    Attributes:
        context_features (list): Description
        feature_keys (list): Description
        figure (TYPE): Description
        ground (TYPE): Description
        output_path (TYPE): Description
        property_path (TYPE): Description
        relation_keys (list): Description
        scene (TYPE): Description
        set_of_features (dict): Description
    """

    # Lots of this could be done with pandas. Doh :/

    # property_path = BasicInfo.feature_output_csv
    # output_path = property_path

    # additional_features = ["location_control"]
    # Location control is the average of the two more basic measures

    context_features = ["ground_lightsource", "ground_container", "ground_verticality"]

    def __init__(self, scene, figure, ground):
        """Summary
        
        Args:
            scene (TYPE): Description
            figure (TYPE): Description
            ground (TYPE): Description
        """
        self.scene = scene
        self.figure = figure
        self.ground = ground
        # Dictionary of features and values
        self.set_of_features = {}
        # Names of all features given by load_all
        self.feature_keys = []
        # Names of all features given by load_all, without above context features
        self.relation_keys = []

    @staticmethod
    def load_all(path):
        """Summary
        
        Args:
            path (None, optional): Description
        
        Returns:
            TYPE: Description
        """
        # Loads a list of all configurations and feature values, with some features removed
        # Path variable optional
        # if path == None:
        #     path = Relationship.property_path
        with open(path, "r") as f:
            reader = csv.reader(f)  # create a 'csv reader' from the file object
            geom_relations = list(reader)  # create a list from the reader

        return geom_relations

    @staticmethod
    def get_feature_keys():
        """Summary
        
        Returns:
            TYPE: Description
        """
        feature_keys = []

        geom_relations = Relationship.load_all()
        for title in geom_relations[0][3:]:
            feature_keys.append(title)

        return feature_keys

    @staticmethod
    def get_relation_keys():
        """Summary
        
        Returns:
            TYPE: Description
        """
        relation_keys = []

        geom_relations = Relationship.load_all()
        for title in geom_relations[0][3:]:
            if title not in Relationship.context_features:
                relation_keys.append(title)

        return relation_keys

    def load_from_csv(self, path):
        """Summary
        
        Args:
            path (None, optional): Description
        """

        geom_relations = Relationship.load_all(path)

        for title in geom_relations[0][3:]:
            self.feature_keys.append(title)
        for relation in geom_relations:
            if (
                self.scene == relation[0]
                and self.figure == relation[1]
                and self.ground == relation[2]
            ):
                # print(geom_relations.index(relation))
                for r in self.feature_keys:
                    if relation[self.feature_keys.index(r) + 3] != "?":
                        self.set_of_features[r] = float(
                            relation[self.feature_keys.index(r) + 3]
                        )
                    else:
                        self.set_of_features[r] = "?"
        # # Add and calculate additional features
        # self.feature_keys.append("location_control")
        # self.set_of_features["location_control"] = (self.set_of_features["location_control_x"] + self.set_of_features["location_control_z"])/2

    def save_to_csv(self):
        """Summary
        """
        row = [self.scene, self.figure, self.ground]

        for r in feature_keys:
            if r in self.set_of_features:
                row.append(self.set_of_features[r])
            else:
                row.append("?")
                self.set_of_features[r] = "?"

        with open(Relationship.output_path) as incsvfile:
            read = csv.reader(incsvfile)  # .readlines())
            reader = list(read)

            if any(
                self.scene == line[0]
                and self.figure == line[1]
                and self.ground == line[2]
                for line in reader
            ):
                try:
                    with open(Relationship.output_path, "w") as csvfile:
                        outputwriter = csv.writer(csvfile)
                        titles = ["scene", "figure", "ground"] + feature_keys
                        outputwriter.writerow(titles)
                        for line in reader[:]:
                            if "scene" not in line:
                                if (
                                    self.scene == line[0]
                                    and self.figure == line[1]
                                    and self.ground == line[2]
                                ):
                                    # Must ofset by 3 here due to each row beginning with scene and object names
                                    for x in range(0, len(feature_keys)):

                                        if self.set_of_features[feature_keys[x]] != "?":
                                            if len(line) > x + 3:
                                                line[x + 3] = self.set_of_features[
                                                    feature_keys[x]
                                                ]
                                            else:
                                                line.append(
                                                    self.set_of_features[
                                                        feature_keys[x]
                                                    ]
                                                )

                                outputwriter.writerow(line)
                except Exception as e:

                    print("Writing to CSV Failed")
                    print(("Figure: " + self.figure))
                    print(("Ground:" + self.ground))
                    print(e)
            else:
                with open(Relationship.output_path, "a") as csvfile:
                    outputwriter = csv.writer(csvfile)
                    outputwriter.writerow(row)


class Constraint:

    """Summary
    
    Attributes:
        csv_row (TYPE): Description
        f1 (TYPE): Description
        f2 (TYPE): Description
        folder_path (str): Description
        ground (TYPE): Description
        lhs (TYPE): Description
        output_path (TYPE): Description
        parity (TYPE): Description
        preposition (TYPE): Description
        rhs (TYPE): Description
        scene (TYPE): Description
        titles (list): Description
        weak (TYPE): Description
        weight (TYPE): Description
    """

    # A constraint is a linear inequality
    # Left hand side is more typical than right hand side
    # LHS is values of first config
    # RHS is values of second config
    folder_path = "constraint data/"
    output_path = folder_path + "constraints.csv"
    titles = [
        "scene",
        "preposition",
        "ground",
        "f1",
        "f2",
        "weight",
        "parity",
        "weak",
        "lhs",
        "rhs",
    ]

    def __init__(
        self, scene, preposition, ground, f1, f2, weight, parity, weak, lhs, rhs
    ):
        """Summary
        
        Args:
            scene (TYPE): Description
            preposition (TYPE): Description
            ground (TYPE): Description
            f1 (TYPE): Description
            f2 (TYPE): Description
            weight (TYPE): Description
            parity (TYPE): Description
            weak (TYPE): Description
            lhs (TYPE): Description
            rhs (TYPE): Description
        """
        self.scene = scene
        self.preposition = preposition
        self.ground = ground
        # f1,f2 are figures being compared
        # f1 should be more typical than f2 to satisfy constraint
        self.f1 = f1
        self.f2 = f2
        # lhs and rhs are arrays of coefficients for the problem
        # coefficients are ordered by Relationship.feature_keys/relation_keys
        # These are configuration values for the instances being compared
        self.lhs = lhs  # np.array([lhs])
        self.rhs = rhs  # np.array([rhs])

        # Weight given to constraint
        self.weight = weight

        # We have two types of constraints
        # Some arise from difference in the data and some from similarity

        self.parity = parity

        # If True, this constraint is for weak instances i.e. LHS is some distance from prototype
        # RHS is not important
        self.weak = weak

        self.csv_row = [
            self.scene,
            self.preposition,
            self.ground,
            self.f1,
            self.f2,
            self.weight,
            str(parity),
            str(weak),
            lhs.tolist(),
            rhs.tolist(),
        ]

    # def row_match(self,row):
    #   if self.scene== row[0] and self.f1 == row[1] and self.f2 == row[2]
    def write_to_csv(self, name=None):
        """Summary
        
        Args:
            name (None, optional): Description
        """
        if name == None:
            path = self.output_path
        else:
            path = self.folder_path + name + ".csv"

        with open(path, "a") as csvfile:
            outputwriter = csv.writer(csvfile)
            outputwriter.writerow(self.csv_row)

    @staticmethod
    def read_from_csv():
        """Summary
        
        Returns:
            TYPE: Description
        """
        # Outputs dictionary of constraints from csv
        out = dict()
        for p in preposition_list:
            out[p] = []
        dataset = pd.read_csv(Constraint.output_path)
        for index, line in dataset.iterrows():
            lhs_list = ast.literal_eval(line["lhs"])
            rhs_list = ast.literal_eval(line["rhs"])
            lhs = np.array(lhs_list)
            rhs = np.array(rhs_list)

            c = Constraint(
                line["scene"],
                line["preposition"],
                line["ground"],
                line["f1"],
                line["f2"],
                line["weight"],
                line["parity"],
                line["weak"],
                lhs,
                rhs,
            )

            out[c.preposition].append(c)
        return out

    @staticmethod
    def clear_csv(name=None):
        """Summary
        
        Args:
            name (None, optional): Description
        """
        if name == None:
            path = Constraint.output_path
        else:
            path = Constraint.folder_path + name + ".csv"
        with open(path, "w") as csvfile:
            outputwriter = csv.writer(csvfile)
            outputwriter.writerow(Constraint.titles)
            # csvfile.truncate()

    # def get_values(self,W,P):
    #   # Works out the Similarity of LHS and RHS to prototype,P, given feature weights W
    #   # Works out the Euclidean distance of the instance represented by LHS and RHS to the prototype
    #   # W is a 1D array, an assignment of weights and P, Prototype
    #   # First take away P from lhs and rhs values
    #   # Then do dot product

    #   lhs = np.subtract(self.lhs,P)
    #   rhs = np.subtract(self.rhs,P)

    #   lhs = np.square(lhs)
    #   rhs = np.square(rhs)

    #   lhs_distance = math.sqrt(np.dot(lhs,W))
    #   rhs_distance = math.sqrt(np.dot(rhs,W))

    #   lhs_value = math.exp(-lhs_distance)
    #   rhs_value = math.exp(-rhs_distance)

    #   # print(lhs_value)

    #   return [lhs_value,rhs_value]

    # Returns True if X satisfies C
    # False otherwise

    def is_satisfied(self, lhs_value, rhs_value):
        """Summary
        
        Args:
            lhs_value (TYPE): Description
            rhs_value (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # Values get calculated elsewhere depending on model
        # values = self.get_values(W,P)
        # lhs_value = values[0]
        # rhs_value = values[1]
        if self.weak:
            if lhs_value < 0.8:
                return True
            else:
                return False
        elif self.parity:
            if abs(lhs_value - rhs_value) < 0.5:
                return True
            else:
                return False
        else:
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
        self.study = study
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
        # For the moment the metric constraint are commented out
        # Don't want to generate constraints when no tests have been done!
        if len(instances) > 0:

            figure_selection_number = self.get_choices(instancelist)

            pairs = list(itertools.combinations(self.possible_figures, 2))

            for p in pairs:

                f1 = p[0]

                f2 = p[1]

                x1 = figure_selection_number[f1]

                x2 = figure_selection_number[f2]

                c1 = Configuration(self.scene, f1, self.ground)
                c2 = Configuration(self.scene, f2, self.ground)

                # Only use values that are not from 'context' features
                c1_array = np.array(c1.relations_row)
                c2_array = np.array(c2.relations_row)

                if x1 == x2:
                    pass
                    # if any (x > x1 for x in figure_selection_number.itervalues()):
                    #   # This case may happen because there is a better figure
                    #   # In which case we create no constraint
                    #   pass
                    # elif x1 != 0:
                    #   # In the case that x1,x2 are not zero
                    #   # They are similar enough instances of the preposition to be confused

                    #   weight = x1

                    #   con = Constraint(self.scene,self.preposition,self.ground,f1,f2,weight,True,False,c1_array,c2_array)
                    #   C.append(con)
                    #   con.write_to_csv()
                elif x1 < x2:
                    weight = x2 - x1
                    con = Constraint(
                        self.scene,
                        self.preposition,
                        self.ground,
                        f2,
                        f1,
                        weight,
                        False,
                        False,
                        c2_array,
                        c1_array,
                    )
                    C.append(con)

                else:
                    weight = x1 - x2
                    con = Constraint(
                        self.scene,
                        self.preposition,
                        self.ground,
                        f1,
                        f2,
                        weight,
                        False,
                        False,
                        c1_array,
                        c2_array,
                    )
                    C.append(con)

            # Deal with case that figure is not ever selected
            # for fig in figure_selection_number:
            #   x1 = figure_selection_number[fig]
            #   if x1 == 0:
            #       pass
            #       c = Configuration(self.scene,fig,self.ground)

            #       # Only use values that are not from 'context' features
            #       c_array = np.array(c.relations_row)

            #       # Add a constraint saying this is a weak configuration
            #       weight = float(len(instances))/len(self.possible_figures)
            #       con = Constraint(self.scene,self.preposition,self.ground,fig,fig,weight,False,True,c_array,c_array)

            #       C.append(con)
        # if len(none_instances) > 0:

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
        s_info = SceneInfo(self.study)
        out = []

        for my_scene in s_info.scene_list:
            if my_scene.name == self.scene:
                for g_object in my_scene.selectable_objects:
                    if g_object != self.ground and g_object not in out:
                        out.append(g_object)

                
        return out


class MyScene:

    """Summary
    
    Attributes:
        example_scenes (list): Description
        mesh_objects (TYPE): Description
        name (TYPE): Description
        selectable_objects (TYPE): Description
        study_scene (TYPE): Description
        unselectable_scene_objects (list): Description
    """

    # Class to store info about scenes
    unselectable_scene_objects = ["wall", "floor", "ceiling"]
    # This is given in MyScene class in Unity

    example_scenes = [
        "example",
        "finish",
        "instruction",
        "template",
        "main",
        "player",
        "screen",
        "test",
    ]

    def __init__(self, name, mesh_objects):
        """Summary
        
        Args:
            name (TYPE): Description
            mesh_objects (TYPE): Description
        """
        self.name = name
        self.mesh_objects = mesh_objects
        self.selectable_objects = self.get_selectable_objects()  #
        # Bool. Is this scene used for study?
        self.study_scene = self.study_scene_check()

    def get_selectable_objects(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        out = []
        for obj in self.mesh_objects:
            if not any(x in obj for x in self.unselectable_scene_objects):
                out.append(obj)
        return out

    def get_all_configs(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        out = []

        x = list(itertools.permutations(self.selectable_objects, 2))

        return x

    def study_scene_check(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        if any(x in self.name for x in self.example_scenes):
            return False
            # print(self.name)
        else:
            return True
            # print(self.name)


class SceneInfo:

    """Summary
    A class to store info on the scenes
    scene_list is a collection of MyScene objects
    Relies on creating a csv file in Unity Editor using write_scene_info.cs script
    Then also run commonsense properties script to get ground info
    
    Attributes:
        basic_info (TYPE): Description
        csv_file (TYPE): Description
        data_list (TYPE): Description
        filename (TYPE): Description
        output_path (TYPE): Description
    
    Deleted Attributes:
        scene_list (TYPE): Description
    """

    filename = "scene_info.csv"

    def __init__(self, study):
        """Summary
        
        Args:
            study (TYPE): Description
        """
        self.basic_info = BasicInfo(study)
        self.output_path = self.basic_info.feature_data_folder
        self.csv_file = self.output_path + "/" + self.filename
        self.data_list = self.get_list()
        self.scene_list, self.scene_name_list = self.get_scenes()

    def get_list(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        with open(self.csv_file, "r") as file:
            reader = csv.reader(file)
            datalist = list(reader)
        return datalist

    def get_scenes(self):
        """Summary
        
        Returns:
            list: Description
        """
        scenes_out = []
        names_out = []

        for scene_info in self.data_list:
            mesh_objects = []
            for i in range(1, len(scene_info)):
                mesh_objects.append(scene_info[i])
            if len(mesh_objects) != 0 and ".unity" in scene_info[0]:
                s_name = remove_dot_unity(scene_info[0])
                my_scene = MyScene(s_name, mesh_objects)

                if my_scene.study_scene:

                    scenes_out.append(my_scene)
                    names_out.append(my_scene.name)

        return scenes_out, names_out


class Configuration:

    """Summary
    
    Attributes:
        figure (TYPE): Description
        full_row (TYPE): Description
        ground (TYPE): Description
        relations_row (list): Description
        row (list): Description
        scene (TYPE): Description
    """

    def __init__(self, scene, figure, ground, study):
        """Summary
        
        Args:
            scene (TYPE): Description
            figure (TYPE): Description
            ground (TYPE): Description
            path (TYPE, optional): Description
        """
        self.scene = scene
        self.figure = figure
        self.ground = ground

        self.path = BasicInfo(study).feature_output_csv
        # Row of feature values for outputing to csv
        self.row = []
        # Row beginning with names
        self.full_row = [self.scene, self.figure, self.ground]
        # Row without context features
        self.relations_row = []
        if self.figure != "none":
            self.append_values()

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

    def append_values(self):
        """Summary
        
        Args:
            path (TYPE): Description
        """
        # print(self.figure)
        # print(self.ground)
        # print(self.scene)
        r = Relationship(self.scene, self.figure, self.ground)
        r.load_from_csv(path=self.path)

        for key in r.feature_keys:

            value = r.set_of_features[key]
            setattr(self, key, value)
            self.row.append(value)
            self.full_row.append(value)
            if key not in r.context_features:
                self.relations_row.append(value)

    def configuration_match(self, instance):
        """Summary
        
        Args:
            instance (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        if (
            self.scene == instance.scene
            and self.figure == instance.figure
            and self.ground == instance.ground
        ):
            return True
        else:
            return False

    def number_of_selections(self, preposition, instancelist):
        """Summary
        
        Args:
            preposition (TYPE): Description
            instancelist (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        counter = 0
        for i in instancelist:
            if self.configuration_match(i) and i.preposition == preposition:
                counter += 1
        return counter

    def number_of_selections_from_annotationlist(self, preposition, annotationlist):
        """Summary
        
        Args:
            preposition (TYPE): Description
            annotationlist (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        counter = 0
        for an in annotationlist:
            if self.annotation_row_match(an) and preposition in an.prepositions:

                counter += 1
        return counter

    def number_of_tests(self, annotationlist):
        """Summary
        
        Args:
            annotationlist (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # Need to use annotation list here as instances are separated by preposition
        counter = 0
        for an in annotationlist:

            if self.annotation_row_match(an):

                counter += 1
        # print(counter)
        return counter

    def ratio_semantic_selections(self, preposition, annotationlist, instancelist):
        """Summary
        
        Args:
            preposition (TYPE): Description
            annotationlist (TYPE): Description
            instancelist (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        t = float(self.number_of_tests(annotationlist))
        s = float(self.number_of_selections(preposition, instancelist))

        if t != 0:
            return s / t
        else:
            return 0

    def annotation_row_match(self, row):
        """Summary
        
        Args:
            row (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # If configuration matches with annotation in raw data list
        if self.scene == row[3] and self.figure == row[5] and self.ground == row[6]:
            return True
        else:
            return False

    def config_row_match(self, value):
        """Summary
        
        Args:
            value (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        if (
            self.scene == value[0]
            and self.figure == value[1]
            and self.ground == value[2]
        ):
            return True
        else:
            return False

    # def create_row(self):
    #   for value in self.value_names:
    #       try:
    #           self.row.append(getattr(self,value))
    #       except Exception as e:
    #           self.row.append('?')
    #           print('Value not added')
    #           print('Figure: ' + self.figure)
    #           print('Ground: ' + self.ground)

    #           print('Scene: ' + self.scene)
    #           print('Value: ' + value)

    #           print(e)
    def print_info(self):
        """Summary
        """
        print(("Scene = " + self.scene))
        print(("Figure = " + self.figure))
        print(("Ground = " + self.ground))


class Instance(Configuration):

    """Summary
    
    Attributes:
        id (TYPE): Description
        preposition (TYPE): Description
        task (TYPE): Description
        user (TYPE): Description
    """

    def __init__(self, ID, user, task, scene, preposition, figure, ground):
        """Summary
        
        Args:
            ID (TYPE): Description
            user (TYPE): Description
            task (TYPE): Description
            scene (TYPE): Description
            preposition (TYPE): Description
            figure (TYPE): Description
            ground (TYPE): Description
        """
        Configuration.__init__(self, scene, figure, ground)
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
        self, ID, user, task, scene, preposition, figure, ground, possible_figures
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
        """
        Configuration.__init__(self, scene, figure, ground)
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

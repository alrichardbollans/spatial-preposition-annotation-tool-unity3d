"""Summary
Methods and classes for loading the data, related to data format.
"""

import os
import csv


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
    
    # Class containing basic info related to data collection
    
    Attributes:
        a_index (dict): Description
        analysis_folder_name (str): Description
        base_collected_data_folder_name (str): Description
        base_feature_data_folder_name (str): Description
        base_stats_folder_name (str): Description
        comp_annotations_name (str): Description
        comp_task (str): Description
        comparative_abbreviations (TYPE): Description
        comparative_preposition_list (TYPE): Description
        config_ratio_folder (TYPE): Description
        constraint_csv (TYPE): Description
        data_folder (TYPE): Description
        feature_output_csv (TYPE): Description
        feature_output_folder (TYPE): Description
        human_readable_feature_output_csv (TYPE): Description
        input_feature_csv (TYPE): Description
        input_feature_data_folder (TYPE): Description
        preposition_list (list): Description
        project_path (TYPE): Description
        raw_annotation_csv (TYPE): Description
        raw_user_csv (TYPE): Description
        sem_annotations_name (str): Description
        semantic_abbreviations (TYPE): Description
        semantic_preposition_list (TYPE): Description
        stats_folder (TYPE): Description
        study (TYPE): Description
        study_list (list): Description
        sv_task (str): Description
        svmod_annotations_name (str): Description
        svmod_task (str): Description
        typ_a_index (dict): Description
        typ_annotations_name (str): Description
        typ_task (str): Description
        u_index (TYPE): Description
        unity_project_folder_name (str): Description
    """

    unity_project_folder_name = "Unity3D Annotation Environment"
    analysis_folder_name = "Analysis"
    base_feature_data_folder_name = "Scene Data"

    base_collected_data_folder_name = "collected data"
    sem_annotations_name = "clean semantic annotation list.csv"
    svmod_annotations_name = "clean svmod annotation list.csv"
    comp_annotations_name = "clean comparative annotation list.csv"
    typ_annotations_name = "clean typicality annotation list.csv"

    base_stats_folder_name = "stats"

    # Prepositions
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

    semantic_preposition_list = preposition_list

    comparative_preposition_list = preposition_list

    # Task abbreviations
    sv_task = "sv"
    comp_task = "comp"
    typ_task = "typ"
    svmod_task = "sv_mod"
    semantic_abbreviations = [sv_task, svmod_task]

    comparative_abbreviations = [comp_task]

    # Dictionary giving the index of each value in raw annotations

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

    # Dictionary giving index of values in raw userlist

    u_index = {
        "user_id": 0,
        "time": 1,
        "native": 2
    }
    study_list = ["2019 study", "2020 study"]
    project_path = get_git_project_directory()

    def __init__(self, study):
        """Summary
        
        Args:
            study (TYPE): Description
        """
        # Abbreviation for files and folders
        self.study = study

        # paths and filenames

        self.input_feature_data_folder = (

                self.study
                + "/"
                + self.base_feature_data_folder_name
        )

        self.input_feature_csv = self.input_feature_data_folder + "/relations.csv"

        self.feature_output_folder = self.study + "/" + "feature values"
        # Path for outputting feature values
        self.feature_output_csv = (
                self.feature_output_folder + "/standardised_values.csv"
        )
        # Path for outputting human-readable feature values
        self.human_readable_feature_output_csv = (
                self.feature_output_folder + "/human_readable_values.csv"
        )

        self.data_folder = self.study + "/" + self.base_collected_data_folder_name
        self.raw_user_csv = self.data_folder + "/" + "userlist.csv"

        self.raw_annotation_csv = self.data_folder + "/" + "annotationlist.csv"

        self.stats_folder = self.study + "/" + self.base_stats_folder_name

        self.config_ratio_folder = self.study + "/" + 'preposition data/'

        self.constraint_csv = self.study + "/constraint data/constraints.csv"

        self.model_info_folder = self.study + "/model info"

    def config_ratio_csv(self, filetag, preposition):
        """Summary
        
        Args:
            filetag (TYPE): Description
            preposition (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        return self.config_ratio_folder + filetag + '-ratio-list' + preposition + ' .csv'
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
        data_path (TYPE): Description
        feature_keys (list): Description
        figure (TYPE): Description
        ground (TYPE): Description
        relation_keys (list): Description
        scene (TYPE): Description
        set_of_features (dict): Description
        study (TYPE): Description
    
    Deleted Attributes:
        output_path (TYPE): Description
        input_feature_csv (TYPE): Description
    """

    # Lots of this could be done with pandas. Doh :/

    # input_feature_csv = BasicInfo.feature_output_csv
    # output_path = input_feature_csv

    # additional_features = ["location_control"]
    # Location control is the average of the two more basic measures

    context_features = ["ground_lightsource", "ground_container", "ground_verticality"]

    def __init__(self, scene, figure, ground, study):
        """Summary
        
        Args:
            scene (TYPE): Description
            figure (TYPE): Description
            ground (TYPE): Description
            study (TYPE): Description
        """
        self.study = study
        self.data_path = BasicInfo(study).feature_output_csv
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
    def load_all(study):
        """Summary
        
        Args:
            study (TYPE): Description
        
        Returns:
            TYPE: Description
        
        Deleted Parameters:
            path (None, optional): Description
        """
        # Loads a list of all configurations and feature values, with some features removed
        path = BasicInfo(study).feature_output_csv
        with open(path, "r") as f:
            reader = csv.reader(f)  # create a 'csv reader' from the file object
            geom_relations = list(reader)  # create a list from the reader

        return geom_relations

    @staticmethod
    def get_feature_keys(study):
        """Summary
        
        Returns:
            TYPE: Description
        
        Args:
            study (TYPE): Description
        """
        feature_keys = []

        geom_relations = Relationship.load_all(study)
        for title in geom_relations[0][3:]:
            feature_keys.append(title)

        return feature_keys

    @staticmethod
    def get_relation_keys(study):
        """Summary
        
        Returns:
            TYPE: Description
        
        Args:
            study (TYPE): Description
        """
        relation_keys = []

        geom_relations = Relationship.load_all(study)
        for title in geom_relations[0][3:]:
            if title not in Relationship.context_features:
                relation_keys.append(title)

        return relation_keys

    def load_from_csv(self):
        """Summary
        
        Deleted Parameters:
            path (None, optional): Description
        """

        geom_relations = Relationship.load_all(self.study)

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

    # def save_to_csv(self):
    #     """Summary
    #     """
    #     row = [self.scene, self.figure, self.ground]
    #
    #     for r in feature_keys:
    #         if r in self.set_of_features:
    #             row.append(self.set_of_features[r])
    #         else:
    #             row.append("?")
    #             self.set_of_features[r] = "?"
    #
    #     with open(self.data_path) as incsvfile:
    #         read = csv.reader(incsvfile)  # .readlines())
    #         reader = list(read)
    #
    #         if any(
    #                 self.scene == line[0]
    #                 and self.figure == line[1]
    #                 and self.ground == line[2]
    #                 for line in reader
    #         ):
    #             try:
    #                 with open(Relationship.output_path, "w") as csvfile:
    #                     outputwriter = csv.writer(csvfile)
    #                     titles = ["scene", "figure", "ground"] + feature_keys
    #                     outputwriter.writerow(titles)
    #                     for line in reader[:]:
    #                         if "scene" not in line:
    #                             if (
    #                                     self.scene == line[0]
    #                                     and self.figure == line[1]
    #                                     and self.ground == line[2]
    #                             ):
    #                                 # Must ofset by 3 here due to each row beginning with scene and object names
    #                                 for x in range(0, len(feature_keys)):
    #
    #                                     if self.set_of_features[feature_keys[x]] != "?":
    #                                         if len(line) > x + 3:
    #                                             line[x + 3] = self.set_of_features[
    #                                                 feature_keys[x]
    #                                             ]
    #                                         else:
    #                                             line.append(
    #                                                 self.set_of_features[
    #                                                     feature_keys[x]
    #                                                 ]
    #                                             )
    #
    #                             outputwriter.writerow(line)
    #             except Exception as e:
    #
    #                 print("Writing to CSV Failed")
    #                 print(("Figure: " + self.figure))
    #                 print(("Ground:" + self.ground))
    #                 print(e)
    #         else:
    #             with open(Relationship.output_path, "a") as csvfile:
    #                 outputwriter = csv.writer(csvfile)
    #                 outputwriter.writerow(row)

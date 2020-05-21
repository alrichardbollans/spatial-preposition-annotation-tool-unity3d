"""Summary

Script to run for newly collected data files which:
    Input: annotation and user info csv from data collection
    Output: Clean annotation lists. Basic stats. User agreement calculations
    Feature values are included later
"""

import csv
import itertools
import scipy

from classes import Comparison, SceneInfo
from data_import import BasicInfo


class User:
    """
    A class to store user information.
    
    Attributes:
        clean_user_id (TYPE): Human readable ID
        list_format (TYPE): Formatting of output list.
        list_headings (TYPE): Headings for outputting
        native (TYPE): 1 if native, else 0.
        time (TYPE): Time user began.
        user_id (TYPE): ID assigned by annotation tool.
    
    
    """

    list_headings = ["User ID", "Short ID", "Time", "Native=1, Non-Native =0"]

    def __init__(
            self, clean_id, user_row
    ):  # The order of this should be the same as in writeuserdata.php
        """Summary
        
        Args:
            clean_id (str): Description
            user_row (TYPE): Description
        
        Deleted Parameters:
            user_id (str): Description
            time (str): Description
            native (str): Description
        """
        self.clean_user_id = clean_id
        self.user_id = user_row[BasicInfo.u_index["user_id"]]
        self.time = user_row[BasicInfo.u_index["time"]]

        self.native = user_row[BasicInfo.u_index["native"]]

        self.list_format = [self.user_id, self.clean_user_id, self.time, self.native]

    def annotation_match(self, annotation):
        """
        Checks if the annotation was made by this user.
        
        Args:
            annotation (Annotation): Annotation to check.
        
        Returns:
            bool: True if match, else False.
        """
        if (
                annotation.clean_user_id == self.user_id
                or annotation.clean_user_id == self.clean_user_id
        ):
            return True
        else:
            return False


class UserData:
    """Summary
    
    Attributes:
        basic_info (TYPE): Description
        raw_data_list (TYPE): Description
        study (TYPE): Description
        user_list (TYPE): Description
    """

    def __init__(self, study):
        """Summary
        
        Args:
            study (str): Name of
        """
        self.study = study
        self.basic_info = BasicInfo(study)
        self.raw_data_list = self.load_raw_users_from_csv()
        self.user_list = self.get_users()

    def load_raw_users_from_csv(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        with open(self.basic_info.raw_user_csv, "r") as f:
            reader = csv.reader(f)
            datalist = list(reader)
        return datalist

    def get_users(self):  # ,datalist):
        """Summary
        
        Returns:
            TYPE: Description
        """

        out = []
        i = 1
        for user_row in self.raw_data_list:
            u = User(i, user_row)

            out.append(u)
            i += 1

        return out

    def output_clean_user_list(self):
        """Summary
        """

        with open(
                self.basic_info.data_folder + "/" + "clean_users.csv", "w"
        ) as csvfile:
            writer = csv.writer(csvfile)

            heading = User.list_headings
            writer.writerow(heading)

            for user in self.user_list:
                writer.writerow(user.list_format)

    def get_non_natives(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        out = []
        for u in self.user_list:
            if u.native == "0":
                out.append(u.clean_user_id)
                print(("Non-Native: " + str(u.clean_user_id)))
        return out


class Annotation:
    """Summary
    
    Attributes:
        c1 (TYPE): Description
        c2 (TYPE): Description
        cam_loc (TYPE): Description
        cam_rot (TYPE): Description
        clean_user_id (TYPE): Description
        figure (str): Description
        ground (TYPE): Description
        id (TYPE): Description
        list_format (TYPE): Description
        list_headings (TYPE): Description
        preposition (TYPE): Description
        prepositions (TYPE): Description
        scene (TYPE): Description
        selection (str): Description
        task (TYPE): Description
        time (TYPE): Description
        user (TYPE): Description
        user_id (TYPE): Description
    """

    # Gets annotations from the raw data
    list_headings = [
        "Annotation ID",
        "Clean User ID",
        "Task",
        "Scene",
        "Preposition",
        "Prepositions",
        "Figure",
        "Ground",
        "Time",
    ]

    def __init__(
            self, userdata, annotation
    ):
        """Summary
        
        Args:
            userdata (TYPE): Description
            annotation (TYPE): Description
        """
        self.task = annotation[BasicInfo.typ_a_index["task"]]

        if self.task == "typ":
            self.id = annotation[BasicInfo.typ_a_index["id"]]
            self.user_id = annotation[BasicInfo.typ_a_index["userid"]]
            self.time = annotation[BasicInfo.typ_a_index["time"]]
            self.task = annotation[BasicInfo.typ_a_index["task"]]
            self.preposition = annotation[BasicInfo.typ_a_index["preposition"]]

            for user in userdata.user_list:
                if user.user_id == self.user_id:
                    self.user = user
                    self.clean_user_id = user.clean_user_id

            self.c1 = annotation[BasicInfo.typ_a_index["c1"]]
            self.c2 = annotation[BasicInfo.typ_a_index["c2"]]
            self.selection = annotation[BasicInfo.typ_a_index["selection"]]
            if self.selection == "":
                self.selection = "none"
            self.list_format = [
                self.id,
                self.clean_user_id,
                self.task,
                self.preposition,
                self.c1,
                self.c2,
                self.selection,
                self.time,
            ]

        else:

            self.id = annotation[BasicInfo.a_index["id"]]

            self.user_id = annotation[BasicInfo.a_index["userid"]]

            self.time = annotation[BasicInfo.a_index["time"]]
            self.task = annotation[BasicInfo.a_index["task"]]
            self.preposition = annotation[BasicInfo.a_index["preposition"]]

            for user in userdata.user_list:
                if user.user_id == self.user_id:
                    self.user = user
                    self.clean_user_id = user.clean_user_id

            selectedFigure = annotation[BasicInfo.a_index["figure"]]
            if selectedFigure == "":
                self.figure = "none"

            else:
                self.figure = selectedFigure
            self.ground = annotation[BasicInfo.a_index["ground"]]

            self.scene = annotation[BasicInfo.a_index["scene"]]

            self.prepositions = annotation[BasicInfo.a_index["prepositions"]]

            self.cam_rot = annotation[BasicInfo.a_index["cam_rot"]]
            self.cam_loc = annotation[BasicInfo.a_index["cam_loc"]]

            self.list_format = [
                self.id,
                self.clean_user_id,
                self.task,
                self.scene,
                self.preposition,
                self.prepositions,
                self.figure,
                self.ground,
                self.time,
            ]


class ComparativeAnnotation(Annotation):
    """Summary
    
    Attributes:
        list_format (TYPE): Description
        list_headings (TYPE): Description
        possible_figures (TYPE): Description
    """

    list_headings = [
        "Annotation ID",
        "Clean User ID",
        "Task",
        "Scene",
        "Preposition",
        "Figure",
        "Ground",
        "Time",
    ]

    # Users selects a figure given a ground and preposition
    def __init__(
            self, userdata, annotation
    ):  # ID,UserID,now,selectedFigure,selectedGround,scene,preposition,prepositions,cam_rot,cam_loc):
        """Summary
        
        Args:
            userdata (TYPE): Description
            annotation (TYPE): Description
        """
        Annotation.__init__(self, userdata, annotation)
        # list format is used to write rows of csv

        # Need to append possible figures to list format and then deal with this in compile_instances
        self.list_format = [
            self.id,
            self.clean_user_id,
            self.task,
            self.scene,
            self.preposition,
            self.figure,
            self.ground,
            self.time,
        ]

        c = Comparison(self.scene, self.preposition, self.ground, userdata.study)
        self.possible_figures = c.possible_figures
        for f in self.possible_figures:
            self.list_format.append(f)

    @staticmethod
    def retrieve_from_data_row(row):
        """Summary
        
        Args:
            row (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        an_id = row[0]
        clean_user_id = row[1]
        task = row[2]
        scene = row[3]
        preposition = row[4]
        figure = row[5]
        ground = row[6]
        time = row[7]

        possible_figures = []
        index = 8
        while index < len(row):
            possible_figures.append(row[index])
            index += 1

        return an_id, clean_user_id, task, scene, preposition, figure, ground, time, possible_figures

    def print_list(self):
        """Summary
        """
        print(
            [
                self.id,
                self.clean_user_id,
                self.preposition,
                self.scene,
                self.figure,
                self.ground,
                self.time,
            ]
        )


class SemanticAnnotation(Annotation):
    """Summary
    
    Attributes:
        list_format (TYPE): Description
        list_headings (TYPE): Description
        preposition_list (TYPE): Description
    """

    list_headings = [
        "Annotation ID",
        "Clean User ID",
        "Task",
        "Scene",
        "Prepositions",
        "Figure",
        "Ground",
        "Time",
    ]

    # User selects multiple prepositions given a figure and ground
    def __init__(
            self, userdata, annotation
    ):
        """Summary
        
        Args:
            userdata (TYPE): Description
            annotation (TYPE): Description
        """
        Annotation.__init__(self, userdata, annotation)
        self.preposition_list = self.make_preposition_list()
        self.list_format = [
            self.id,
            self.clean_user_id,
            self.task,
            self.scene,
            self.prepositions,
            self.figure,
            self.ground,
            self.time,
        ]

    @staticmethod
    def retrieve_from_data_row(row):
        """Summary
        
        Args:
            row (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        an_id = row[0]
        clean_user_id = row[1]
        task = row[2]
        scene = row[3]
        prepositions = row[4]
        figure = row[5]
        ground = row[6]
        time = row[7]

        return an_id, clean_user_id, task, scene, prepositions, figure, ground, time

    def print_list(self):
        """Summary
        """
        print(
            [
                self.id,
                self.clean_user_id,
                self.prepositions,
                self.scene,
                self.figure,
                self.ground,
                self.time,
            ]
        )

    def make_preposition_list(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        x = self.prepositions.split(";")
        return x


class TypicalityAnnotation(Annotation):
    """Summary
    
    Attributes:
        list_headings (TYPE): Description
    """

    list_headings = [
        "Annotation ID",
        "Clean User ID",
        "Task",
        "Preposition",
        "Config1",
        "Config2",
        "Selection",
        "Time",
    ]

    # User selects multiple prepositions given a figure and ground
    def __init__(
            self, userdata, annotation
    ):
        """Summary
        
        Args:
            userdata (TYPE): Description
            annotation (TYPE): Description
        """
        Annotation.__init__(self, userdata, annotation)

    def print_list(self):
        """Summary
        """
        print((self.list_format))


class Data:
    """Summary
    
    Attributes:
        annotation_list (TYPE): Description
        basic_info (TYPE): Description
        clean_csv_name (str): Description
        clean_data_list (TYPE): Description
        data_list (TYPE): Description
        native_users (TYPE): Description
        scene_info (TYPE): Description
        scene_list (TYPE): Description
        study (TYPE): Description
        task (str): Description
        user_list (TYPE): Description
    
    Deleted Attributes:
        alldata (TYPE): Description
    """
    task = "all"
    clean_csv_name = "all_clean_annotations.csv"

    def __init__(self, userdata):
        """Summary
        
        Args:
            userdata (TYPE): Description
        
        Deleted Parameters:
            study (TYPE): Description
        """

        self.study = userdata.study
        self.basic_info = BasicInfo(self.study)
        self.scene_info = SceneInfo(self.study)

        self.data_list = self.load_annotations_from_csv()
        self.annotation_list = self.get_annotations(userdata)

        # Annotation list without non-natives
        self.clean_data_list = self.clean_list()

        self.user_list = self.get_users()
        self.native_users = self.get_native_users()
        # Scene list is used for checking sv and comp task
        self.scene_list = self.get_scenes()

    def load_annotations_from_csv(self):
        """Summary
        Gets list of annotations from csv.
        
        Returns:
            list: strings.
        """
        with open(self.basic_info.raw_annotation_csv, "r") as f:
            reader = csv.reader(f)
            datalist = list(reader)
        return datalist

    def get_annotations(self, userdata):
        """Summary
        Gets annotations for specific task.
        Userdata is passed to create the annotations and assign correct clean id.
        
        Args:
            userdata (TYPE): Description
        
        Returns:
            List: List of annotations
        """
        out = []
        for annotation in self.data_list:

            ann = Annotation(userdata, annotation)
            if self.task == "all":
                out.append(ann)
            elif ann.task == self.task:
                if self.task == BasicInfo.typ_task:
                    new_annotation = TypicalityAnnotation(userdata, annotation)
                if self.task in BasicInfo.semantic_abbreviations:
                    new_annotation = SemanticAnnotation(userdata, annotation)

                if self.task in BasicInfo.comparative_abbreviations:
                    new_annotation = ComparativeAnnotation(userdata, annotation)

                out.append(new_annotation)

        return out

    def clean_list(self):
        """Summary
        Creates a clean version of the annotation list.
        
        Returns:
            TYPE: Description
        """
        out = self.annotation_list[:]

        out = self.remove_non_natives(out)

        return out

    def remove_non_natives(self, list_of_annotations):
        """Summary
        
        Args:
            list_of_annotations (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        for annotation in list_of_annotations[:]:
            if annotation.user.native == "0":
                list_of_annotations.remove(annotation)
        return list_of_annotations

    def get_users(self):
        """Summary
        Creates User list of all users.
        
        Returns:
            TYPE: Description
        """
        out = []
        for an in self.annotation_list:
            if an.user not in out:
                out.append(an.user)

        return out

    def get_native_users(self):
        """Summary
        Creates User list of native users.
        
        Returns:
            TYPE: Description
        """
        out = []
        for u in self.user_list:
            if u.native == "1":
                out.append(u)
        return out

    def get_scenes(self):
        """Summary
        Gets list of scenes completed by any user (native or non-native).
        
        Returns:
            TYPE: Description
        """
        out = []

        for an in self.annotation_list:
            if hasattr(an, "scene"):
                if an.scene not in out:
                    out.append(an.scene)
        return out

    def get_scenes_done_x_times(self, x):
        """Summary
        Finds scenes which have been annotated at least list_of_annotations times by native users.
        
        Args:
            x (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        out = []
        # This only counts native speakers
        for sc in self.scene_list:
            y = self.number_of_users_per_scene(sc)
            if y >= x:
                out.append(sc)
        return out

    def print_scenes_done_x_times(self, x, task):
        """Summary
        
        Args:
            x (TYPE): Description
            task (TYPE): Description
        """
        # This only counts native speakers
        for sc in self.scene_list:
            y = self.number_of_users_per_scene(sc, task)
            if y >= x:
                print(("Scene: " + sc + " done " + str(y) + "times"))

    def print_scenes_need_doing(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        print(("Total number of scenes:" + str(len(self.scene_info.scene_name_list))))
        out = []
        # This only counts native speakers
        for sc in self.scene_list:
            x = self.number_of_users_per_scene(sc, BasicInfo.sv_task)
            y = self.number_of_users_per_scene(sc, BasicInfo.comp_task)
            z = self.number_of_users_per_scene(sc, BasicInfo.typ_task)

            if x < 3 or y < 3:
                out.append(sc)
                print(
                    (
                            "Scene: "
                            + sc
                            + " sv done "
                            + str(x)
                            + "times"
                            + " comp done "
                            + str(y)
                            + "times"
                    )
                )
        print("Number of scenes left: ")
        print((len(out)))
        return out

    def print_scenes_need_removing(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        print("To remove")
        out = []
        # This only counts native speakers
        for sc in self.scene_list:
            x = self.number_of_users_per_scene(sc, BasicInfo.sv_task)
            y = self.number_of_users_per_scene(sc, BasicInfo.comp_task)
            if x >= 3 and y >= 3:
                out.append(sc)
                print(
                    (
                            "Scene: "
                            + sc
                            + " sv done "
                            + str(x)
                            + "times"
                            + " comp done "
                            + str(y)
                            + "times"
                    )
                )
        print("Number of scenes to remove: ")
        print((len(out)))
        return out

    def number_of_scenes_done_by_user(self, user, task):
        """Summary
        
        Args:
            user (TYPE): Description
            task (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        out = []
        for annotation in self.annotation_list:
            if user.annotation_match(annotation):

                if annotation.task == task:
                    if annotation.scene not in out:
                        out.append(annotation.scene)

        return len(out)

    def number_of_users_per_scene(self, scene, task):
        """Summary
        
        Args:
            scene (TYPE): Description
            task (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        scenecounter = []
        for annotation in self.clean_data_list:

            if (
                    annotation.scene == scene
                    and annotation.clean_user_id not in scenecounter
            ):
                if annotation.task == task:
                    scenecounter.append(annotation.clean_user_id)

        return len(scenecounter)

    def number_of_completed_users(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        i = 0
        for user in self.user_list:
            x = self.number_of_scenes_done_by_user(user, BasicInfo.sv_task)
            y = self.number_of_scenes_done_by_user(user, BasicInfo.comp_task)

            if x == 10 and y == 10:
                i += 1
        return i

    def total_number_users(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        return len(self.user_list)

    def output_clean_annotation_list(self):
        """Summary
        """
        with open(
                self.basic_info.data_folder + "/" + self.clean_csv_name, "w"
        ) as csvfile:
            writer = csv.writer(csvfile)
            heading = self.clean_data_list[0].list_headings
            writer.writerow(heading)

            for annotation in self.clean_data_list:
                writer.writerow(annotation.list_format)

    # Gets user annotations for a particular task
    def get_user_task_annotations(self, user1, task):
        """Summary
        
        Args:
            user1 (TYPE): Description
            task (TYPE): Description
        
        Returns:
            TYPE: Description
        """

        out = []

        for a in self.annotation_list:

            if a.task == task:
                if user1.annotation_match(a):
                    out.append(a)
        return out

    def question_match(self, a1, a2):
        """Summary
        Compares two annotations. Returns true if the same question is being asked of the annotators.
        
        Args:
            a1 (TYPE): Description
            a2 (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        if a1.task == a2.task:
            if a1.task in BasicInfo.comparative_abbreviations:
                if (
                        a1.scene == a2.scene
                        and a1.ground == a2.ground
                        and a1.preposition == a2.preposition
                ):
                    return True
                else:
                    return False
            elif a1.task in BasicInfo.semantic_abbreviations:
                if (
                        a1.scene == a2.scene
                        and a1.ground == a2.ground
                        and a1.figure == a2.figure
                ):
                    return True
                else:
                    return False

            elif a1.task == BasicInfo.typ_task:
                if (
                        a1.c1 == a2.c1
                        and a1.c2 == a2.c2
                        and a1.preposition == a2.preposition
                ):
                    return True
                else:
                    return False

            else:
                print("Task mismatch in 'question_match()'")
                print((a1.task))
                print((a2.task))
                return False
        else:
            return False

    def write_user_agreements(self):
        """Summary
        Works out user agreements and writes to file.
        """
        with open(
                self.basic_info.stats_folder + "/" + self.agreements_csv_name, "w"
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Task: " + self.task,
                    "Number of Native English users: " + str(len(self.native_users)),
                ]
            )
            # writer.writerow(['User1','User2', 'observed_agreement','Number of Shared Annotations', 'Number of agreements', 'Expected Number of Agreements','observed_agreement(AFF)','Number of Shared Annotations(AFF)', 'NUmber of agreements(AFF)', 'Expected Agreement(AFF)'])
            number_of_comparisons = 0
            total_shared_annotations = 0
            total_expected_agreement_sum = 0
            total_observed_agreement_sum = 0
            total_cohens_kappa_sum = 0

            writer.writerow(
                [
                    "Preposition",
                    "Number of Shared Annotations",
                    "Average Expected agreement",
                    "Average observed Agreement",
                    "Average cohens_kappa",
                ]
            )

            for p in BasicInfo.comparative_preposition_list:
                p_number_of_comparisons = 0
                preposition_shared_annotations = 0
                preposition_expected_agreement_sum = 0
                preposition_observed_agreement_sum = 0
                preposition_cohens_kappa_sum = 0

                user_pairs = list(itertools.combinations(self.native_users, 2))
                for user_pair in user_pairs:
                    user1 = user_pair[0]
                    user2 = user_pair[1]
                    # for user1 in self.native_users:
                    #   for user2 in self.native_users:
                    if user1 != user2:
                        # Calculate agreements for user pair and add values to totals

                        x = Agreements(self.study, self.annotation_list, self.task, p, user1, user2)

                        if x.shared_annotations != 0:
                            number_of_comparisons += 1
                            p_number_of_comparisons += 1

                            preposition_shared_annotations += x.shared_annotations
                            preposition_expected_agreement_sum += (
                                    x.expected_agreement * x.shared_annotations
                            )
                            preposition_observed_agreement_sum += (
                                    x.observed_agreement * x.shared_annotations
                            )
                            preposition_cohens_kappa_sum += (
                                    x.cohens_kappa * x.shared_annotations
                            )

                            total_shared_annotations += x.shared_annotations
                            total_expected_agreement_sum += (
                                    x.expected_agreement * x.shared_annotations
                            )
                            total_observed_agreement_sum += (
                                    x.observed_agreement * x.shared_annotations
                            )
                            total_cohens_kappa_sum += (
                                    x.cohens_kappa * x.shared_annotations
                            )

                if preposition_shared_annotations != 0:
                    p_expected_agreement = float(preposition_expected_agreement_sum) / (
                        preposition_shared_annotations
                    )
                    p_observed_agreement = float(preposition_observed_agreement_sum) / (
                        preposition_shared_annotations
                    )
                    p_cohens_kappa = float(preposition_cohens_kappa_sum) / (
                        preposition_shared_annotations
                    )
                else:
                    p_expected_agreement = 0
                    p_observed_agreement = 0
                    p_cohens_kappa = 0

                # Write a row for each preposition

                row = [
                    p,
                    preposition_shared_annotations,
                    p_expected_agreement,
                    p_observed_agreement,
                    p_cohens_kappa,
                ]
                writer.writerow(row)

            if total_shared_annotations != 0:
                total_expected_agreement = float(total_expected_agreement_sum) / (
                    total_shared_annotations
                )
                total_observed_agreement = float(total_observed_agreement_sum) / (
                    total_shared_annotations
                )
                total_cohens_kappa = float(total_cohens_kappa_sum) / (
                    total_shared_annotations
                )
            else:
                total_expected_agreement = 0
                total_observed_agreement = 0
                total_cohens_kappa = 0

            # Write a row of total averages
            writer.writerow(
                [
                    "Total Number of Shared Annotations",
                    "Average Expected Agreements",
                    "Average observed agreements",
                    "Average Cohens Kappa",
                ]
            )
            row = [
                total_shared_annotations,
                total_expected_agreement,
                total_observed_agreement,
                total_cohens_kappa,
            ]
            writer.writerow(row)


class ComparativeData(Data):
    """Summary
    Stores and handles data from 'comp' task.
    
    Attributes:
        agreements_csv_name (str): Description
        clean_csv_name (TYPE): Description
        preposition_list (TYPE): Description
        stats_csv_name (str): Description
        task (TYPE): Description
    
    Deleted Attributes:
        annotation_list (TYPE): Description
        basic_info (TYPE): Description
        clean_data_list (TYPE): Description
        data_list (TYPE): Description
        native_users (TYPE): Description
        scene_list (TYPE): Description
        study (TYPE): Description
        user_list (TYPE): Description
    """
    task = BasicInfo.comp_task
    clean_csv_name = BasicInfo.comp_annotations_name
    stats_csv_name = "comparative stats.csv"
    agreements_csv_name = "comparative agreements.csv"

    def __init__(self, userdata):
        """Summary
        
        Args:
            userdata (TYPE): Description
        
        Deleted Parameters:
            study (TYPE): Description
        """

        Data.__init__(self, userdata)

        self.preposition_list = self.get_prepositions()

    def get_prepositions(self):
        """Summary
        Get list of prepositions used in comp task.
        
        Returns:
            TYPE: Description
        """
        out = []
        for annotation in self.clean_data_list:
            if annotation.preposition not in out:
                out.append(annotation.preposition)
        return out

    def get_preposition_info_for_scene(self, scene):
        """Summary
        
        Args:
            scene (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        out = []

        for p in self.preposition_list:
            grounds = []
            for annotation in self.clean_data_list:
                if annotation.scene == scene and annotation.preposition == p:
                    g = annotation.ground
                    if g not in grounds:
                        grounds.append(g)

            for grd in grounds:
                c = Comparison(scene, p, grd, self.study)
                instances = c.get_instances(self.clean_data_list)
                figure_selection_number = c.get_choices(self.clean_data_list)

                i = len(instances)
                row = []
                crow = [p, grd, i]
                row.append(crow)
                for f in figure_selection_number:
                    x = [f, figure_selection_number[f]]
                    row.append(x)
                out.append(row)

        return out

    # This is a very basic list of information about the task
    # compile_instances gives a better overview
    def output_statistics(self):
        """Summary
        """
        with open(self.basic_info.stats_folder + "/" + self.stats_csv_name, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["Number of Native English users: " + str(len(self.native_users))]
            )
            writer.writerow(["Scene", "Number of Users Annotating", "Selection Info"])

            for s in self.scene_list:
                row = [
                    s,
                    self.number_of_users_per_scene(s, self.task),
                    self.get_preposition_info_for_scene(s),
                ]
                # for p in self.get_prepositions_for_scene(s):
                #   row.append(p)
                writer.writerow(row)


class SemanticData(Data):
    """Summary
    
    Attributes:
        agreements_csv_name (str): Description
        categorisation_stats_csv (str): Description
        clean_csv_name (TYPE): Description
        stats_csv_name (str): Description
        task (TYPE): Description
    
    Deleted Attributes:
        annotation_list (TYPE): Description
        basic_info (TYPE): Description
        clean_data_list (TYPE): Description
        data_list (TYPE): Description
        native_users (TYPE): Description
        scene_list (TYPE): Description
        study (TYPE): Description
        user_list (TYPE): Description
    """
    task = BasicInfo.sv_task
    clean_csv_name = BasicInfo.sem_annotations_name
    stats_csv_name = "semantic stats.csv"
    categorisation_stats_csv = "categorisation stats.csv"
    agreements_csv_name = "semantic agreements.csv"

    def __init__(self, userdata):
        """Summary
        
        Args:
            userdata (TYPE): Description
        
        Deleted Parameters:
            study (TYPE): Description
        """

        Data.__init__(self, userdata)

    def get_prepositions_for_scene(self, scene):
        """Summary
        
        Args:
            scene (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        out = []
        for annotation in self.clean_data_list:
            if annotation.scene == scene:
                for p in annotation.preposition_list:
                    if p not in out:
                        out.append(p)

        return out

    def pair_prepositions_scenes(self):
        """Summary
        """
        for scene in self.scene_list:
            if scene in self.get_scenes_done_x_times(1):
                print(("Scene: " + scene))
                ps = self.get_prepositions_for_scene(scene)
                for p in ps:
                    print(p)

    # Identifies number of times prepositions are selected or left blank
    def get_positive_selection_info(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        positive_selections = 0
        negative_selections = 0

        for p in BasicInfo.semantic_preposition_list:
            for a in self.clean_data_list:
                if p in a.prepositions:
                    positive_selections += 1
                elif p not in a.prepositions:
                    negative_selections += 1
        return [positive_selections, negative_selections]

    def check_categorisation_difference(self, preposition, c1, c2):
        """Calculates whether c1 is significantly better category member than c2 and vice versa.
        
        Parameters:
            preposition (TYPE): Description
            c1 (TYPE): Description
            c2 (TYPE): Description
            preposition -- preposition
            c1 --  configuration to compare
            c2 -- configuration to compare
        
        Returns:
            TYPE: Description
        """

        c1_times_labelled = float(
            c1.number_of_selections_from_annotationlist(
                preposition, self.clean_data_list
            )
        )
        c1_times_not_labelled = (
                float(c1.number_of_tests(self.clean_data_list)) - c1_times_labelled
        )

        c2_times_labelled = float(
            c2.number_of_selections_from_annotationlist(
                preposition, self.clean_data_list
            )
        )
        c2_times_not_labelled = (
                float(c2.number_of_tests(self.clean_data_list)) - c2_times_labelled
        )

        # I think this needs flipping. Also set alternative parameter
        # Also need to check if c1>c2 or c2>c1 seperately
        oddsratio, p_value = scipy.stats.fisher_exact(
            [c1_times_labelled, c2_times_labelled],
            [c1_times_not_labelled, c2_times_not_labelled],
        )

        return [
            c1_times_labelled,
            c1_times_not_labelled,
            c2_times_labelled,
            c2_times_not_labelled,
            p_value,
        ]

    def output_categorisation_check(self):
        """Summary
        """
        # Change how relations.csv is found and loaded so we can look at multiple studies.
        config_list = Relationship.load_all()
        for preposition in preposition_list:
            with open(
                    self.basic_info.stats_folder
                    + "/"
                    + preposition
                    + "/"
                    + self.categorisation_stats_csv,
                    "w",
            ) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "Scene1",
                        "Figure1",
                        "Ground1",
                        "Scene2",
                        "Figure2",
                        "Ground2",
                        "c1_times_labelled",
                        "c1_times_not_labelled",
                        "c2_times_labelled",
                        "c2_times_not_labelled",
                        "p_value",
                    ]
                )
                for row in config_list:
                    c1 = Configuration(row[0], row[1], row[2])
                    for row2 in config_list:
                        c2 = Configuration(row2[0], row2[1], row2[2])
                        stat = self.check_categorisation_difference(preposition, c1, c2)
                        to_write = (
                                [c1.scene, c1.figure, c1.ground]
                                + [c2.scene, c2.figure, c2.ground]
                                + stat
                        )
                        writer.writerow(to_write)

    # This is a very basic list of information about the task
    # compile_instances gives a better overview
    def output_statistics(self):
        """Summary
        """
        with open(
                self.basic_info.stats_folder + "/" + self.stats_csv_name, "w"
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["Number of Native English users: " + str(len(self.native_users))]
            )

            writer.writerow(["Positive Selections", "negative_selections"])
            row = self.get_positive_selection_info()
            writer.writerow(row)
            writer.writerow(
                ["Scene", "Number of Users Annotating", "Given Prepositions"]
            )

            for s in self.scene_list:
                writer.writerow(
                    [
                        s,
                        self.number_of_users_per_scene(s, self.task),
                        self.get_prepositions_for_scene(s),
                    ]
                )


class ModSemanticData(SemanticData):
    """Summary
    
    Attributes:
        agreements_csv_name (str): Description
        categorisation_stats_csv (str): Description
        clean_csv_name (TYPE): Description
        stats_csv_name (str): Description
        task (TYPE): Description
    
    Deleted Attributes:
        annotation_list (TYPE): Description
        basic_info (TYPE): Description
        clean_data_list (TYPE): Description
        data_list (TYPE): Description
        native_users (TYPE): Description
        scene_list (TYPE): Description
        study (TYPE): Description
        user_list (TYPE): Description
    """
    task = BasicInfo.svmod_task
    clean_csv_name = BasicInfo.svmod_annotations_name
    stats_csv_name = "svmod stats.csv"
    categorisation_stats_csv = "categorisation stats.csv"
    agreements_csv_name = "svmod agreements.csv"

    def __init__(self, userdata, study):
        """Summary
        
        Args:
            userdata (TYPE): Description
            study (TYPE): Description
        """

        Data.__init__(self, userdata)


class TypicalityData(Data):
    """Summary
    
    Attributes:
        agreements_csv_name (str): Description
        annotation_list (TYPE): Description
        basic_info (TYPE): Description
        clean_csv_name (TYPE): Description
        clean_data_list (TYPE): Description
        data_list (TYPE): Description
        native_users (TYPE): Description
        stats_csv_name (str): Description
        study (TYPE): Description
        task (TYPE): Description
        user_list (TYPE): Description
    """

    def __init__(self, userdata, study):
        """Summary
        
        Args:
            userdata (TYPE): Description
            study (TYPE): Description
        """
        self.task = BasicInfo.typ_task
        self.clean_csv_name = BasicInfo.typ_annotations_name
        self.stats_csv_name = "typicality stats.csv"
        self.agreements_csv_name = "typicality agreements.csv"
        self.study = study
        self.basic_info = BasicInfo(study)

        self.data_list = self.load_annotations_from_csv()
        self.annotation_list = self.get_annotations(userdata)
        self.clean_data_list = self.clean_list()
        self.user_list = self.get_users()
        self.native_users = self.get_native_users()

    # This is a very basic list of information about the task
    # compile_instances gives a better overview
    def output_statistics(self):
        """Summary
        """
        with open(
                self.basic_info.stats_folder + "/" + self.stats_csv_name, "w"
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["Number of Native English users: " + str(len(self.native_users))]
            )


class Agreements(Data):
    """Summary
    
    Attributes:
        annotation_list (TYPE): Description
        cohens_kappa (TYPE): Description
        expected_agreement (TYPE): Description
        observed_agreement (TYPE): Description
        preposition (TYPE): Description
        shared_annotations (TYPE): Description
        study (TYPE): Description
        task (TYPE): Description
        user1 (TYPE): Description
        user1_annotations (TYPE): Description
        user2 (TYPE): Description
        user2_annotations (TYPE): Description
    """

    # Looks at agreements between two users for a particular task and particular preposition
    def __init__(
            self,
            study,
            annotation_list,
            task,
            preposition,
            user1,
            user2=None,
            agent_task_annotations=None,
    ):
        """Summary
        
        Args:
            study (TYPE): Description
            annotation_list (TYPE): Description
            task (TYPE): Description
            preposition (TYPE): Description
            user1 (TYPE): Description
            user2 (None, optional): Description
            agent_task_annotations (None, optional): Description
        """
        self.annotation_list = annotation_list
        self.user1 = user1
        self.user2 = user2
        self.study = study
        self.task = task
        # All user annotations for particular task
        self.user1_annotations = self.get_user_task_annotations(user1, task)
        if user2 != None:
            self.user2_annotations = self.get_user_task_annotations(user2, task)
        else:
            # We can test agent models instead of users
            self.user2_annotations = agent_task_annotations

        self.preposition = preposition

        if self.task in BasicInfo.semantic_abbreviations and self.preposition == "none":
            print("Error: Checking 'none' agreement")
        self.user_calculations()

    # Agreement of users

    def count_sem_agreements(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        # Number of shared annotations by u1 and u2
        shared_annotations = 0
        # Times u1 says yes to preposition
        y1 = 0
        # Times u2 says yes to preposition
        y2 = 0
        # Times u1 says no to preposition
        n1 = 0
        # Times u2 says no to preposition
        n2 = 0
        agreements = 0
        for a1 in self.user1_annotations:
            for a2 in self.user2_annotations:
                if self.question_match(a1, a2):
                    if a1.task in BasicInfo.semantic_abbreviations:
                        shared_annotations += 1
                        if self.preposition in a1.prepositions:
                            y1 += 1
                        else:
                            n1 += 1
                        if self.preposition in a2.prepositions:
                            y2 += 1
                        else:
                            n2 += 1

                        if (
                                self.preposition in a1.prepositions
                                and self.preposition in a2.prepositions
                        ):
                            agreements += 1

                        elif (
                                self.preposition not in a1.prepositions
                                and self.preposition not in a2.prepositions
                        ):
                            agreements += 1
        return shared_annotations, y1, y2, n1, n2, agreements

    def count_comp_agreements(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        # Number of shared annotations by u1 and u2
        shared_annotations = 0
        # Number of times none selected by u1 in comp task
        comp_none_selections1 = 0
        # Number of times none selected by u2 in comp task
        comp_none_selections2 = 0

        number_of_compared_figures = 0
        # expected_agreement_with_none = 0
        agreements = 0
        for a1 in self.user1_annotations:
            for a2 in self.user2_annotations:
                if self.question_match(a1, a2):

                    if a1.task in BasicInfo.comparative_abbreviations:
                        if a1.preposition == self.preposition:
                            shared_annotations += 1
                            if a1.figure == "none":
                                comp_none_selections1 += 1
                            if a2.figure == "none":
                                comp_none_selections2 += 1

                            c = Comparison(
                                a1.scene, a1.preposition, a1.ground, self.study
                            )
                            no_possible_selections = len(c.possible_figures)
                            number_of_compared_figures += no_possible_selections

                            if a1.figure == a2.figure:
                                agreements += 1
        return (
            shared_annotations,
            comp_none_selections1,
            comp_none_selections2,
            number_of_compared_figures,
            agreements,
        )

    def count_typ_agreements(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        # Number of shared annotations by u1 and u2
        shared_annotations = 0
        # Number of times none selected by u1 in typ task
        typ_none_selections1 = 0
        # Number of times none selected by u2 in typ task
        typ_none_selections2 = 0

        agreements = 0
        for a1 in self.user1_annotations:
            for a2 in self.user2_annotations:
                if self.question_match(a1, a2):

                    if a1.task == BasicInfo.typ_task:
                        if a1.preposition == self.preposition:
                            shared_annotations += 1
                            if a1.selection == "none":
                                typ_none_selections1 += 1
                            if a2.selection == "none":
                                typ_none_selections2 += 1

                            if a1.selection == a2.selection:
                                agreements += 1
        return (
            shared_annotations,
            typ_none_selections1,
            typ_none_selections2,
            agreements,
        )

    def calculate_sem_expected_agreement(self, shared_annotations, y1, y2, n1, n2):
        """Calculate expected agreement for semantic task between two users.
        
        Parameters:
            shared_annotations (TYPE): Description
            y1 (TYPE): Description
            y2 (TYPE): Description
            n1 (TYPE): Description
            n2 (TYPE): Description
            shared_annotations -- Number of shared annotations
            y1 --  Times u1 says yes to preposition
            y2 -- Times u2 says yes to preposition
            n1 -- Times u1 says no to preposition
            n2 -- Times u2 says no to preposition
        
        Returns:
            TYPE: Description
        """

        if shared_annotations != 0:

            expected_agreement = float((y1 * y2 + n1 * n2)) / float(
                (shared_annotations) ** 2
            )
        else:
            expected_agreement = 0
        return expected_agreement

    def calculate_comp_expected_agreement(
            self,
            shared_annotations,
            comp_none_selections1,
            comp_none_selections2,
            number_of_compared_figures,
    ):
        """Summary
        
        Args:
            shared_annotations (TYPE): Description
            comp_none_selections1 (TYPE): Description
            comp_none_selections2 (TYPE): Description
            number_of_compared_figures (TYPE): Description
        """
        if self.task in BasicInfo.comparative_abbreviations:
            if shared_annotations != 0:
                u1_p_none = float(comp_none_selections1) / shared_annotations
                u2_p_none = float(comp_none_selections2) / shared_annotations

                expected_none_agreement = float(u1_p_none * u2_p_none)

                # As there are a different number of distractors in each scene and the distractors change
                # We make an approximation here and work out there overall chance of agreeing on an object

                average_probability_agree_on_object = (
                        float(shared_annotations * (1 - u1_p_none) * (1 - u2_p_none))
                        / number_of_compared_figures
                )

                expected_agreement = (
                        expected_none_agreement + average_probability_agree_on_object
                )
            else:
                expected_agreement = 0
        return expected_agreement

    def calculate_typ_expected_agreement(
            self, shared_annotations, typ_none_selections1, typ_none_selections2
    ):
        """Summary
        
        Args:
            shared_annotations (TYPE): Description
            typ_none_selections1 (TYPE): Description
            typ_none_selections2 (TYPE): Description
        """
        if self.task == BasicInfo.typ_task:
            if shared_annotations != 0:
                u1_p_none = float(typ_none_selections1) / shared_annotations
                u2_p_none = float(typ_none_selections2) / shared_annotations

                expected_none_agreement = float(u1_p_none * u2_p_none)

                average_probability_agree_on_object = (
                        float((1 - u1_p_none) * (1 - u2_p_none)) / 2
                )

                expected_agreement = (
                        expected_none_agreement + average_probability_agree_on_object
                )
            else:
                expected_agreement = 0
        return expected_agreement

    def user_calculations(self):
        """Summary
        """
        observed_agreement = 0
        cohens_kappa = 0

        if self.task in BasicInfo.semantic_abbreviations:
            shared_annotations, y1, y2, n1, n2, agreements = self.count_sem_agreements()
            expected_agreement = self.calculate_sem_expected_agreement(
                shared_annotations, y1, y2, n1, n2
            )
        elif self.task in BasicInfo.comparative_abbreviations:
            (
                shared_annotations,
                comp_none_selections1,
                comp_none_selections2,
                number_of_compared_figures,
                agreements,
            ) = self.count_comp_agreements()
            expected_agreement = self.calculate_comp_expected_agreement(
                shared_annotations,
                comp_none_selections1,
                comp_none_selections2,
                number_of_compared_figures,
            )
        elif self.task == BasicInfo.typ_task:

            (
                shared_annotations,
                typ_none_selections1,
                typ_none_selections2,
                agreements,
            ) = self.count_typ_agreements()
            expected_agreement = self.calculate_typ_expected_agreement(
                shared_annotations, typ_none_selections1, typ_none_selections2
            )

        if shared_annotations != 0:
            observed_agreement = float(agreements) / float(shared_annotations)

        if observed_agreement != 1:
            cohens_kappa = float(observed_agreement - expected_agreement) / float(
                1 - expected_agreement
            )
        else:
            cohens_kappa = 1

        self.shared_annotations = shared_annotations
        self.expected_agreement = expected_agreement
        self.observed_agreement = observed_agreement
        self.cohens_kappa = cohens_kappa


if __name__ == "__main__":
    # Begin by loading users
    userdata2019 = UserData("2019 study")
    # userdata2020 = UserData("2020 study")
    # Output user list
    userdata2019.output_clean_user_list()
    # userdata2020.output_clean_user_list()

    # Load all csv
    alldata_2019 = Data(userdata2019)
    # alldata_2020 = Data(userdata2020, "2020 study")

    alldata_2019.output_clean_annotation_list()
    # d.print_scenes_need_doing()
    # d.print_non_users()
    # d.output_clean_annotation_list()

    #
    # Load and process semantic annotations
    semantic_data = SemanticData(userdata2019)

    # Output semantic csv
    semantic_data.output_clean_annotation_list()

    semantic_data.output_statistics()

    semantic_data.write_user_agreements()

    # #Load and process comparative annotations
    comparative_data = ComparativeData(userdata2019)

    # # output comparative csv

    comparative_data.output_clean_annotation_list()

    comparative_data.output_statistics()

    comparative_data.write_user_agreements()

    ## typicality data
    # typ_data = TypicalityData(userdata2020)

    # # output typicality csv

    # typ_data.output_clean_annotation_list()

    # typ_data.output_statistics()

    # typ_data.write_user_agreements()

    # # Load and process semantic annotations
    # svmod_data = ModSemanticData(userdata2020)

    # # Output semantic csv
    # svmod_data.output_clean_annotation_list()

    # svmod_data.output_statistics()

    # svmod_data.write_user_agreements()

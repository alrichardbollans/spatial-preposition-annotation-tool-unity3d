"""Summary
## Run process_data and preprocess_features before this

## Input: cleaned annotation, user lists and list of feature values for configurations (see Relationship class)
## Compiles annotation instances, adds feature values to them
## Output: For each task: Basic stats for each preposition.
# For sv task writes a csv of feature values with selection information

## Also has functions to generate constraints from comparative data

Deleted Attributes:
    comp_annotations_name (TYPE): Description
    data_folder_name (TYPE): Description
    feature_data_folder_name (TYPE): Description
    project_folder_name (TYPE): Description
    sem_annotations_name (TYPE): Description
     (TYPE): Description
"""

import csv
import numpy as np

import preprocess_features
from classes import Instance, Configuration, SceneInfo, CompInstance, Constraint, Comparison, Relationship, BasicInfo
from process_data import SemanticAnnotation, ComparativeAnnotation


class Preposition:
    """Summary
    
    Attributes:
        array (TYPE): Description
        config_list (TYPE): Description
        instance_list (TYPE): Description
        name (TYPE): Description
        study (TYPE): Description
        value_list (list): Description
    """

    # Preposition is instantiated for a particular list of instances
    # each preposition ends up with an associated array
    def __init__(self, name, instance_list, study):
        """Summary
        
        Args:
            name (TYPE): Description
            instance_list (TYPE): Description
            study (TYPE): Description
        """
        self.name = name
        self.study = study
        self.value_list = []
        self.instance_list = self.get_instances(instance_list)
        self.config_list = self.get_configs()

    def get_instances(self, instance_list):
        """Summary
        
        Args:
            instance_list (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        instances = []
        for i in instance_list:
            if i.preposition == self.name or self.name == 'all':
                if i.figure != "none":
                    instances.append(i)
        return instances

    ## Get configurations and appends value rows to value list
    def get_configs(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        configs = []
        for i in self.instance_list:
            c = Configuration(i.scene, i.figure, i.ground, self.study)

            self.value_list.append(c.row)
            ## Collect distinct configurations
            if not any(i.scene == c.scene and i.figure == c.figure and i.ground == c.ground for c in configs):
                configs.append(c)
        return configs

    # def append_values(self,instance_list):
    # 	for i in instance_list:
    # 		if i.preposition == self.name or self.name == 'all':
    # 			if hasattr(i, 'containment'):
    # 				self.value_list.append([i.support_top,i.containment,i.contact,i.contact_scaled,i.above_measure,i.ground_verticalness,i.ground_CN_ISA_CONTAINER,i.support_cobb,i.support_bottom,i.ground_CN_UsedFor_Light])

    def create_array(self):
        """Summary
        """
        self.array = np.array(self.value_list)

    def average_value(self, relation):
        """Summary
        
        Args:
            relation (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        ### Note this uses all selected instances rather than all distinct instances e.g. if a config is selected more it will skew value - which may be desirable depending on use
        counter = 0
        values = 0

        for i in self.instance_list:
            # print(i.scene)
            values += getattr(i, relation)
            counter += 1
        if counter == 0:
            return 0
        else:
            average = values / counter
            return average


class Collection:
    """Summary
    Generic collection class
    Does not require annotations/instances
    
    Attributes:
        annotation_list (list): Description
        basic_info (TYPE): Description
        data_folder_name (TYPE): Description
        feature_data_csv (TYPE): Description
        feature_keys (TYPE): Description
        instance_list (list): Description
        relation_keys (TYPE): Description
        stats_folder (TYPE): Description
        study (TYPE): Description
    
    Deleted Attributes:
        preposition_list (list): Description
    """

    def __init__(self, study):
        """Summary
        
        Args:
            study (TYPE): Description
        """
        self.basic_info = BasicInfo(study)
        self.study = study
        self.relation_keys = Relationship.get_relation_keys(study)
        self.feature_keys = Relationship.get_feature_keys(study)

        self.feature_data_csv = self.basic_info.feature_output_csv
        self.data_folder_name = self.basic_info.data_folder
        self.stats_folder = self.basic_info.stats_folder

        ## Annotation list is raw list of annotations from csv
        ## Useful for counting number of tests of particular configuration
        self.annotation_list = []
        ## Instance list is processed annotation list into list of separate instance objects
        self.instance_list = []

    def append_values(self):
        """Summary
        """
        for i in self.instance_list:
            if i.figure != "none":
                try:

                    r = Relationship(i.scene, i.figure, i.ground, self.study)
                    r.load_from_csv()

                    for key in r.feature_keys:
                        setattr(i, key, r.set_of_features[key])



                except Exception as e:
                    print('Instance not added')
                    print(('Figure: ' + i.figure))
                    print(('Ground: ' + i.ground))

                    print(e)

    def get_relation_values(self, relation):
        """Summary
        Gets values of a particular feature
        
        Args:
            relation (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        set1 = []
        for i in self.instance_list:
            set1.append(getattr(i, relation))
        return set1

    def get_mean_relation_value(self, relation):
        """Summary
        Gets mean value of a particular feature
        
        Args:
            relation (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        x = np.mean(self.get_relation_values(relation))
        return x

    def get_std_relation_value(self, relation):
        """Summary
        
        Args:
            relation (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        x = np.std(self.get_relation_values(relation))
        return x


class InstanceCollection(Collection):
    """Summary
    
    Attributes:
        categorisation_feature_name (str): Description
        fig_feature_name (str): Description
        ground_feature_name (str): Description
        ratio_feature_name (str): Description
        scene_feature_name (str): Description
    """

    ## Store some feature names for writing datasets
    ratio_feature_name = "selection_ratio"
    categorisation_feature_name = "selected_atleast_once"
    scene_feature_name = 'Scene'
    fig_feature_name = 'Figure'
    ground_feature_name = 'Ground'

    def __init__(self, study):
        """Summary
        
        Args:
            study (TYPE): Description
        """
        Collection.__init__(self, study)

    ### Instance Collection contains instances with a preposition
    def get_used_prepositions(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        out = []
        for i in self.instance_list:
            if i.preposition not in BasicInfo.preposition_list:
                BasicInfo.preposition_list.append(i.preposition)
            if i.preposition not in out:
                out.append(i.preposition)
        return out

    ## Write csvs for each preposition giving feature values and data on number of selections
    ## Also tags as test/training instance
    def write_config_ratios(self):
        """Summary
        """
        scene_list = []
        s = SceneInfo(self.study)
        for scene in s.scene_list:
            scene_list.append(scene.name)

        for preposition in self.get_used_prepositions():
            config_list = Relationship.load_all(self.study)
            config_list.pop(0)

            ## Write file of all instances
            with open(self.basic_info.config_ratio_csv(self.filetag, preposition), "w") as csvfile:
                outputwriter = csv.writer(csvfile)
                outputwriter.writerow(['Scene', 'Figure', 'Ground'] + self.feature_keys + [self.ratio_feature_name,
                                                                                           self.categorisation_feature_name])

                for row in config_list:

                    c = Configuration(row[0], row[1], row[2], self.study)
                    t = float(c.number_of_tests(self.annotation_list))
                    s = float(c.number_of_selections(preposition, self.instance_list))

                    ## If at least one test has been done for this configuration
                    if t != 0:
                        ratio = s / t

                        r = str(ratio)

                        row.append(r)

                        if (ratio == 0):
                            row.append(str(0))
                        else:
                            row.append(str(1))

                        # if c.scene in train_scenes:
                        # 	row.append("train")
                        # elif c.scene in test_scenes:
                        # 	row.append("test")
                        outputwriter.writerow(row)

    #### Write General Stats for each preposition
    def write_preposition_stats_csvs(self):
        """Summary
        """
        config_list = Relationship.load_all(self.study)

        # for preposition in self.get_used_prepositions():

        # 	## Write file of all instances
        # 	with open('preposition data/'+self.filetag+ '-' + preposition + ' data.csv', "w") as csvfile:
        # 		outputwriter1 = csv.writer(csvfile)
        # 		outputwriter1.writerow(['Scene','Figure','Ground']+self.feature_keys)

        # 		for i in self.instance_list:
        # 			if i.preposition == preposition:
        # 				for row in config_list:
        # 					if i.config_row_match(row):
        # 						outputwriter1.writerow(row)

        ### Write file summarizing stats
        with open(self.basic_info.stats_folder + '/' + self.filetag + ' preposition stats.csv', "w") as csvfile:
            outputwriter = csv.writer(csvfile)
            outputwriter.writerow(['', '', '', 'Average Values'])
            outputwriter.writerow(
                ['Preposition', 'Number of Selections', 'Number of Distinct Configurations'] + self.feature_keys)

        for preposition in self.get_used_prepositions() + ['all']:

            row = [preposition]

            p = Preposition(preposition, self.instance_list, self.study)

            row.append(len(p.instance_list))
            row.append(len(p.config_list))

            for at in self.feature_keys:
                # print(at)
                value = p.average_value(at)
                # print('Property: ')
                # print(at)
                # print('average_value: ')
                # print(value)
                row.append(value)

            with open(self.basic_info.stats_folder + '/' + self.filetag + ' preposition stats.csv',
                      "a") as csvfile:
                outputwriter = csv.writer(csvfile)

                outputwriter.writerow(row)


class SemanticCollection(InstanceCollection):
    """Summary
    
    Attributes:
        filetag (str): Description
    
    Deleted Attributes:
        preposition_list (list): Description
    """

    filetag = 'semantic'

    def __init__(self, study):
        """Summary
        
        Args:
            study (TYPE): Description
        """
        InstanceCollection.__init__(self, study)
        self.append_annotations()

    def append_annotations(self):
        """Summary
        """
        ### Reads annotations from clean files
        ### Must be updated if process_data prints csvs differently
        filename = self.basic_info.data_folder + "/" + self.basic_info.sem_annotations_name

        with open(filename, "r") as f:
            reader = csv.reader(f)  # create a 'csv reader' from the file object
            annotationlist = list(reader)  # create a list from the reader

        annotationlist.pop(0)  # removes first line of data list which is headings

        for clean_annotation in annotationlist:
            self.annotation_list.append(clean_annotation)

            an_id, clean_user_id, task, scene, prepositions, figure, ground, time = SemanticAnnotation.retrieve_from_data_row(
                clean_annotation)

            given_prepositions = prepositions.split(";")

            for p in given_prepositions:
                if p != "":
                    i = Instance(an_id, clean_user_id, task, scene, p, figure, ground, self.study)

                    self.instance_list.append(i)

        self.append_values()


class ComparativeCollection(InstanceCollection):
    """Summary
    
    Attributes:
        constraints (TYPE): Description
        filetag (str): Description
    """
    filetag = 'comparative'

    def __init__(self, study):
        """Summary
        
        Args:
            study (TYPE): Description
        """
        InstanceCollection.__init__(self, study)
        self.append_annotations()

    # self.constraints = self.get_constraints()

    ### Reads annotation file and appends to annotation and instance lists.
    def append_annotations(self):
        """Summary
        """
        ### Reads annotations from clean files
        ### Must be updated if process_data prints csvs differently
        filename = self.basic_info.data_folder + "/" + self.basic_info.comp_annotations_name

        with open(filename, "r") as f:
            reader = csv.reader(f)  # create a 'csv reader' from the file object
            annotationlist = list(reader)  # create a list from the reader

        annotationlist.pop(0)  # removes first line of data list which is headings

        for clean_annotation in annotationlist:
            self.annotation_list.append(clean_annotation)

            an_id, clean_user_id, task, scene, preposition, figure, ground, time, possible_figures = ComparativeAnnotation.retrieve_from_data_row(
                clean_annotation)

            i = CompInstance(an_id, clean_user_id, task, scene, preposition, figure, ground, possible_figures,
                             self.study)

            self.instance_list.append(i)

        self.append_values()

    def get_constraints(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        # First clear written constraints
        Constraint.clear_csv(self.basic_info.constraint_csv)
        ## Creates a dictionary, prepositions are keys
        ### Values are lists of constraints for the preposition
        out = dict()
        s_info = SceneInfo(self.study)

        for preposition in BasicInfo.preposition_list:
            # print(preposition)
            C = []
            for s in s_info.scene_list:

                grounds = s.selectable_objects

                for grd in grounds:

                    c = Comparison(s.name, preposition, grd, self.study)
                    Cons = c.generate_constraints(self.instance_list)
                    for con in Cons:
                        C.append(con)
            out[preposition] = C
            # print(C)
            for con in C:
                con.write_to_csv(self.basic_info.constraint_csv)
        self.constraints = out
        return out


class ConfigurationCollection(Collection):
    """Summary
    
    Attributes:
        filetag (str): Description
        instance_list (list): Description
    """
    filetag = 'configs'

    def __init__(self, study):
        """Summary
        
        Args:
            study (TYPE): Description
        """
        Collection.__init__(self, study)

        ## List of configuration instances
        self.instance_list = []
        self.append_configurations()

    def append_configurations(self):
        """Summary
        """
        s = SceneInfo(self.study)
        for scene in s.scene_list:

            for c in scene.get_all_configs():
                if c[0] != c[1]:
                    config = Configuration(scene.name, c[0], c[1], self.study)

                    self.instance_list.append(config)

    def write_data_csv(self):
        """Summary
        """
        ## Writes a new csv giving all configs and data
        # Not sure this is necessary

        titles = ['Scene', 'Figure', 'Ground'] + self.feature_keys

        with open(self.basic_info.stats_folder + '/' + 'configuration data.names', "w") as csvfile:
            outputwriter = csv.writer(csvfile)

            outputwriter.writerow(titles)

        with open(self.basic_info.stats_folder + '/' + 'configuration data.csv', "w") as csvfile:
            outputwriter = csv.writer(csvfile)

            for c in self.instance_list:
                outputwriter.writerow(c.full_row)

    def write_preposition_data_csvs(self, preposition, datalist):
        """Summary
        
        Args:
            preposition (TYPE): Description
            datalist (TYPE): Description
        """
        conf = self.instance_list[1]

        with open('labelled config data/selection data.names', "w") as csvfile:
            outputwriter = csv.writer(csvfile)

            outputwriter.writerow(
                ['scene', 'figure', 'ground'] + conf.value_names + ['Selected?', 'Number of Selections'])
        ### should edit this to only use configurations that appear in the 'all' preposition (as I mention in the write up)
        with open('configuration data.csv', "r") as readfile:
            reader = csv.reader(readfile)  # create a 'csv reader' from the file object
            config_list = list(reader)  # create a list from the reader
            with open('labelled config data/selection data-' + preposition + '.csv', "w") as csvfile:
                outputwriter = csv.writer(csvfile)
                for row in config_list:
                    for c in self.instance_list:
                        if c.config_row_match(row):
                            x = c.number_of_selections(preposition, datalist)
                            if x > 0:
                                row.append(1)
                            else:
                                row.append(0)
                            row.append(x)
                            outputwriter.writerow(row)

    def greater_than_configs(self, relation, value):
        """Summary
        
        Args:
            relation (TYPE): Description
            value (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        instances = []
        for i in self.instance_list:
            if getattr(i, relation) > value:
                instances.append(i)

        return instances

    def less_than_configs(self, relation, value):
        """Summary
        
        Args:
            relation (TYPE): Description
            value (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        instances = []
        for i in self.instance_list:
            if getattr(i, relation) < value:
                instances.append(i)
        return instances

    def get_instance(self, scene, f, g):
        """Summary
        
        Args:
            scene (TYPE): Description
            f (TYPE): Description
            g (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        for i in self.instance_list:
            if i.config_row_match([scene, f, g]):
                return i


if __name__ == '__main__':
    # First preprocess features
    preprocess_features.process_all_features()

    ### Semantic Annotations
    ### Collect annotation instances and attach values to them
    svcollection = SemanticCollection("2019 study")

    svcollection.write_preposition_stats_csvs()
    svcollection.write_config_ratios()

    #### Comparative Annotations

    compcollection = ComparativeCollection("2019 study")

    compcollection.write_preposition_stats_csvs()
    compcollection.get_constraints()
# compcollection.write_config_ratios()

## Collect all possible configurations and attach values to them

# configurations = ConfigurationCollection()

# configurations.write_data_csv()

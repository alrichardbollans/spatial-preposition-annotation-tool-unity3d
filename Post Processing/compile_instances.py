## Run process_data and preprocess_features before this

## Input: cleaned annotation, user lists and list of feature values for configurations (see Relationship class)
## Compiles annotation instances, adds feature values to them
## Output: For each task: Basic stats for each preposition. For sv task writes a csv of feature values with selection information

## Also has functions to generate constraints from comparative data

import csv
import os
import numpy as np
import uuid
import itertools

from sklearn.model_selection import train_test_split



import preprocess_features
from classes import Instance, Configuration, SceneInfo, CompInstance, Constraint, Comparison, Relationship, BasicInfo

### Files are shared with process_data
project_folder_name = BasicInfo.project_folder_name
feature_data_folder_name = BasicInfo.feature_data_folder_name 
data_folder_name = BasicInfo.data_folder_name
stats_folder_name = BasicInfo.stats_folder_name
sem_annotations_name  = BasicInfo.sem_annotations_name
comp_annotations_name  =  BasicInfo.comp_annotations_name







	
class User:
	def __init__(self,name):
		self.name = name

		self.value_list = []

	def append_values(self,instance_list):
		for i in instance_list:
			if i.user == self.name:
				# if hasattr(i, 'containment'):
				self.value_list.append([i.support,i.containment,i.contact,i.contact_scaled,i.above_measure, i.ground_verticalness])

class Preposition():
	# Preposition is instantiated for a particular list of instances
	#each preposition ends up with an associated array
	def __init__(self,name,instance_list):
		self.name = name

		self.value_list = []
		self.instance_list = self.get_instances(instance_list)
		self.config_list = self.get_configs()
	
	def get_instances(self,instance_list):
		instances = []
		for i in instance_list:
			if i.preposition == self.name or self.name == 'all':
				if i.figure != "none":
					instances.append(i)
		return instances
	## Get configurations and appends value rows to value list
	def get_configs(self):
		configs = []
		for i in self.instance_list:
			c = Configuration(i.scene,i.figure,i.ground)
			
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
		self.array = np.array(self.value_list)

	def average_value(self,relation):
		### Note this uses all selected instances rather than all distinct instances e.g. if a config is selected more it will skew value - which may be desirable depending on use
		counter = 0
		values = 0
		
		for i in self.instance_list:
			# print(i.scene)
			values += getattr(i,relation)
			counter += 1
		if counter == 0:
			return 0
		else:
			average = values/counter
			return average

	




class Collection:
	## Generic collection class
	## Does not require annotations/instances
	relation_keys = Relationship.get_relation_keys()
	feature_keys = Relationship.get_feature_keys()
	preposition_list=['in', 'inside', 'against','on','on top of', 'under',  'below',  'over','above'] # list of prepositions which exist in the data
	
	def __init__(self):
		## Annotation list is raw list of annotations from csv
		## Useful for counting number of tests of particular configuration
		self.annotation_list = []
		## Instance list is processed annotation list into list of separate instance objects
		self.instance_list = []

	def append_values(self):

		for i in self.instance_list:
			if i.figure != "none":
				try:
						
					r = Relationship(i.scene,i.figure,i.ground)
					r.load_from_csv()

					for key in  r.feature_keys:
						setattr(i,key,r.set_of_features[key])
						

				
				except Exception as e:
					print('Instance not added')
					print('Figure: ' + i.figure)
					print('Ground: ' + i.ground)
					
					print(e)

	

	def get_relation_values(self,relation):
		set1 = []
		for i in self.instance_list:
			set1.append(getattr(i,relation))
		return set1

	def get_mean_relation_value(self,relation):
		x = numpy.mean(self.get_relation_values(relation))
		return x

	def get_std_relation_value(self,relation):
		x = numpy.std(self.get_relation_values(relation))
		return x

class InstanceCollection(Collection):
	## Store some feature names for writing datasets
	ratio_feature_name = "selection_ratio"
	categorisation_feature_name = "selected_atleast_once"
	scene_feature_name = 'Scene'
	fig_feature_name = 'Figure'
	ground_feature_name = 'Ground'
	
	
	def __init__(self):
		Collection.__init__(self)
		
	### Instance Collection contains instances with a preposition
	def get_used_prepositions(self):
		out = []
		for i in self.instance_list:
			if i.preposition not in self.preposition_list:
		
				self.preposition_list.append(i.preposition)
			if i.preposition not in out:
				out.append(i.preposition)
		return out

	## Write csvs for each preposition giving feature values and data on number of selections
	## Also tags as test/training instance
	def write_config_ratios(self):
		scene_list = []
		s= SceneInfo()
		for scene in s.scene_list:
			scene_list.append(scene.name)
		
		
		for preposition in self.get_used_prepositions():
			config_list = Relationship.load_all()
			config_list.pop(0)
			
			

			## Write file of all instances
			with open('preposition data/'+self.filetag+'-ratio-list' + preposition + ' .csv', "w") as csvfile:
				outputwriter = csv.writer(csvfile)
				outputwriter.writerow(['Scene','Figure','Ground']+self.feature_keys+[self.ratio_feature_name,self.categorisation_feature_name])
							
				for row in config_list:
					
					c = Configuration(row[0],row[1],row[2])
					t = float(c.number_of_tests(self.annotation_list))
					s = float(c.number_of_selections(preposition,self.instance_list))
					
					## If at least one test has been done for this configuration
					if t != 0:
						ratio = s/t

						r= str(ratio)
						
						row.append(r)

						if(ratio ==0):
							row.append(str(0))
						else:
							row.append(str(1))
						
						# if c.scene in train_scenes:
						# 	row.append("train")
						# elif c.scene in test_scenes:
						# 	row.append("test")
						outputwriter.writerow(row)
	
						


	#### Write General Stats for each preposition
	def write_preposition_data_csvs(self):
		config_list = Relationship.load_all()
		
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
		with open(stats_folder_name+'/'+self.filetag+' preposition stats.csv', "w") as csvfile:
				outputwriter = csv.writer(csvfile)
				outputwriter.writerow(['','','','Average Values'])
				outputwriter.writerow(['Preposition','Number of Selections','Number of Distinct Configurations']+self.feature_keys)
			
		for preposition in self.get_used_prepositions() + ['all']:

			
			row = [preposition]
			
			p = Preposition(preposition,self.instance_list)
			

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


			with open(stats_folder_name+'/'+self.filetag+' preposition stats.csv', "a") as csvfile:
				outputwriter = csv.writer(csvfile)
				
				outputwriter.writerow(row)


	
class SemanticCollection(InstanceCollection):
	preposition_list=['in', 'inside', 'against','on','on top of', 'under',  'below',  'over','above'] # list of prepositions which exist in the data
	filetag = 'semantic'
	def __init__(self):
		InstanceCollection.__init__(self)
		self.append_annotations()

	


	def append_annotations(self):
		### Reads annotations from clean files
		### Must be updated if process_data prints csvs differently
		filename=data_folder_name + "/" + sem_annotations_name
		
		with open(filename, "r") as f:
			reader = csv.reader(f)     # create a 'csv reader' from the file object
			annotationlist = list( reader )  # create a list from the reader

		annotationlist.pop(0) #removes first line of data list which is headings
		
		for annotation in annotationlist:
			self.annotation_list.append(annotation)

			prepositions = annotation[4].split(";")

			for p in prepositions:
				if p != "":
					# scene = annotation[3][:annotation[3].index('-')] + '.blend'
					i = Instance(annotation[0],annotation[1],annotation[2],annotation[3],p,annotation[5],annotation[6])

					self.instance_list.append(i)
		
		self.append_values()
	
			

	


	


class ComparativeCollection(InstanceCollection):
	
	def __init__(self):
		
		InstanceCollection.__init__(self)
		self.filetag = 'comparative'
		self.append_annotations()
		# self.constraints = self.get_constraints()

	

	### Reads annotation file and appends to annotation and instance lists.
	def append_annotations(self):
		### Reads annotations from clean files
		### Must be updated if process_data prints csvs differently
		filename=data_folder_name + "/" + comp_annotations_name
		
		with open(filename, "r") as f:
			reader = csv.reader(f)     # create a 'csv reader' from the file object
			annotationlist = list( reader )  # create a list from the reader

		annotationlist.pop(0) #removes first line of data list which is headings
		
		for annotation in annotationlist:
			self.annotation_list.append(annotation)
			possible_figures = []
			index = 8
			while index < len(annotation):
				possible_figures.append(annotation[index])
				index+=1
			
					# scene = annotation[3][:annotation[3].index('-')] + '.blend'
			i = CompInstance(annotation[0],annotation[1],annotation[2],annotation[3],annotation[4],annotation[5],annotation[6],possible_figures)

			self.instance_list.append(i)
		
		self.append_values()
	
	def get_constraints(self):
		#First clear written constraints
		Constraint.clear_csv()
		## Creates a dictionary, prepositions are keys
		### Values are lists of constraints for the preposition
		out = dict()
		s_info = SceneInfo()
		
		
		for preposition in self.preposition_list:
			# print(preposition)
			C = []
			for s in s_info.scene_list:
				
				grounds = s.selectable_objects
				
				for grd in grounds:
					
					c = Comparison(s.name,preposition,grd)
					Cons = c.generate_constraints(self.instance_list)
					for con in Cons:
						C.append(con)
			out[preposition] = C
			# print(C)
			for con in C:
				con.write_to_csv()
		self.constraints = out
		return out
		





class ConfigurationCollection(Collection):

	def __init__(self):
		Collection.__init__(self)
		self.filetag = 'configs'
		## List of configuration instances
		self.instance_list = []
		self.append_configurations()
	
	

	def append_configurations(self):
		s= SceneInfo()
		for scene in s.scene_list:

			for c in scene.get_all_configs():
				if c[0] != c[1]:
					config  =  Configuration(scene.name,c[0],c[1])
					
					self.instance_list.append(config)

	

	def write_data_csv(self):
		## Writes a new csv giving all configs and data
		# Not sure this is necessary
		
		titles = ['Scene','Figure','Ground']+self.feature_keys
		
		with open(stats_folder_name+'/'+'configuration data.names', "w") as csvfile:
			outputwriter = csv.writer(csvfile)
			
			outputwriter.writerow(titles)
		
		with open(stats_folder_name+'/'+'configuration data.csv', "w") as csvfile:
			outputwriter = csv.writer(csvfile)
			
		
			for c in self.instance_list:
				
				outputwriter.writerow(c.full_row)				
	def write_preposition_data_csvs(self,preposition,datalist):
		conf = self.instance_list[1]

		with open('labelled config data/selection data.names', "w") as csvfile:
			outputwriter = csv.writer(csvfile)
			
			outputwriter.writerow(['scene','figure','ground']+conf.value_names + ['Selected?','Number of Selections'])
		### should edit this to only use configurations that appear in the 'all' preposition (as I mention in the write up)
		with open('configuration data.csv', "r") as readfile:
			reader = csv.reader(readfile)     # create a 'csv reader' from the file object
			config_list = list( reader )  # create a list from the reader 
			with open('labelled config data/selection data-' + preposition+'.csv', "w") as csvfile:
				outputwriter = csv.writer(csvfile)
				for row in config_list:		
					for c in self.instance_list:
						if c.config_row_match(row):
							x = c.number_of_selections(preposition,datalist)
							if x > 0:
									row.append(1)
							else:
								row.append(0)
							row.append(x)
							outputwriter.writerow(row)



	def greater_than_configs(self,relation,value):
		instances = []
		for i in self.instance_list:
			if getattr(i,relation) > value:
				instances.append(i)

		return instances



	def less_than_configs(self,relation,value):
		instances = []
		for i in self.instance_list:
			if getattr(i,relation) < value:
				instances.append(i)
		return instances

	def get_instance(self,scene,f,g):
		for i in self.instance_list:
			if i.config_row_match([scene,f,g]):
				return i

if __name__ == '__main__':
	# First preprocess features
	f= preprocess_features.Features()
	nd = f.standardise_values()
	f.write_new(nd)
	f.write_mean_std()
	
	### Semantic Annotations
	### Collect annotation instances and attach values to them
	svcollection = SemanticCollection()

	

	svcollection.write_preposition_data_csvs()
	svcollection.write_config_ratios()


	#### Comparative Annotations

	compcollection = ComparativeCollection()

	compcollection.write_preposition_data_csvs()
	compcollection.get_constraints()
	# compcollection.write_config_ratios()


	## Collect all possible configurations and attach values to them

	# configurations = ConfigurationCollection()

	# configurations.write_data_csv()
	




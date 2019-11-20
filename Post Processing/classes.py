import itertools
import os
import csv
import numpy as np
import pandas as pd
import math
import ast

from relationship import *

preposition_list=['in', 'inside', 'against','on','on top of', 'under',  'below',  'over','above']

def clean_name(object_name):
	if '.' in object_name:
		clean_name = object_name[:object_name.find(".")]
	elif '_' in object_name:
		clean_name = object_name[:object_name.find("_")]
	else: 
		clean_name = object_name
	return clean_name.lower()

def remove_dot_unity(scene_name):
	clean_name = scene_name
	if '.unity' in scene_name:
		clean_name = scene_name[:scene_name.find(".unity")]
	
	return clean_name
class Constraint:
	
	### A constraint is a linear inequality
	## Left hand side is more typical than right hand side
	## LHS is values of first config
	## RHS is values of second config
	
	output_path = "constraint data/constraints.csv"
	titles = ["scene","preposition","ground","f1","f2","weight","parity","weak","lhs","rhs"]

	def __init__(self,scene,preposition,ground,f1,f2,weight,parity,weak,lhs,rhs):
		self.scene = scene
		self.preposition = preposition
		self.ground = ground
		# f1,f2 are figures being compared
		# f1 should be more typical than f2 to satisfy constraint
		self.f1 = f1
		self.f2 = f2
		## lhs and rhs are arrays of coefficients for the problem
		## coefficients are ordered by Relationship.feature_keys/relation_keys
		## These are configuration values for the instances being compared
		self.lhs = lhs##np.array([lhs])
		self.rhs = rhs##np.array([rhs])
		
		## Weight given to constraint
		self.weight = weight

		## We have two types of constraints
		## Some arise from difference in the data and some from similarity
		
		self.parity = parity
		
		# If True, this constraint is for weak instances i.e. LHS is some distance from prototype
		## RHS is not important
		self.weak = weak

		self.csv_row = [self.scene,self.preposition,self.ground,self.f1,self.f2,self.weight,str(parity),str(weak), lhs.tolist(), rhs.tolist()]
		
		

	# def row_match(self,row):
	# 	if self.scene== row[0] and self.f1 == row[1] and self.f2 == row[2]
	def write_to_csv(self):
		

		
		with open(self.output_path, 'a') as csvfile:
			outputwriter = csv.writer(csvfile)
			outputwriter.writerow(self.csv_row)
	@staticmethod
	def read_from_csv():
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
			
				
			
			c = Constraint(line["scene"],line["preposition"],line["ground"],line["f1"],line["f2"],line["weight"],line["parity"],line["weak"],lhs,rhs)
		
			out[c.preposition].append(c)
		return out
	@staticmethod
	def clear_csv():
		with open(Constraint.output_path, 'w') as csvfile:
			outputwriter = csv.writer(csvfile)
			outputwriter.writerow(Constraint.titles)
			# csvfile.truncate()
	# def get_values(self,W,P):
	# 	## Works out the Similarity of LHS and RHS to prototype,P, given feature weights W
	# 	## Works out the Euclidean distance of the instance represented by LHS and RHS to the prototype
	# 	## W is a 1D array, an assignment of weights and P, Prototype
	# 	## First take away P from lhs and rhs values
	# 	## Then do dot product

		
	# 	lhs = np.subtract(self.lhs,P)
	# 	rhs = np.subtract(self.rhs,P)

	# 	lhs = np.square(lhs)
	# 	rhs = np.square(rhs)

	# 	lhs_distance = math.sqrt(np.dot(lhs,W))
	# 	rhs_distance = math.sqrt(np.dot(rhs,W))

	# 	lhs_value = math.exp(-lhs_distance)
	# 	rhs_value = math.exp(-rhs_distance)

	# 	# print(lhs_value)

	# 	return [lhs_value,rhs_value]

	## Returns True if X satisfies C
	## False otherwise

	def is_satisfied(self,lhs_value,rhs_value):
		## Values get calculated elsewhere depending on model
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
			if lhs_value>rhs_value:
				return True
			else:
				return False

	## Returns the value of RHS-LHS, with values from X
	def separation(self,X):
		lhs_value = np.dot(self.lhs,X)
		rhs_value = np.dot(self.rhs,X)

		return rhs_value-lhs_value

	# def write_constraint(self):

	# def read_constraints(self):


class Comparison():
	def __init__(self,scene,preposition,ground):
		self.preposition = preposition
		self.scene = scene
		self.ground = ground
		# self.annotations = self.get_annotations(datalist)
		# self.figure_selection_number = self.get_choices()
		self.possible_figures = self.get_possible_figures()
		### Note the following doesn't account for the fact that users can select none
		self.chance_agreement = 1/float(len(self.possible_figures))
		self.chance_agreement_with_none = 1/float(len(self.possible_figures) + 1)

	def generate_constraints(self,instancelist):
		
		### Takes an annotation list
		### Generates constraints and outputs a list
		C = []
		instances = self.get_instances(instancelist)
		# For the moment the metric constraint are commented out
		## Don't want to generate constraints when no tests have been done!
		if len(instances) > 0:
			
			figure_selection_number = self.get_choices(instancelist)

			pairs = list(itertools.combinations(self.possible_figures,2))
			
			for p in pairs:
				
				f1 = p[0]
				
				f2 = p[1]

				x1 = figure_selection_number[f1]

				x2 = figure_selection_number[f2]

				c1 = Configuration(self.scene,f1,self.ground)
				c2 = Configuration(self.scene,f2,self.ground)
				

				# Only use values that are not from 'context' features
				c1_array = np.array(c1.relations_row)
				c2_array = np.array(c2.relations_row)
				
				if x1 == x2:
					pass
					# if any (x > x1 for x in figure_selection_number.itervalues()):
					# 	# This case may happen because there is a better figure
					# 	# In which case we create no constraint
					# 	pass
					# elif x1 != 0:
					# 	# In the case that x1,x2 are not zero
					# 	# They are similar enough instances of the preposition to be confused
						 
					# 	weight = x1
						
					# 	con = Constraint(self.scene,self.preposition,self.ground,f1,f2,weight,True,False,c1_array,c2_array)
					# 	C.append(con)
					# 	con.write_to_csv()
				elif x1 < x2:
					weight = x2 - x1
					con = Constraint(self.scene,self.preposition,self.ground,f2,f1,weight,False,False,c2_array,c1_array)
					C.append(con)

				else:
					weight = x1 - x2
					con = Constraint(self.scene,self.preposition,self.ground,f1,f2,weight,False,False,c1_array,c2_array)
					C.append(con)

				
			
			## Deal with case that figure is not ever selected
			# for fig in figure_selection_number:
			# 	x1 = figure_selection_number[fig]
			# 	if x1 == 0:
			# 		pass
			# 		c = Configuration(self.scene,fig,self.ground)
					
					

			# 		# Only use values that are not from 'context' features
			# 		c_array = np.array(c.relations_row)

			# 		# Add a constraint saying this is a weak configuration
			# 		weight = float(len(instances))/len(self.possible_figures)
			# 		con = Constraint(self.scene,self.preposition,self.ground,fig,fig,weight,False,True,c_array,c_array)

			# 		C.append(con)
		# if len(none_instances) > 0:
		for con in C:
			con.write_to_csv()
		return C
	def get_instances(self,instancelist):
		out = []
		for i in instancelist:
			# if i.figure != "none":
			if self.instance_match(i):
				out.append(i)
		return out
	def get_none_instances(self,instancelist):
		out = []
		for i in instancelist:
			if i.figure == "none":
				if self.instance_match(i):
					out.append(i)
		return out
	def instance_match(self,i):
		if i.scene == self.scene and i.preposition == self.preposition and i.ground ==self.ground:#
			return True
		else:
			return False
	def get_choices(self,instancelist):
		out = dict()
		for f in self.possible_figures:
			out[f] = 0
		for i in self.get_instances(instancelist):
			# print(i.figure)
			f = i.figure
			if f in out:
				out[f] +=1
			else:
				out[f] = 1
		return out

	def get_possible_figures(self):
		
		s_info = SceneInfo()
		out = []
		
		for s in s_info.scene_list:
			
			
			if s.name == self.scene:
				
				## Selectable objects in scene
				selectable_objects = s.selectable_objects
				
				## Remove ground
				for g in selectable_objects[:]:
					if g == self.ground:
						selectable_objects.remove(g)

				out = selectable_objects
		return out
				
				


class MyScene:
	## Class to store info about scenes
	unselectable_scene_objects = ["wall","floor","ceiling"]
	## This is given in MyScene class in Unity

	example_scenes = ["example","finish","instruction","template","main","player","screen","test"]

	def __init__(self,name,mesh_objects):
		self.name = name
		self.mesh_objects = mesh_objects
		self.selectable_objects = self.get_selectable_objects()#
		# Bool. Is this scene used for study?
		self.study_scene = self.study_scene_check()

	def get_selectable_objects(self):
		out = []
		for obj in self.mesh_objects:
			if not any(x in obj for x in self.unselectable_scene_objects):
				out.append(obj)
		return out
			
	def get_all_configs(self):
		out = []
		
		x = list(itertools.permutations(self.selectable_objects, 2) )
			
		return x

	def study_scene_check(self):
		if any (x in self.name  for x in self.example_scenes):
			return False
			# print(self.name)
		else:
			return True
			# print(self.name)
		

class SceneInfo:
	### A class to store info on the scenes
	### scene_list is a collection of MyScene objects
	### Relies on creating a csv file in Unity Editor using write_scene_info.cs script
	### Then also run commonsense properties script to get ground info
	project_path = get_project_directory()
	output_path = project_path + "/Data Collection Game"+ "/Scene Data/";
	filename = "scene_info.csv";
	csv_file = output_path + filename; 

	
	def __init__(self):
		self.data_list = self.get_list()
		self.scene_list = self.get_scenes()



	def get_list(self):
		

		with open(self.output_path + self.filename, "r") as f: 
			reader = csv.reader(f)     
			datalist = list( reader )
		return datalist

	def get_scenes(self):#,datalist):
		# self.data_list.pop(0) #removes first line of data list which is headings
		# datalist=datalist[::-1] #inverts data list to put in time order
		out = []
		
		for s in self.data_list:
			mesh_objects = []
			for i in range(1,len(s)):
				mesh_objects.append(s[i])
			if len(mesh_objects) != 0 and ".unity" in s[0]:
				s_name = remove_dot_unity(s[0])
				scene = MyScene(s_name,mesh_objects)
				
				if scene.study_scene:
					# print(scene.name)
					out.append(scene)
			

		return out




	


class Configuration:
	def __init__(self,scene,figure,ground):
		self.scene = scene
		self.figure = figure
		self.ground = ground
		# Row of feature values for outputing to csv
		self.row = []
		### Row beginning with names
		self.full_row = [self.scene,self.figure,self.ground]
		### Row without ground properties
		self.relations_row =[]
		if self.figure != "none":
			self.append_values()

	def __str__(self):
		return "["+str(self.scene)+","+str(self.figure)+","+str(self.ground)+"]"
	
	def append_values(self):
		# print(self.figure)
		# print(self.ground)
		# print(self.scene)
		r = Relationship(self.scene,self.figure,self.ground)
		r.load_from_csv()

		for key in  r.feature_keys:

		
			value = r.set_of_features[key]
			setattr(self,key,value)
			self.row.append(value)
			self.full_row.append(value)
			if key not in r.context_features:
				self.relations_row.append(value)


	def configuration_match(self,instance):
		if self.scene == instance.scene and self.figure == instance.figure and self.ground == instance.ground:
			return True
		else:
			return False
	def number_of_selections(self,preposition,instancelist):
		counter = 0
		for i in instancelist:
			if self.configuration_match(i) and i.preposition == preposition:
				counter += 1
		return counter
	def number_of_tests(self,annotationlist):
		## Need to use annotation list here as instances are separated by preposition
		counter = 0
		for an in annotationlist:
			
			if self.annotation_row_match(an):

				counter += 1
		# print(counter)
		return counter

	def ratio_semantic_selections(self,preposition,annotationlist,instancelist):
		t = float(self.number_of_tests(annotationlist))
		s = float(self.number_of_selections(preposition,instancelist))
		
		if t != 0:
			return s/t
		else:
			return 0
	def annotation_row_match(self,row):
		## If configuration matches with annotation in raw data list
		if self.scene == row[3] and self.figure == row[5] and self.ground == row[6]:
			return True
		else:
			return False
	def config_row_match(self,value):
		if self.scene == value[0] and self.figure == value[1] and self.ground == value[2]:
			return True
		else:
			return False
	
	# def create_row(self):
	# 	for value in self.value_names:	
	# 		try:
	# 			self.row.append(getattr(self,value))
	# 		except Exception as e:
	# 			self.row.append('?')
	# 			print('Value not added')
	# 			print('Figure: ' + self.figure)
	# 			print('Ground: ' + self.ground)
				
	# 			print('Scene: ' + self.scene)
	# 			print('Value: ' + value)

	# 			print(e)
	def print_info(self):
		print('Scene = ' + self.scene)
		print('Figure = ' + self.figure)
		print('Ground = ' + self.ground)


class Instance(Configuration):
	def __init__(self,ID,user,task,scene,preposition,figure,ground):
		Configuration.__init__(self,scene,figure,ground)
		self.task = task
		self.preposition = preposition
		self.id = ID
		self.user = user
	def __str__(self):
		return "["+str(self.scene)+","+str(self.figure)+","+str(self.ground)+"]"
	def print_info(self):
		Configuration.print_info()
		print('preposition = ' + self.preposition)
		print('annotation id = ' + self.id)
		print('user = ' + self.user)

class CompInstance(Configuration):
	
	###Figure is the selected figure
	### We can consider this as a usual preposition instance and plot it
	### /compare with selection instances
	def __init__(self,ID,user,task,scene,preposition,figure,ground,possible_figures):
		Configuration.__init__(self,scene,figure,ground)
		self.possible_figures = possible_figures
		self.task = task
		self.preposition = preposition
		self.id = ID
		self.user = user
		

	def print_info(self):
		Configuration.print_info(self)
		print('preposition = ' + self.preposition)
		print('annotation id = ' + self.id)
		print('user = ' + self.user)

	


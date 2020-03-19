# Script to run for newly collected data files which:
# Input: annotation and user info csv from data collection
# Output: Clean annotation lists. Basic stats. User agreement calculations
# Feature values are included later

import csv
import itertools

from classes import Comparison, BasicInfo






class User():

	def __init__(self,clean_id,user_id,time,native): #The order of this should be the same as in writeuserdata.php
		
		self.user_id = user_id
		self.time = time
		
		self.native = native
		self.clean_user_id = clean_id

		self.list_format = [self.user_id,self.clean_user_id,self.time,self.native]
		self.list_headings = ['User ID','Short ID','Time','Native=1, Non-Native =0']

	
	def annotation_match(self,annotation):
		if annotation.clean_user_id == self.user_id or annotation.clean_user_id == self.clean_user_id:
			return True
		else:
			return False

class UserData():
	def __init__(self):
		self.raw_data_list=self.load_raw_users_from_csv()
		self.user_list = self.get_users()
		
	def load_raw_users_from_csv(self):
		

		with open(BasicInfo.data_folder_name + '/'  + BasicInfo.raw_user_list, "r") as f: 
			reader = csv.reader(f)     
			datalist = list( reader )
		return datalist

	def get_users(self):#,datalist):
		# datalist.pop(0) #removes first line of data list which is headings
		# datalist=datalist[::-1] #inverts data list to put in time order
		out = []
		i=1
		for user in self.raw_data_list:
			u = User(i,user[0],user[1],user[2]) # Keep order 1,2,3,etc..
			
			out.append(u)
			i += 1

		return out

	def output_clean_user_list(self):
		with open(BasicInfo.data_folder_name + '/' +'clean_users.csv', "w") as csvfile:
			writer = csv.writer(csvfile)

			heading = self.user_list[0].list_headings
			writer.writerow(heading)
			
			for user in self.user_list:
				writer.writerow(user.list_format)
	def get_non_natives(self):
		out = []
		for u in self.user_list:
			if u.native == '0':
				out.append(u.clean_user_id)
				print("Non-Native: " + str(u.clean_user_id))
		return out
				
class Annotation:
	# Gets annotations from the raw data
	

	def __init__(self,userdata,annotation):#ID,UserID,now,selectedFigure,selectedGround,task,scene,preposition,prepositions,cam_rot,cam_loc):
		self.id = annotation[BasicInfo.a_index['id']]
		self.user_id = annotation[BasicInfo.a_index['userid']]
		self.time = annotation[BasicInfo.a_index['time']]
		selectedFigure = annotation[BasicInfo.a_index['figure']]
		if selectedFigure == "":
			self.figure = "none"

		else:
			self.figure = selectedFigure
		self.ground = annotation[BasicInfo.a_index['ground']]
		self.task = annotation[BasicInfo.a_index['task']]
		self.scene = annotation[BasicInfo.a_index['scene']]
		self.preposition = annotation[BasicInfo.a_index['preposition']]
		self.prepositions = annotation[BasicInfo.a_index['prepositions']]
		
		self.cam_rot =annotation[BasicInfo.a_index['cam_rot']]
		self.cam_loc = annotation[BasicInfo.a_index['cam_loc']]
		
		
		
		for user in userdata.user_list:
			if user.user_id == self.user_id:
				self.user = user
				self.clean_user_id = user.clean_user_id

		self.list_format = [self.id,self.clean_user_id,self.task,self.scene,self.preposition,self.prepositions,self.figure,self.ground,self.time]
		self.list_headings = ['Annotation ID','Clean User ID','Task','Scene','Preposition', 'Prepositions','Figure','Ground', 'Time']
	def clean_name(object_name):
		if '(' in object_name:
			clean_name = object_name[:object_name.find(".")]
		elif '_' in object_name:
			clean_name = object_name[:object_name.find("_")]
		else: 
			clean_name = object_name
		return clean_name.lower()

class ComparativeAnnotation(Annotation):
	# Users selects a figure given a ground and preposition
	def __init__(self,userdata,annotation):#ID,UserID,now,selectedFigure,selectedGround,scene,preposition,prepositions,cam_rot,cam_loc):
		Annotation.__init__(self,userdata,annotation)
		# list format is used to write rows of csv
		c = Comparison(self.scene,self.preposition,self.ground)
		self.possible_figures = c.possible_figures
		# Need to append possible figures to list format and then deal with this in compile_instances
		self.list_format = [self.id,self.clean_user_id,self.task,self.scene,self.preposition,self.figure,self.ground,self.time]
		
		for f in self.possible_figures:
			self.list_format.append(f)
		self.list_headings = ['Annotation ID','Clean User ID','Task','Scene', 'Preposition','Figure','Ground', 'Time']
	def print_list(self):
		print([self.id,self.clean_user_id,self.preposition,self.scene,self.figure,self.ground,self.time])


class SemanticAnnotation(Annotation):
	# User selects multiple prepositions given a figure and ground
	def __init__(self,userdata,annotation):#ID,UserID,now,selectedFigure,selectedGround,scene,preposition,prepositions,cam_rot,cam_loc):
		Annotation.__init__(self,userdata,annotation)
		self.preposition_list = self.make_preposition_list()
		self.list_format = [self.id,self.clean_user_id,self.task,self.scene,self.prepositions,self.figure,self.ground,self.time]
		self.list_headings = ['Annotation ID','Clean User ID','Task','Scene', 'Prepositions','Figure','Ground', 'Time']
	def print_list(self):
		print([self.id,self.clean_user_id,self.prepositions,self.scene,self.figure,self.ground,self.time])

	def make_preposition_list(self):
		
		x= self.prepositions.split(';')
		return x


class Data():
	
	
	

	def __init__(self,userdata):
		self.alldata = self.load_annotations_from_csv()
		self.annotation_list = self.get_annotations(userdata)
		
		# Annotation list without non-natives
		self.clean_data_list = self.clean_list()
		
		self.user_list = self.get_users()
		self.native_users = self.get_native_users()
		self.scene_list = self.get_scenes()
		
		self.clean_csv_name = "all_clean_annotations.csv"
		self.task = "all"

		

	def load_annotations_from_csv(self):

		with open(BasicInfo.data_folder_name + '/'  + BasicInfo.raw_annotation_list, "r") as f: 
			reader = csv.reader(f)     
			datalist = list( reader )
		return datalist
	
	def get_annotations(self,userdata):
		
		out = []
		for annotation in self.alldata:
			ann = Annotation(userdata,annotation)
			
			out.append(ann)
		# print(len(out))	
		return out


	

	def get_non_users(self):
		out = []
		for user in userdata.user_list:
			x = self.number_of_scenes_done_by_user(user,"sv")
			y = self.number_of_scenes_done_by_user(user,"comp")
			if x == 0 and y ==0:
				out.append(user)
		return out
	def get_users(self):
		out = []
		for an in self.annotation_list:
			if an.user not in out:
				out.append(an.user)
			
		return out

	def get_native_users(self):
		out = []
		for u in self.user_list:
			if u.native == '1':
				out.append(u)
		return out

	def get_scenes(self):
		out = []

		for an in self.annotation_list:
			if an.scene not in out:
				out.append(an.scene)
		return out 

	def get_scenes_done_x_times(self,x):
		out = []
		# This only counts native speakers
		for sc in self.scene_list:
			y = self.number_of_users_per_scene(sc)
			if y >= x:
				out.append(sc)
		return out
	
	def print_scenes_done_x_times(self,x,task):
		# This only counts native speakers
		for sc in self.scene_list:
			y = self.number_of_users_per_scene(sc,task)
			if y >= x:
				print("Scene: " + sc +" done " + str(y) + "times")
	def print_scenes_need_doing(self):
		sl = BasicInfo.get_scene_list()
		print("Total number of scenes:" + str(len(sl)))
		out = []
		# This only counts native speakers
		for sc in self.scene_list:
			x = self.number_of_users_per_scene(sc,"sv")
			y = self.number_of_users_per_scene(sc,"comp")
			if x< 3 or y <3:
				out.append(sc)
				print("Scene: " + sc +" sv done " + str(x) + "times" +" comp done " + str(y) + "times")
		print("Number of scenes left: ")
		print(len(out))
		return out
	def print_scenes_need_removing(self):
		print("To remove")
		out = []
		# This only counts native speakers
		for sc in self.scene_list:
			x = self.number_of_users_per_scene(sc,"sv")
			y = self.number_of_users_per_scene(sc,"comp")
			if x>= 3 and y >=3:
				out.append(sc)
				print("Scene: " + sc +" sv done " + str(x) + "times" +" comp done " + str(y) + "times")
		print("Number of scenes to remove: ")
		print(len(out))
		return out
	def number_of_scenes_done_by_user(self,user,task):
		out = []
		for annotation in self.annotation_list:
			if user.annotation_match(annotation):
			# if annotation.clean_user_id == user or annotation.user_id == user:
				if annotation.task == task:
					if annotation.scene not in out:
						out.append(annotation.scene)
		
		return len(out)
	
	def number_of_users_per_scene(self,scene,task):
		scenecounter = []
		for annotation in self.clean_data_list:

			if annotation.scene == scene and annotation.clean_user_id not in scenecounter:
				if annotation.task == task:
					scenecounter.append(annotation.clean_user_id)
				
		return len(scenecounter)

	def remove_non_natives(self,x):
		for annotation in x[:]:
			if annotation.user.native == '0':# in userdata.list_non_natives:
				x.remove(annotation)
		return x
	def number_of_completed_users(self):
		i=0
		for user in self.user_list:
			x = self.number_of_scenes_done_by_user(user,"sv")
			y = self.number_of_scenes_done_by_user(user,"comp")
			# print(user)
			# print(x)
			# print(y)

			if  x == 10 and  y == 10:
				i+=1
		return i

	def total_number_users(self):
		
		return len(self.user_list)

	def output_clean_annotation_list(self):
		with open(BasicInfo.data_folder_name + '/'  + self.clean_csv_name, "w") as csvfile:
			writer = csv.writer(csvfile)
			# self.list_format = [self.id,self.clean_user_id,self.task,self.scene,self.preposition,self.figure,self.ground,self.time]
			heading = self.clean_data_list[0].list_headings
			writer.writerow(heading)
			
			for annotation in self.clean_data_list:
				writer.writerow(annotation.list_format)
	def clean_list(self):
		
		out = self.annotation_list[:]
		
		out = self.remove_non_natives(out)
		
		
		
		return out

	# Gets user annotations for a particular task
	def get_user_task_annotations(self,user1,task):
		out = []
		for a in self.annotation_list:
			if a.task == task:
				if user1.annotation_match(a):# a.clean_user_id == user1 or a.user_id == user1:
					out.append(a)
		return out

	# Gets user annotations for a particular task where they didn't select none
	def get_user_affirmative_task_annotations(self,user1,task):
		out = []
		for a in self.annotation_list:
			if a.task == task:
				if user1.annotation_match(a):

					if task in BasicInfo.comparative_abbreviations and a.figure != "none":
						
						out.append(a)

					if task in BasicInfo.semantic_abbreviations and "none" not in a.prepositions:
						out.append(a)

		return out
	

	# Compares two annotations. Returns true if the same question is being asked of the annotators.
	def question_match(self,a1,a2):
		if a1.task == a2.task:
			if a1.task in BasicInfo.comparative_abbreviations:
				if  a1.scene == a2.scene and a1.ground ==a2.ground and a1.preposition == a2.preposition:
					return True
				else:
					return False
			elif a1.task in BasicInfo.semantic_abbreviations:
				if  a1.scene == a2.scene and a1.ground ==a2.ground and a1.figure == a2.figure:
					return True
				else:
					return False
			else:
				print("Task mismatch in 'question_match()'")
				print(a1.task)
				print(a2.task)
				return False
		else:
			return False

	def write_user_agreements(self):
		with open(BasicInfo.stats_folder_name+'/' + self.agreements_csv_name, "w") as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(["Task: " + self.task,'Number of Native English users: ' + str(len(self.native_users))])
			# writer.writerow(['User1','User2', 'observed_agreement','Number of Shared Annotations', 'Number of agreements', 'Expected Number of Agreements','observed_agreement(AFF)','Number of Shared Annotations(AFF)', 'NUmber of agreements(AFF)', 'Expected Agreement(AFF)'])
			number_of_comparisons = 0
			total_shared_annotations = 0
			total_expected_agreement_sum = 0
			total_observed_agreement_sum = 0
			total_cohens_kappa_sum = 0

			

			writer.writerow(['Preposition','Number of Shared Annotations', 'Average Expected agreement', 'Average observed Agreement', 'Average cohens_kappa'])
				
			for p in BasicInfo.comparative_preposition_list:
				p_number_of_comparisons = 0
				preposition_shared_annotations = 0
				preposition_expected_agreement_sum = 0
				preposition_observed_agreement_sum = 0
				preposition_cohens_kappa_sum= 0
				
				user_pairs = list(itertools.combinations(self.native_users,2))
				for user_pair in user_pairs:
					user1 = user_pair[0]
					user2 = user_pair[1]
				# for user1 in self.native_users:
				# 	for user2 in self.native_users:
					if user1 != user2:
						# Calculate agreements for user pair and add values to totals

						x = Agreements(self.annotation_list,self.task,p,user1,user2)
						
						
						if(x.shared_annotations !=0):
							number_of_comparisons +=1
							p_number_of_comparisons += 1
							

							preposition_shared_annotations += x.shared_annotations
							preposition_expected_agreement_sum += x.expected_agreement * x.shared_annotations
							preposition_observed_agreement_sum += x.observed_agreement * x.shared_annotations
							preposition_cohens_kappa_sum += x.cohens_kappa * x.shared_annotations


							total_shared_annotations += x.shared_annotations
							total_expected_agreement_sum += x.expected_agreement * x.shared_annotations
							total_observed_agreement_sum += x.observed_agreement * x.shared_annotations
							total_cohens_kappa_sum += x.cohens_kappa * x.shared_annotations

							

				p_expected_agreement = float(preposition_expected_agreement_sum)/(preposition_shared_annotations)
				p_observed_agreement = float(preposition_observed_agreement_sum)/(preposition_shared_annotations)
				p_cohens_kappa = float(preposition_cohens_kappa_sum)/(preposition_shared_annotations)
				# Write a row for each preposition
				
				row = [p,preposition_shared_annotations,p_expected_agreement,p_observed_agreement,p_cohens_kappa]
				writer.writerow(row)
			
			total_expected_agreement = float(total_expected_agreement_sum)/(total_shared_annotations)
			total_observed_agreement = float(total_observed_agreement_sum)/(total_shared_annotations)
			total_cohens_kappa = float(total_cohens_kappa_sum)/(total_shared_annotations)

			# Write a row of total averages
			writer.writerow(['Total Number of Shared Annotations', 'Average Expected Agreements','Average observed agreements', 'Average Cohens Kappa'])
			row = [total_shared_annotations,total_expected_agreement,total_observed_agreement,total_cohens_kappa]
			writer.writerow(row)
		

class ComparativeData(Data):
	def __init__(self,userdata):
		self.userdata = userdata
		self.task = "comp"
		self.data_list = self.load_annotations_from_csv()
		self.annotation_list = self.get_comp_annotations(userdata)
		self.clean_data_list = self.clean_list()
		self.user_list = self.get_users()
		self.native_users = self.get_native_users()
		self.preposition_list = self.get_prepositions()
		self.clean_csv_name = BasicInfo.comp_annotations_name
		self.stats_csv_name = "comparative stats.csv"
		self.agreements_csv_name = "comparative agreements.csv"
		self.scene_list = self.get_scenes()


	def get_comp_annotations(self,userdata):
		# datalist.pop(0) #removes first line of data list which is headings
		# datalist=datalist[::-1] #inverts data list to put in time order
		out = []
		for annotation in self.data_list:
			ann = Annotation(userdata,annotation)
			if ann.task in BasicInfo.comparative_abbreviations:
				#__init__(self,userdata,ID,UserID,now,selectedFigure,selectedGround,task,scene,prepositions)
				# Match with appendannotation.php
				an = ComparativeAnnotation(userdata,annotation)#[BasicInfo.a_index['id']],annotation[1],annotation[2],annotation[3],annotation[4],annotation[5],annotation[6],annotation[7],annotation[8],annotation[9],annotation[10],annotation[11])

				out.append(an)
				# self.data_list.append(an)

				# self.clean_data_list.append(an)
		return out
	
	def get_prepositions(self):
		out = []
		for annotation in self.clean_data_list:
			if annotation.preposition not in out:
				out.append(annotation.preposition)
		return out
	def get_preposition_info_for_scene(self,scene):
		out = []
		
		for p in self.preposition_list:
			grounds = []
			for annotation in self.clean_data_list:
				if annotation.scene == scene and annotation.preposition == p:
					g = annotation.ground
					if g not in grounds:
						grounds.append(g)

			for grd in grounds:
				c = Comparison(scene,p,grd)
				instances = c.get_instances(self.clean_data_list)
				figure_selection_number = c.get_choices(self.clean_data_list)
				
				i = len(instances)
				row = []
				crow = [p,grd,i]
				row.append(crow)
				for f in figure_selection_number:
					x = [f,figure_selection_number[f]]
					row.append(x)
				out.append(row)

		return out

	# This is a very basic list of information about the task
	# compile_instances gives a better overview
	def output_statistics(self):
		with open(BasicInfo.stats_folder_name+'/' + self.stats_csv_name, "w") as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['Number of Native English users: ' + str(len(self.native_users))])
			writer.writerow(['Scene','Number of Users Annotating', 'Selection Info'])
			
			
			for s in self.scene_list:
				row = [s,self.number_of_users_per_scene(s,self.task),self.get_preposition_info_for_scene(s)]
				# for p in self.get_prepositions_for_scene(s):
				# 	row.append(p)
				writer.writerow(row)
	
	
class SemanticData(Data):
	def __init__(self,userdata):
		#The object from this class will be a list containing all the semantic annotations
		self.task = "sv"
		self.data_list = self.load_annotations_from_csv()
		self.annotation_list = self.get_semantic_annotations(userdata)
		self.clean_data_list = self.clean_list()
		# self.configuration_list = self.get_configurations()
		self.user_list = self.get_users()
		self.native_users = self.get_native_users()
		self.clean_csv_name = BasicInfo.sem_annotations_name
		self.stats_csv_name = "semantic stats.csv"
		self.agreements_csv_name = "semantic agreements.csv"
		# self.preposition_list = self.get_prepositions()
		self.scene_list = self.get_scenes()


	def get_semantic_annotations(self,userdata):
		# datalist.pop(0) #removes first line of data list which is headings
		# datalist=datalist[::-1] #inverts data list to put in time order
		out = []
		for annotation in self.data_list:
			ann = Annotation(userdata,annotation)
			if ann.task in BasicInfo.semantic_abbreviations:
				
				an = SemanticAnnotation(userdata,annotation)
				out.append(an)
				
		return out
	



	def get_prepositions_for_scene(self,scene):
		out = []
		for annotation in self.clean_data_list:
			if annotation.scene == scene:
				for p in annotation.preposition_list:
					if p not in out:
						out.append(p)

		return out


	def pair_prepositions_scenes(self):
		for scene in self.scene_list:
			if scene in self.get_scenes_done_x_times(1):
				print("Scene: " + scene)
				ps = self.get_prepositions_for_scene(scene)
				for p in ps:
					print(p)
	

	# Identifies number of times prepositions are selected or left blank
	def get_positive_selection_info(self):
		positive_selections = 0
		negative_selections = 0
		
		for p in BasicInfo.semantic_preposition_list: 
			for a in self.clean_data_list:
				if p in a.prepositions:
					positive_selections += 1
				elif p not in a.prepositions:
					negative_selections += 1
		return[positive_selections,negative_selections]

	# This is a very basic list of information about the task
	# compile_instances gives a better overview
	def output_statistics(self):
		with open(BasicInfo.stats_folder_name+'/' + self.stats_csv_name, "w") as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['Number of Native English users: ' + str(len(self.native_users))])

			writer.writerow(['Positive Selections','negative_selections'])
			row = self.get_positive_selection_info()
			writer.writerow(row)
			writer.writerow(['Scene','Number of Users Annotating', 'Given Prepositions'])
			
			
			for s in self.scene_list:
				writer.writerow([s,self.number_of_users_per_scene(s,self.task),self.get_prepositions_for_scene(s)])


	
class Agreements(Data):
	# Looks at agreements between two users for a particular task and particular preposition
	def __init__(self,annotation_list,task, preposition,user1,user2=None,agent_task_annotations=None):
		self.annotation_list = annotation_list
		self.user1 = user1
		self.user2 = user2
		
		self.task = task
		# All user annotations for particular task
		self.user1_annotations = self.get_user_task_annotations(user1,task)
		if user2 != None:
			self.user2_annotations = self.get_user_task_annotations(user2,task)
		else:
			# We can test agent models instead of users
			self.user2_annotations = agent_task_annotations

		
		self.preposition = preposition

		
		if self.task in BasicInfo.semantic_abbreviations and self.preposition == "none":
			print("Error: Checking 'none' agreement")
		self.user_calculations()
		


	# Agreement of users
	
	
	def count_sem_agreements(self):
		#Number of shared annotations by u1 and u2
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
				if self.question_match(a1,a2):
					if a1.task in BasicInfo.semantic_abbreviations:
						shared_annotations +=1
						if self.preposition in a1.prepositions:
							y1 +=1
						else:
							n1 +=1
						if self.preposition in a2.prepositions:
							y2 +=1
						else:
							n2 +=1
						
						if self.preposition in a1.prepositions and self.preposition in a2.prepositions:
							agreements +=1
							
						elif self.preposition not in a1.prepositions and self.preposition not in a2.prepositions:
							agreements +=1
		return shared_annotations,y1,y2,n1,n2,agreements
	def count_comp_agreements(self):
		#Number of shared annotations by u1 and u2
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
				if self.question_match(a1,a2):
					

					if a1.task in BasicInfo.comparative_abbreviations:
						if a1.preposition == self.preposition:
							shared_annotations +=1
							if a1.figure == "none":
								comp_none_selections1 +=1
							if a2.figure == "none":
								comp_none_selections2 += 1

							c = Comparison(a1.scene,a1.preposition,a1.ground)
							no_possible_selections = len(c.possible_figures)
							number_of_compared_figures += no_possible_selections
							
							if a1.figure == a2.figure:
								agreements +=1
		return shared_annotations,comp_none_selections1,comp_none_selections2,number_of_compared_figures,agreements
	
	def calculate_sem_expected_agreement(self,shared_annotations,y1,y2,n1,n2):
		
							
		if shared_annotations !=0:
			
			expected_agreement = float((y1*y2 + n1*n2))/float((shared_annotations)**2)
		else:
			expected_agreement = 0
		return expected_agreement
	def calculate_comp_expected_agreement(self,shared_annotations,comp_none_selections1,comp_none_selections2,number_of_compared_figures):
		if self.task in BasicInfo.comparative_abbreviations:
			if (shared_annotations != 0):
				u1_p_none = float(comp_none_selections1)/shared_annotations
				u2_p_none = float(comp_none_selections2)/shared_annotations
			

				expected_none_agreement = float(u1_p_none * u2_p_none)

				# As there are a different number of distractors in each scene and the distractors change
				# We make an approximation here and work out there overall chance of agreeing on an object
			
				
				average_probability_agree_on_object = float(shared_annotations * (1-u1_p_none) * (1-u2_p_none))/number_of_compared_figures
			

				expected_agreement = expected_none_agreement + average_probability_agree_on_object
			else:
				expected_agreement = 0
		return expected_agreement
	def user_calculations(self):
		observed_agreement = 0
		cohens_kappa = 0
		

		if self.task in BasicInfo.semantic_abbreviations:
			shared_annotations,y1,y2,n1,n2,agreements = self.count_sem_agreements()
		elif self.task in BasicInfo.comparative_abbreviations:
			shared_annotations,comp_none_selections1,comp_none_selections2,number_of_compared_figures,agreements = self.count_comp_agreements()
		
		if self.task in BasicInfo.semantic_abbreviations:
			expected_agreement = self.calculate_sem_expected_agreement(shared_annotations,y1,y2,n1,n2)
		elif self.task in BasicInfo.comparative_abbreviations:
			expected_agreement =self.calculate_comp_expected_agreement(shared_annotations,comp_none_selections1,comp_none_selections2,number_of_compared_figures)
		
		
		if shared_annotations != 0:
			observed_agreement = float(agreements)/float(shared_annotations)

		if observed_agreement != 1:
			cohens_kappa = float(observed_agreement - expected_agreement)/float(1- expected_agreement)
		else:
			cohens_kappa = 1

		self.shared_annotations = shared_annotations
		self.expected_agreement = expected_agreement
		self.observed_agreement = observed_agreement
		self.cohens_kappa = cohens_kappa

		
		

		



		
if __name__ == '__main__':
	# Begin by loading users  
	userdata = UserData()




	# Output user list
	userdata.output_clean_user_list()

	# Load all csv
	d= Data(userdata)
	

	# d.print_scenes_need_doing()
	# d.print_non_users()
	# d.output_clean_annotation_list()

	# 
	# Load and process semantic annotations
	semantic_data = SemanticData(userdata)



	# Output semantic csv
	semantic_data.output_clean_annotation_list()

	semantic_data.output_statistics()

	semantic_data.write_user_agreements()
	
	#Load and process comparative annotations
	comparative_data = ComparativeData(userdata)



	# output comparative csv

	comparative_data.output_clean_annotation_list()

	comparative_data.output_statistics()


	comparative_data.write_user_agreements()



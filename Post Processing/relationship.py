import csv
import os

from preprocess_features import Features

class Relationship:
	# Lots of this could be done with pandas. Doh :/

	
	property_path = Features.output_path
	output_path = property_path

	# additional_features = ["location_control"]
	# Location control is the average of the two more basic measures

	
	# Ground properties which we think distinguish polysemes
	context_features = ["ground_lightsource","ground_container","ground_verticality"]
	def __init__(self,scene,figure,ground):
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
	def load_all(path = None):
		# Loads a list of all configurations and feature values, with some features removed
		# Path variable optional
		if path == None:
			path = Relationship.property_path
		with open(path, "r") as f:
			reader = csv.reader(f)     # create a 'csv reader' from the file object
			geom_relations = list( reader )  # create a list from the reader
		
		return geom_relations
		

	@staticmethod
	def get_feature_keys():
		feature_keys = []
		
		geom_relations = Relationship.load_all()
		for title in geom_relations[0][3:]:
			feature_keys.append(title)
		
		return feature_keys

	@staticmethod
	def get_relation_keys():
		relation_keys = []
		
		geom_relations = Relationship.load_all()
		for title in geom_relations[0][3:]:
			if title not in Relationship.context_features:
				relation_keys.append(title)
		
		return relation_keys

	def load_from_csv(self,path= None):
		
		
				
		if path != None:
			geom_relations = Relationship.load_all(path)
		else:
			geom_relations = Relationship.load_all()

		for title in geom_relations[0][3:]:
			self.feature_keys.append(title)
		for relation in geom_relations:
			if self.scene == relation[0] and self.figure == relation[1] and self.ground == relation[2]:
				# print(geom_relations.index(relation))
				for r in self.feature_keys:
					if relation[self.feature_keys.index(r)+3] != '?':
						self.set_of_features[r] =float(relation[self.feature_keys.index(r)+3])
					else:
						self.set_of_features[r] = '?'
		# # Add and calculate additional features
		# self.feature_keys.append("location_control")
		# self.set_of_features["location_control"] = (self.set_of_features["location_control_x"] + self.set_of_features["location_control_z"])/2
		

	def save_to_csv(self):
		

		row = [self.scene,self.figure,self.ground]

		for r in feature_keys:
			if r in self.set_of_features:
				row.append(self.set_of_features[r])
			else:
				row.append('?')
				self.set_of_features[r] = '?'
		
		with open(Relationship.output_path) as incsvfile:
			read = csv.reader(incsvfile) #.readlines())
			reader = list(read)

			if any(self.scene == line[0] and self.figure == line[1] and self.ground == line[2] for line in reader):
				try:
					with open(Relationship.output_path, "w") as csvfile:
						outputwriter = csv.writer(csvfile)
						titles = ['scene','figure','ground'] + feature_keys
						outputwriter.writerow(titles)
						for line in reader[:]:
							if 'scene' not in line:
								if self.scene == line[0] and self.figure == line[1] and self.ground == line[2]:
									# Must ofset by 3 here due to each row beginning with scene and object names
									for x in range(0,len(feature_keys)):
										
										if self.set_of_features[feature_keys[x]] != '?':
											if len(line) > x+3:
												line[x+3] = self.set_of_features[feature_keys[x]]
											else:
												line.append(self.set_of_features[feature_keys[x]])
									
									

								outputwriter.writerow(line)
				except Exception as e:

					print('Writing to CSV Failed')
					print('Figure: ' + self.figure)
					print('Ground:' + self.ground)
					print(e)
			else:
				with open(Relationship.output_path, "a") as csvfile:
					outputwriter = csv.writer(csvfile)
					outputwriter.writerow(row)

	




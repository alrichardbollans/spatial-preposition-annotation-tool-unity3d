## Reads feature value file from unity project and standardises values
## Uses "standard score"
## Outputs in given directory

import pandas as pd
import os


def get_project_directory():
	unity_folder_name = "Unity Projects"

	current_directory = os.getcwd()
	user_home = os.path.expanduser("~")

	if os.path.basename(current_directory) == unity_folder_name:
		return current_directory
	elif os.path.basename(os.path.dirname(current_directory)) == unity_folder_name:
		return os.path.dirname(current_directory)
	else:
		return user_home + '/Dropbox/' + unity_folder_name

class Features:
	## Get path for calculated features
	project_path = get_project_directory()
	project_folder_name = "Data Collection Game"
	feature_data_folder_name = "Scene Data"
	property_path = project_path +"/"+ project_folder_name + "/" + feature_data_folder_name +"/relations.csv"
	output_path = "feature values/standardised_values.csv"
	human_readable_path = "feature values/human_readable_values.csv"
	
	## Features given in relation.csv that aren't needed for analysis
	features_to_remove = ["size_ratio","horizontal_distance_normalized","shortest_distance_normalized","location_control_-x","location_control_-z","location_control_x","location_control_z","figure_volume","ground_volume","horizontal_projection_overlap_proportion"]
	
	def __init__(self):
		self.dataset = pd.read_csv(self.property_path)
		# Remove values where fig=ground
		self.dataset = self.dataset[self.dataset.Figure!=self.dataset.Ground]

		## Calculate location control and append column
		location_control_parts = ["location_control_x","location_control_-x","location_control_-z","location_control_z"]

		self.dataset["location_control"] = self.dataset[location_control_parts].sum(axis=1)/4

		# Remove unwanted columns
		self.dataset = self.dataset.drop(self.features_to_remove,axis=1)
		## Output cleaned dataset
		self.dataset.to_csv(self.human_readable_path,index =False)
		## Calculate means and standard deviations
		## Can be used to convert standardised values back to real values
		self.means = self.dataset.iloc[:,3:].mean()
		self.stds = self.dataset.iloc[:,3:].std()

	def standardise_values(self):
		new_dataframe = self.dataset
		
		new_dataframe.iloc[:,3:] = new_dataframe.iloc[:,3:] - self.means
		new_dataframe.iloc[:,3:] = new_dataframe.iloc[:,3:]/(self.stds)
		
		return new_dataframe

	def write_new(self,dataframe):
		
		dataframe.to_csv(self.output_path,index =False)
	def write_mean_std(self):
		
		self.means.to_csv("feature values/feature_means.csv")
		self.stds.to_csv("feature values/feature_stds.csv")

	def convert_standardised_value_to_normal(self,feature, value):
		mean = self.means[feature]
		std = self.stds[feature]

		x = (value * std) + mean

		return x
	def convert_normal_value_to_standardised(self,feature, value):
		mean = self.means[feature]
		std = self.stds[feature]

		x = (value -mean)/std

		return x

if __name__ == '__main__':
	f= Features()
	nd = f.standardise_values()
	f.write_new(nd)
	f.write_mean_std()

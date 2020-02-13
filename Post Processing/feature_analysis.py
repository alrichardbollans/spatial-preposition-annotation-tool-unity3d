# First run compile_instances.py

#Input: Configuration selection info from compile_instances. Constraints from Comp task
## Looks at how features relate to categorisation and likelihood of selection
## Outputs prototypes and plots
## Can be run after compile_instances

import csv
import pandas as pd  
import numpy as np 
import matplotlib as mpl

import matplotlib.pyplot as plt




## Import validation modules
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression, TheilSenRegressor


from scipy.special import comb
import math




from relationship import *
from preprocess_features import *  
from compile_instances import InstanceCollection
from compile_instances import ComparativeCollection
from classes import *
from process_data import BasicInfo


	
sv_filetag = 'semantic'
comp_filetag = 'comparative'



preposition_list = BasicInfo.preposition_list
## Feature keys are all features
feature_keys = Relationship.get_feature_keys()

## keys are feature keys with some features removed e.g. properties of ground
relation_keys = Relationship.get_relation_keys()
print("# of Relation Features = "+ str(len(relation_keys)))
#To do
### Need to update filenames and plot/look at comparative graphs


def convert_index(x,number_of_columns):
	### Function used to convert index to place in row/columns for plots
	if x==0 or x ==6 or x == 12:
		i = 0
		j=0
	elif x <  6:
		i = x/number_of_columns
		j = x % number_of_columns
	else:
		x= x-6
		i = x/number_of_columns
		j = x % number_of_columns
	
	return [i,j]







class PrepositionModels():
	### Given training scenes, works out models for individual preposition
	ratio_feature_name = InstanceCollection.ratio_feature_name
	categorisation_feature_name = InstanceCollection.categorisation_feature_name
	scene_feature_name = InstanceCollection.scene_feature_name
	fig_feature_name = InstanceCollection.fig_feature_name
	ground_feature_name = InstanceCollection.ground_feature_name
	
	ratio_index = -2
	category_index = -1
	scene_index = 0
	

	interval = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]).reshape(-1,1)
	def __init__(self,preposition,train_scenes, feature_to_remove = None,polyseme = None):
		self.feature_to_remove = feature_to_remove
		# Given polyseme if being used to model polyseme
		self.polyseme = polyseme

		## Use pandas dataframes for ease of importing etc..
		# Created in compile_instances write_config_ratios()
		dataset = pd.read_csv('preposition data/semantic-ratio-list' + preposition + ' .csv')
		
		# self.name = name
		self.preposition = preposition
		## Use pandas dataframes for ease of importing etc..
		# Created in compile_instances write_config_ratios()
		## Row is created in dataset only if the configuration was tested
		# Set of values with selection information
		self.dataset = dataset
		if self.polyseme != None:
			# # Remove none polyseme preposition instances from dataset
			indexes_to_drop =[]
			for index, row in self.dataset.iterrows():
				if polyseme.potential_instance(row[self.scene_feature_name],row[self.fig_feature_name],row[self.ground_feature_name]):
					pass
				elif row[self.categorisation_feature_name] ==0:
					pass
				else:
					indexes_to_drop.append(index)


			self.dataset.drop(self.dataset.index[indexes_to_drop], inplace=True )
			
			
			
		
		# Remove rows from above where not training scene
		self.train_dataset = self.dataset[(self.dataset.iloc[:,self.scene_index].isin(train_scenes))]
		
		## Remove selection info columns and names to only have features
		self.allFeatures = self.remove_nonfeatures(self.train_dataset)
		
		## Feature dataframe for regressions etc.
		self.feature_dataframe = self.remove_nonrelations(self.allFeatures)
		
		
		
		self.features_keys = self.allFeatures.columns
		self.relation_keys = []
		for f in self.features_keys:
			if f not in Relationship.context_features:
				self.relation_keys.append(f)

		# Remove rows from above where not a preposition instance
		self.aff_dataset = self.train_dataset[(self.train_dataset.iloc[:,self.category_index]==1)]
		### Remove seleciton info columns to only have features
		self.affFeatures = self.remove_nonfeatures(self.aff_dataset)
		# Only instances are the best examples -- typical instances
		ratio_max = self.train_dataset[self.ratio_feature_name].max()
		self.typical_dataset = self.train_dataset[(self.train_dataset.iloc[:,self.ratio_index]==ratio_max)]
		### Remove seleciton info columns to only have features
		self.typical_features = self.remove_nonfeatures(self.typical_dataset)
		# Ratio dataset with non-instances - nobody selected the preposition
		self.neg_dataset = self.train_dataset[(self.train_dataset.iloc[:,self.category_index]==0)]
		### Remove seleciton info columns to only have features
		self.neg_features = self.remove_nonfeatures(self.neg_dataset)


		## prototype calculated using regression. Stored as array
		self.prototype = []
		

		self.prototype_csv = "model info/prototypes/"+preposition+".csv"

		## regression weights calculated by linear regression. stored as array and dataframe
		self.poly_regression_model = None
		self.linear_regression_model = None
		
		self.regression_weights = []
		self.regression_weight_csv = "model info/regression weights/"+preposition+".csv"
		self.all_features_regression_weight_csv = "model info/regression weights/allfeatures_"+preposition+".csv"

		## Stores model predictions for later plotting
		self.interval_predictions = dict()

		

		## barycentre_prototype . stored as array
		self.barycentre_prototype = None
		
		self.barycentre_csv = "model info/barycentre model/"+preposition+"-prototype.csv"

		## exemplar_mean . stored as array
		self.exemplar_mean = None

		self.exemplar_csv = "model info/exemplar/"+preposition+"-exemplar_means.csv"
		
		
	def remove_nonfeatures(self,d):
		## Remove seleciton info columns and names to only have features
		return d.drop(["Scene","Figure","Ground",self.ratio_feature_name,self.categorisation_feature_name],axis=1)
	def remove_nonrelations(self,d):
		## Remove features which are for identifying polysemes
		new_d = d.drop(Relationship.context_features,axis=1)
		if self.feature_to_remove != None:
			## Remove features to remove
			new_d = new_d.drop([self.feature_to_remove],axis=1)
		return new_d

	def plot_features_ratio(self,no_columns,axes,feature,X,y_pred,Y):
		# X = Selection Ratio
		# y_pred =  predicted Y values on unit interval
		# Y =  feature value
		index = self.relation_keys.index(feature)
		## Get position to  display, by index
		l = convert_index(index,no_columns)
		ax1 = axes[l[0],l[1]]
		
		
	
		
		

		ax1.set_xlabel("Selection Ratio")
		ylabel = feature
		## Rename some features
		if ylabel == "contact_proportion":
			ylabel = "contact"
		if ylabel == "bbox_overlap_proportion":
			ylabel = "containment"
		
		ax1.set_ylabel(ylabel)
		# ax1.set_xbound(0,1)
		# ax1.set_ybound(0,1)
		# ax1.set_xlim(-0.1,1.1)
		# ax1.set_ylim(-0.1,1.1)
		ax1.grid(True)
		
		# Plot data point scatter
		ax1.plot(X, Y ,'k.')
		# Plot regression line
		ax1.plot(X, y_pred,color='red', linewidth=2)
		# Plot barycentre and exemplar values
		end = [1]
		end = np.array(end).reshape(-1,1)
		
		if self.barycentre_prototype is not None:
			b = self.barycentre_prototype[index]
			b = np.array([b]).reshape(-1,1)
			#Plot barycentre value
			ax1.plot(end,b,markersize=10,markeredgewidth=2,marker='+')
		if self.exemplar_mean is not None:
			ex = self.exemplar_mean[index]
			ex = np.array([ex]).reshape(-1,1)

			
			#Plot exemplar mean value
			ax1.plot(end,ex,markersize=10,markeredgewidth=2, marker=(5, 2))

		# if self.prototype is not None:
		# 	p = self.prototype[index]
		# 	p = np.array([p]).reshape(-1,1)

			
		# 	#Plot exemplar mean value
		# 	ax1.plot(p,end,markersize=10,markeredgewidth=2, marker=(4, 2))

	
	def work_out_models(self):
		self.work_out_linear_regression_model()
		# self.work_out_polynomial_regression_model(3)
		self.work_out_barycentre_prototype()
		self.work_out_exemplar_mean()
		self.work_out_prototype_model()

		

		

	def work_out_barycentre_prototype(self):
		out = []
		X = self.affFeatures
		


		for feature in self.relation_keys:
			pro_value = X[feature].mean()

			out.append(pro_value)
			
		out = np.array(out)
		self.barycentre_prototype = out
		return self.barycentre_prototype
	
	def work_out_exemplar_mean(self):
		out = []
		X = self.typical_features

		for feature in self.relation_keys:
			pro_value = X[feature].mean()
			out.append(pro_value)
			
		out = np.array(out)
		self.exemplar_mean = out
		return self.exemplar_mean
	
	def get_plot_filename(self,file_no):
		x = str(file_no)
		
		if self.polyseme != None:
			filename = self.polyseme.plot_folder  + self.preposition+"-" + self.polyseme.polyseme_name + x +' .pdf'
		else:

			filename = "model info/plots/"+self.preposition + x+ ".pdf"
		return filename
		
	def plot_models(self):
		## Plots simple linear regressions used to find prototypes
		no_rows = 3
		no_columns = 2
		
		fig, axes = plt.subplots(nrows=no_rows, ncols=no_columns, sharex=False, sharey=False)
		fig.tight_layout()
		fig.canvas.set_window_title('Ratio vs. Feature')
		plot_count = 0
		file_no = 1

		for feature in self.relation_keys:
			
			plot_count +=1
			
			r= plot_count % (no_columns * no_rows)
			
			## Reshape data first
			Y = self.train_dataset[feature].values.reshape(-1,1)
			X = self.train_dataset[self.ratio_feature_name].values.reshape(-1,1)
			## Get prediction of all points on interval
			y_pred = self.interval_predictions[feature]

			self.plot_features_ratio(no_columns,axes,feature,X,y_pred,Y)

			filename = self.get_plot_filename(file_no)
			
			## When the figure is full of plots, save figure
			if r == 0:
				
				plt.savefig(filename, bbox_inches='tight')
				file_no +=1

				## Clear plots for new figure
				fig, axes = plt.subplots(nrows=no_rows, ncols=no_columns, sharex=False, sharey=False)
				fig.tight_layout()
				fig.canvas.set_window_title('Ratio vs. Feature')
		## Save remaining plots
		filename = self.get_plot_filename(file_no)
		plt.savefig(filename, bbox_inches='tight')
	def work_out_feature_prototype(self,feature):
		## First predict feature value given selection ratio of 1
		## Reshape data first
		X = self.train_dataset[self.ratio_feature_name].values.reshape(-1,1)
		Y = self.train_dataset[feature].values.reshape(-1,1)
		
		
		model1 = LinearRegression()
		
			
		# Fit model to data
		model1.fit(X,Y)
		y_pred = model1.predict(X)
		self.interval_predictions[feature] = y_pred
		
		# Get prototype for feature
		max_point = np.array([1]).reshape(-1,1)

		feature_prototype = model1.predict(max_point)
		
		pro_value = feature_prototype[0][0]
		return pro_value

	def work_out_linear_regression_model(self):
		## Next get gradient when feature predicts selection ratio
		## Reshape data first
		
		X = self.feature_dataframe
		Y = self.train_dataset[self.ratio_feature_name].values.reshape(-1,1)
		
		
		lin_model = LinearRegression()
		
			
		# Fit model to data
		lin_model.fit(X,Y)

		self.linear_regression_model =lin_model
		# print(self.preposition)
		# print("Linear Score")
		# print(lin_model.score(X,Y))

		return lin_model

		

	def work_out_feature_weights(self):
		if self.linear_regression_model == None:
			model2 = self.work_out_linear_regression_model()
		else:
			model2 = self.linear_regression_model

		
		X = self.feature_dataframe

		v = pd.DataFrame(model2.coef_,index = ["coefficient"]).transpose()
		w = pd.DataFrame(X.columns,columns = ["feature"])
		coeff_df = pd.concat([w,v],axis = 1,join="inner")
		coeff_df =coeff_df.set_index("feature")
		
		
		weights = []
		
		for feature in self.relation_keys:
			if self.feature_to_remove != None:
				## If the feature is removed, append 0 instead
				if feature == self.feature_to_remove:
					weights.append(0)
				else:
					w= abs(coeff_df.loc[feature,"coefficient"])
					weights.append(w)
			else:
				w= abs(coeff_df.loc[feature,"coefficient"])
				weights.append(w)
		weights = np.array(weights)

		self.regression_weights = weights

		
		

	def work_out_prototype_model(self):
		## Work out linear regression on each feature by comparing to the ratio of times selected
		
		### This step gives prototypes for later steps, which are saved to prototypes folder
		
		
		prototype = []
		
		
		for feature in self.relation_keys:
			## First predict feature value given selection ratio of 1
			pro_value = self.work_out_feature_prototype(feature)
			
			# Add to dictionary
			prototype.append(pro_value)
			
		self.work_out_feature_weights()
		prototype = np.array(prototype)
		
		self.prototype = prototype
		
		
		
		return self.prototype

	def work_out_polynomial_regression_model(self,n):
		## Next get gradient when feature predicts selection ratio
		## Reshape data first
		X = self.feature_dataframe
		Y = self.train_dataset[self.ratio_feature_name].values.reshape(-1,1)
		
		polynomial_features= PolynomialFeatures(degree=n)
		x_poly = polynomial_features.fit_transform(X)
		
		model2 = LinearRegression()
		
			
		# Fit model to data
		model2.fit(x_poly,Y)

		self.poly_regression_model =model2
		print(self.preposition)
		print("Polynomial Score" + str(n))
		print(model2.score(x_poly,Y))

		return model2
	def output_models(self):
		## Only called once when training scenes are all scenes, so these are the best model parameters
		wf = pd.DataFrame(self.regression_weights, self.relation_keys)
		
		wf.to_csv(self.regression_weight_csv)

		pf = pd.DataFrame(self.prototype, self.relation_keys)
		
		pf.to_csv(self.prototype_csv)
		

		epf = pd.DataFrame(self.barycentre_prototype, self.relation_keys)
		
		epf.to_csv(self.barycentre_csv)

		exf = pd.DataFrame(self.exemplar_mean, self.relation_keys)
		
		exf.to_csv(self.exemplar_csv)

	def all_feature_weights(self):
		## Calculates regression weights for all features
		weights = []
		
		for feature in feature_keys:		
			
			weight = work_out_feature_weight(feature)
			
			weights.append(weight)
			
		
		weights = np.array(weights)
		
		wf = pd.DataFrame(weights, feature_keys)
		
		wf.to_csv(self.all_features_regression_weight_csv)
		return weights
		
	def read_all_feature_weights(self):
		# Read regression weights for all features
		wf = pd.read_csv(self.all_features_regression_weight_csv, index_col=0)

		return wf
	def read_regression_weights(self):
		# Read regression weights for relations
		wf = pd.read_csv(self.regression_weight_csv, index_col=0)

		return wf



	
	
		
		

	
		
		
	def read_calculations(self,classifier):
		with open("figures/csv_tables"+self.name+classifier+":"+self.preposition+".csv") as csvfile:
			read = csv.reader(csvfile) #.readlines())
			reader = list(read)

			for line in reader:
				if line[0] in self.relation_keys:
					value = line[1]
					setattr(self,classifier +":"+line[0],value)

			

	
	

class Model:
	

	## Puts together preposition models and has various functions for testing
	def __init__(self,name,train_scenes,test_scenes,weight_dict=None,constraint_dict= None,feature_to_remove = None,prototype_dict = None,regression_model_dict = None, regression_dimension = None):
		
		self.name = name
		# Prepositions to test
		self.test_prepositions = preposition_list
		## Dictionary containing constraints to satisfy
		self.constraint_dict = constraint_dict
		## A feature to remove from models to see how results change
		self.feature_to_remove = feature_to_remove
		## Input dictionarys of prototype and feature weights for each preposition, stored as arrays
		## prototype_dict = None => exemplar model
		self.prototype_dict = prototype_dict
		self.weight_dict = weight_dict
		self.test_scenes = test_scenes
		self.train_scenes = train_scenes
		self.regression_model_dict = regression_model_dict
		self.regression_dimension = regression_dimension
		
		
		
		
	
	def semantic_similarity(self,weight_array,x,y):
		#Similarity of x to y
		# x y are 1d arrays
		## Works out the weighted Euclidean distance of x and y and then negative exponential
		## Weights are given in class
		
		
		
		
		## Edit values to calculate without considering feature
		if self.feature_to_remove != None:
			
			i = relation_keys.index(self.feature_to_remove)
			weight_array[i] = 0
		# Subtract arrays point wise 
		point  = np.subtract(x,y)
		# Square pointwise
		point = np.square(point)
		# Dot product pointwise by weights
		summ = np.dot(point,weight_array)

		# Square root to get distance
		distance = math.sqrt(summ)
		## Get typicality
		out = math.exp(-distance)
		return out

	def get_typicality(self,preposition,point):
		## Works out the typicality of the given point (1D array)
		# Point taken as input is from one side of constraint inequality
		if self.regression_model_dict != None:
			
			point_array = np.array(point).reshape(1,-1)
			if self.regression_dimension != None:
				
				# Must transform the point for polynomial regression
				polynomial_features= PolynomialFeatures(degree=self.regression_dimension)
				point_array = polynomial_features.fit_transform(point_array)
			


			t = self.regression_model_dict[preposition].predict(point_array)
			return t
		if self.prototype_dict != None:
			

			prototype_array = self.prototype_dict[preposition]
			weight_array = self.weight_dict[preposition]
			out = self.semantic_similarity(weight_array,point,prototype_array)
			
			
			return out
		if self.prototype_dict == None:
			## When no prototype_dict is given calculate typicality using exemplar model
			## Load exemplars and non instancesfor preposition and remove context features
			p = PrepositionModels(preposition,self.train_scenes)

			
			exemplars = p.typical_features.drop(Relationship.context_features,axis=1)
			none_instances = p.neg_features.drop(Relationship.context_features,axis=1)
			
			# Find average semantic similarity to points in exemplar model
			counter = 0
			semantic_similarity_sum = 0
			
			## Iterate over rows in exemplar dataframe
			for index, row in exemplars.iterrows():
				
				## Get row values
				e = row.values
				
				## Convert values to np array
				e =np.array(e)
				counter += 1
				## Calculate similarity of current point to exemplar
				weight_array = self.weight_dict[preposition]
				semantic_similarity_sum += self.semantic_similarity(weight_array,point,e)
			
			if counter == 0:
				return 0
			else:
				top = float(semantic_similarity_sum)/counter

		

			return top


	def get_score(self):
		### Calculates scores on all constraints for particular model
		### Prototype dict is a dictionary of prototypes (1D arrays) for each preposition
		
		scores = [] # Scores for each preposition to be averaged later
		weight_totals = [] # Total constraint weights for each preposition
		totals = [] # Total number of constraints for each preposition
		average_score = 0
		weighted_average_score = 0
		total_weight_counter = 0
		for preposition in self.test_prepositions:
			
			
			allConstraints = self.constraint_dict[preposition]
			# Constraints to test on
			Constraints = []
			
			for c in allConstraints:
				if c.scene in self.test_scenes:
					Constraints.append(c)
			
			# Constraint info
			weight_counter = 0
			counter =0
			for c in Constraints:
				weight_counter += c.weight
				counter +=1
			total_weight_counter += weight_counter
			weight_totals.append(weight_counter)
			totals.append(counter)
			
			# Get score for preposition
			score_two = self.weighted_score(preposition,Constraints)
			
			
			weighted_average_score += score_two

			score = float(score_two)/weight_counter
			average_score += score
			
				
			# score =  round(score,3)
			scores.append(score)
		
		average_score = float(average_score)/len(self.test_prepositions)
		weighted_average_score = float(weighted_average_score)/total_weight_counter
		# average_score =  round(average_score,3)
		# weighted_average_score =  round(weighted_average_score,3)
		scores.append(average_score)
		scores.append(weighted_average_score)

		self.weight_totals = weight_totals
		self.totals = totals
		self.scores = scores
		
		return scores

	

	## First score based on number of satisfied constraints
	def unweighted_score(self,preposition,Constraints):
		# Calculates how well W and P satisfy the constraints, NOT accounting for constraint weight
		# W and P are 1D arrays
		

		total = 0
		counter = 0
		for c in Constraints:
			total +=1
			
			lhs = self.get_typicality(preposition,c.lhs)
			rhs = self.get_typicality(preposition,c.rhs)
			if c.is_satisfied(lhs,rhs):
				counter +=1
		
		
		return counter

	def weighted_score(self,preposition,Constraints):
		# Calculates how well W and P satisfy the constraints, accounting for constraint weight
		counter = 0
		
		for c in Constraints:
			lhs = self.get_typicality(preposition,c.lhs)
			rhs = self.get_typicality(preposition,c.rhs)
			if c.is_satisfied(lhs,rhs):
				counter +=c.weight
			# else:
			# 	if len(self.test_scenes) == len(self.train_scenes):
			# 		if preposition == "under" and self.name == "Our Prototype":
						
			# 			print("#########")
			# 			print(c.scene)
			# 			print("ground:" + c.ground)
			# 			print("Correct Figure: " + c.f1)
			# 			print("Incorrectly better figure:" + c.f2)
			# 			cp = Comparison(c.scene,preposition,c.ground)
			# 			print("possible_figures:" + str(cp.possible_figures))
					
					
					
					
					
					

		return counter 

class GenerateModels():
	feature_processer = Features()
	lin_model_name ="Linear Regression"
	poly_model_name = "Polynomial Regression"
	our_model_name = "Our Prototype"
	exemplar_model_name = "Exemplar"
	cs_model_name ="Conceptual Space"
	proximity_model_name = "Proximity"
	best_guess_model_name = "Best Guess"
	simple_model_name ="Simple"

	## List of all model names
	model_name_list = [our_model_name,exemplar_model_name,cs_model_name,best_guess_model_name,simple_model_name,proximity_model_name]
	
	## List of model names except ours
	other_name_list = [exemplar_model_name,cs_model_name,best_guess_model_name,simple_model_name,proximity_model_name]
	
	#Generating models to test
	def __init__(self,train_scenes,test_scenes,constraint_dict,feature_to_remove = None,only_test_our_model = None):
		# Dictionary of constraints to satisfy
		self.constraint_dict = constraint_dict
		
		# Values of prototypes and feature weights
		self.prototypes = dict()
		self.barycentre_prototypes = dict()
		self.all_regression_weights = dict()
		self.poly_regression_model_dict = dict()
		self.linear_regression_model_dict = dict()
		# Scenes used to train models
		self.train_scenes = train_scenes
		# Scenes used to test models
		self.test_scenes = test_scenes
		self.feature_to_remove = feature_to_remove


		## Get data models
		for p in preposition_list:
			M = PrepositionModels(p,self.train_scenes,feature_to_remove = self.feature_to_remove)
			M.work_out_models()
			
			self.prototypes[p] = M.prototype
			self.barycentre_prototypes[p] = M.barycentre_prototype
			self.all_regression_weights[p] = M.regression_weights
			self.linear_regression_model_dict[p] = M.linear_regression_model
			self.poly_regression_model_dict[p] = M.poly_regression_model

		m = Model(self.our_model_name,self.train_scenes,self.test_scenes,weight_dict=self.all_regression_weights,constraint_dict=self.constraint_dict,feature_to_remove =  self.feature_to_remove,prototype_dict = self.prototypes)
		# linear_r_model = Model(self.lin_model_name,self.all_regression_weights,self.train_scenes,self.test_scenes,self.constraint_dict,regression_model_dict = self.linear_regression_model_dict)
		# poly_r_model = Model(self.poly_model_name,self.all_regression_weights,self.train_scenes,self.test_scenes,self.constraint_dict,regression_model_dict = self.poly_regression_model_dict, regression_dimension = 3)
		if only_test_our_model == None:#feature_to_remove == None:
			
			# Only include others if not testing features
			m1 = Model(self.exemplar_model_name,self.train_scenes,self.test_scenes,weight_dict=self.all_regression_weights,constraint_dict=self.constraint_dict,feature_to_remove =self.feature_to_remove)
			m2 = Model(self.cs_model_name,self.train_scenes,self.test_scenes,weight_dict=self.all_regression_weights,constraint_dict=self.constraint_dict,feature_to_remove =  self.feature_to_remove,prototype_dict = self.barycentre_prototypes)
			m3 = self.get_proximity_model()
			m4 = self.get_simple_model()
			m5 = self.get_best_guess_model()
			models = [m,m1,m2,m3,m4,m5]#,linear_r_model,poly_r_model]
			
		else:
			
			models = [m]

		self.models = models
	
	def get_proximity_model(self):
		## 
		pro_dict = dict()
		weight_dict = dict()
		
		pro_array = []
		weight_array = []
		for feature in relation_keys:
			if feature != "shortest_distance":
				pro_array.append(0)
				weight_array.append(0)
			else:
				x = self.feature_processer.convert_normal_value_to_standardised(feature, 0)
				pro_array.append(x)
				weight_array.append(1)

		pro_array = np.array(pro_array)
		weight_array = np.array(weight_array)
		for preposition in preposition_list:


			pro_dict[preposition] = pro_array
			weight_dict[preposition] = weight_array
			
		m = Model(self.proximity_model_name,self.train_scenes,self.test_scenes,weight_dict=weight_dict,constraint_dict=self.constraint_dict,feature_to_remove =self.feature_to_remove,prototype_dict =pro_dict)
		return m

	def get_best_guess_model(self):
		## best guess model uses intuition
		pro_dict = dict()
		weight_dict = dict()

		for preposition in preposition_list:
			pro_array = []
			weight_array = []
			if preposition == "inside":
				for feature in relation_keys:
					if feature != "bbox_overlap_proportion":
						pro_array.append(0)
						weight_array.append(0)
					else:
						x = self.feature_processer.convert_normal_value_to_standardised(feature, 1)
						pro_array.append(x)
						weight_array.append(1)
				
			if preposition == "in":
				for feature in relation_keys:
					salient_features = ["bbox_overlap_proportion","location_control"]
					if feature not in salient_features:
						pro_array.append(0)
						weight_array.append(0)
					else:
						x = self.feature_processer.convert_normal_value_to_standardised(feature, 1)
						pro_array.append(x)
						weight_array.append(1)
				
			if preposition == "on":
				salient_features = ["above_proportion","contact_proportion","support"]
				for feature in relation_keys:
					if feature not in salient_features:
						pro_array.append(0)
						weight_array.append(0)
					else:
						x = self.feature_processer.convert_normal_value_to_standardised(feature, 1)
						pro_array.append(x)
						weight_array.append(1)
			if preposition == "on top of":
				salient_features = ["above_proportion","contact_proportion"]
				for feature in relation_keys:
					if feature not in salient_features:
						pro_array.append(0)
						weight_array.append(0)
					else:
						x = self.feature_processer.convert_normal_value_to_standardised(feature, 1)
						pro_array.append(x)
						weight_array.append(1)
				
			if preposition == "above":
				salient_features = ["above_proportion","horizontal_distance"]
				for feature in relation_keys:
					if feature not in salient_features:
						pro_array.append(0)
						weight_array.append(0)
					else:
						if feature == "above_proportion":
							v =1
						else:
							v= 0
						x = self.feature_processer.convert_normal_value_to_standardised(feature, v)
						pro_array.append(x)
						weight_array.append(1)
				
				
			if preposition == "over":
				salient_features = ["above_proportion","f_covers_g"]
				for feature in relation_keys:
					if feature in salient_features:
						x = self.feature_processer.convert_normal_value_to_standardised(feature, 1)
						pro_array.append(x)
						weight_array.append(1)
					else:
						

						pro_array.append(0)
						weight_array.append(0)
				
				
			if preposition == "below":
				salient_features = ["below_proportion","horizontal_distance"]
				for feature in relation_keys:
					if feature not in salient_features:
						pro_array.append(0)
						weight_array.append(0)
					else:
						if feature == "below_proportion":
							v =1
						else:
							v= 0
						x = self.feature_processer.convert_normal_value_to_standardised(feature, v)
						pro_array.append(x)
						weight_array.append(1)
				
				
			if preposition == "under":
				salient_features = ["below_proportion","g_covers_f"]
				for feature in relation_keys:
					if feature in salient_features:
						x = self.feature_processer.convert_normal_value_to_standardised(feature, 1)
						pro_array.append(x)
						weight_array.append(1)

						
					else:
						pro_array.append(0)
						weight_array.append(0)
				
				
			if preposition == "against":
				salient_features = ["contact_proportion","horizontal_distance","location_control"]
				for feature in relation_keys:
					if feature not in salient_features:
						pro_array.append(0)
						weight_array.append(0)
					else:
						if feature == "contact_proportion":
							v =1
						elif feature == "horizontal_distance":
							v= 0
						else:
							v=0.5
						x = self.feature_processer.convert_normal_value_to_standardised(feature, v)
						pro_array.append(x)
						weight_array.append(1)
				
			pro_array = np.array(pro_array)
			weight_array = np.array(weight_array)
			pro_dict[preposition]= pro_array
			weight_dict[preposition]= weight_array
		m = Model(self.best_guess_model_name,self.train_scenes,self.test_scenes,weight_dict=weight_dict,constraint_dict=self.constraint_dict,feature_to_remove =self.feature_to_remove,prototype_dict = pro_dict)
		return m

		

	def get_simple_model(self):
		## Simple model uses simple geometric features which are equally weighted
		
		pro_dict = dict()
		weight_dict = dict()

		for preposition in preposition_list:
			pro_array = []
			weight_array = []
			if preposition == "inside" or preposition == "in":
				for feature in relation_keys:
					if feature != "bbox_overlap_proportion":
						pro_array.append(0)
						weight_array.append(0)
					else:
						x = self.feature_processer.convert_normal_value_to_standardised(feature, 1)
						pro_array.append(x)
						weight_array.append(1)
				
				
			if preposition == "on top of" or preposition == "on":
				salient_features = ["above_proportion","contact_proportion"]
				for feature in relation_keys:
					if feature not in salient_features:
						pro_array.append(0)
						weight_array.append(0)
					else:
						x = self.feature_processer.convert_normal_value_to_standardised(feature, 1)
						pro_array.append(x)
						weight_array.append(1)
				
			if preposition == "above" or preposition == "over":
				salient_features = ["above_proportion","horizontal_distance"]
				for feature in relation_keys:
					if feature not in salient_features:
						pro_array.append(0)
						weight_array.append(0)
					else:
						if feature == "above_proportion":
							v =1
						else:
							v= 0
						x = self.feature_processer.convert_normal_value_to_standardised(feature, v)
						pro_array.append(x)
						weight_array.append(1)
				
				
			
				
				
			if preposition == "below" or preposition == "under":
				salient_features = ["below_proportion","horizontal_distance"]
				for feature in relation_keys:
					if feature not in salient_features:
						pro_array.append(0)
						weight_array.append(0)
					else:
						if feature == "below_proportion":
							v =1
						else:
							v= 0
						x = self.feature_processer.convert_normal_value_to_standardised(feature, v)
						pro_array.append(x)
						weight_array.append(1)
				
				
			if preposition == "against":
				salient_features = ["contact_proportion","horizontal_distance"]
				for feature in relation_keys:
					if feature not in salient_features:
						pro_array.append(0)
						weight_array.append(0)
					else:
						if feature == "contact_proportion":
							v =1
						if feature == "horizontal_distance":
							v= 0
						
						x = self.feature_processer.convert_normal_value_to_standardised(feature, v)
						pro_array.append(x)
						weight_array.append(1)
				
			pro_array = np.array(pro_array)
			weight_array = np.array(weight_array)
			pro_dict[preposition]= pro_array
			weight_dict[preposition]= weight_array
		m = Model(self.simple_model_name,self.train_scenes,self.test_scenes,weight_dict=weight_dict,constraint_dict=self.constraint_dict,feature_to_remove =self.feature_to_remove,prototype_dict = pro_dict)
		return m

class TestModels():
	# Takes input set of models and outputs database of scores
	
	
	def __init__(self,models,version_name):
		
		self.version_name = version_name
		self.models = models
		self.model_name_list = []
		
		out = dict()

		
		for model in self.models:
			self.model_name_list.append(model.name)
			model.get_score()
			out[model.name] = model.scores
		
		
		# out["Total Constraint Weights"] = models[0].weight_totals + ["",""]

		df = pd.DataFrame(out,self.models[0].test_prepositions + ["Average", "Overall"])
		

		# Reorder columns
	 	new_column_order = self.model_name_list
	 	reordered_df = df[new_column_order]

		self.score_dataframe = reordered_df
		
		

	
		

class MultipleRuns:
	# This class carries out multiple runs of model tests and outputs the results
	# Number of runs must be specified as well as either test_size for standard repeated sampling
	# or k for repeated k-fold sampling
	def __init__(self,constraint_dict,number_runs=None,test_size= None,k=None,compare = None,features_to_test = None):

		self.constraint_dict = constraint_dict
		self.number_runs = number_runs
 		self.test_size= test_size
 		self.k=k
 		self.compare = compare
 		self.features_to_test = features_to_test

		self.scene_list = BasicInfo.get_scene_list()

		
		self.run_count = 0
		# Dictionary of dataframes giving scores. Indexed by removed features.
	 	self.dataframe_dict = dict()
 		self.all_csv = "scores/tables/all-model scores.csv"
 		self.all_plot = "scores/plots/ScoresUsingAllData.pdf"
 		if self.features_to_test == None:

	 		self.scores_tables_folder = "scores/tables/all features"
	 		self.scores_plots_folder = "scores/plots/all features"
	 	else:
	 		self.scores_tables_folder = "scores/tables/removed features"
	 		self.scores_plots_folder = "scores/plots/removed features"
 		if self.test_size !=None:
 			self.file_tag = "rss" + str(self.test_size)
 			self.average_plot_title = "Scores Using RRSS Validation"
 		if self.k != None:
 			self.file_tag = str(self.k) +"fold"
 			self.average_plot_title = "Scores Using Repeated K-Fold Validation. K = "+str(self.k) + " N = " + str(self.number_runs)
		
 			
	 		self.average_plot_pdf = self.scores_plots_folder +"/average" + self.file_tag+".pdf"
			self.average_csv = self.scores_tables_folder + "/averagemodel scores "+self.file_tag+".csv"
			self.comparison_csv = self.scores_tables_folder + "/repeatedcomparisons "+self.file_tag+".csv"
		if self.features_to_test != None:
			self.feature_removed_average_csv = dict()
			for feature in self.features_to_test:
				self.feature_removed_average_csv[feature] = self.scores_tables_folder + "/averagemodel scores "+self.file_tag+" "+feature +"removed.csv"
 		self.Generate_Models_all_scenes = self.generate_models(self.scene_list,self.scene_list)
 		self.test_prepositions = self.Generate_Models_all_scenes.models[0].test_prepositions
 		self.prepare_comparison_dicts()

		if self.features_to_test != None:
			for feature in self.features_to_test:
				self.count_without_feature_better[feature] = dict()
		 		self.count_with_feature_better[feature] = dict()
				for p in self.test_prepositions + ["Average", "Overall"]:
					self.count_without_feature_better[feature][p] = 0
			 		self.count_with_feature_better[feature][p] = 0
 		# following lists help confirm all scenes get used for both training and testing
	 	self.scenes_used_for_testing = []
	 	self.scenes_used_for_training = []
	 	
	def prepare_comparison_dicts(self):
 		## Counts to compare models
 		self.count_our_model_wins = dict()
 		self.count_other_model_wins = dict()
 		## Counts to compare features
 		self.count_without_feature_better = dict()
 		self.count_with_feature_better = dict()
 		
 		# Prepare dict
		for other_model in self.Generate_Models_all_scenes.other_name_list:
			
			self.count_our_model_wins[other_model] = 0
			self.count_other_model_wins[other_model] = 0
 	
	def generate_models(self,train_scenes,test_scenes):
		if self.features_to_test != None:
			# Test model with no removed features
			generate_models = GenerateModels(train_scenes,test_scenes,self.constraint_dict,only_test_our_model = True)
			
		else:
			# Test all models with no removed features
			generate_models = GenerateModels(train_scenes,test_scenes,self.constraint_dict)
		return generate_models

	def test_all_scenes(self):
 		generate_models = self.Generate_Models_all_scenes
 		models = generate_models.models
	 	t= TestModels(models, "all")
	 	self.all_dataframe = t.score_dataframe
	 	
	 	
		
		self.all_dataframe.to_csv(self.all_csv)

		self.plot_dataframe_bar_chart(self.all_dataframe,self.all_plot,"Preposition","Score","Scores Using All Data")
	
	def single_validation_test(self,train_scenes,test_scenes):
		generate_models = self.generate_models(train_scenes,test_scenes)
			
		t= TestModels(generate_models.models,str(self.run_count))
		# Get generated scores
		dataset = t.score_dataframe
		

		## Add scores to total
		if "all_features" in self.dataframe_dict:
			self.dataframe_dict["all_features"] = self.dataframe_dict["all_features"].add(dataset)
			
		else:
			self.dataframe_dict["all_features"] = dataset

		## Get our score from dataframe
		our_score = dataset.at["Overall",generate_models.our_model_name]

		## Compare Models
		if self.compare != None:
			for other_model in generate_models.other_name_list:
				
				# Get score
				other_score = dataset.at["Overall",other_model]
				# Update counts
				if our_score > other_score:
					self.count_our_model_wins[other_model] += 1
		
				if other_score > our_score:
					self.count_other_model_wins[other_model] +=1
			

		
		# Add scores to dataframe
		if self.features_to_test != None:
			
			for feature in self.features_to_test:
				generate_models = GenerateModels(train_scenes,test_scenes,self.constraint_dict,feature_to_remove =feature,only_test_our_model = True)
				t= TestModels(generate_models.models,str(self.run_count))

				feature_dataset = t.score_dataframe
				# feature_dataset = feature_dataset.drop(["Total Constraint Weights"],axis=1)
				
				for p in self.test_prepositions + ["Average", "Overall"]:
					without_feature_score = feature_dataset.at[p,generate_models.our_model_name]
					with_feature_score = dataset.at[p,generate_models.our_model_name]

					
					if without_feature_score > with_feature_score:
						
				 		self.count_without_feature_better[feature][p] += 1
					if with_feature_score > without_feature_score:
						self.count_with_feature_better[feature][p] += 1
				
				# Add to totals
				if feature in self.dataframe_dict:
					self.dataframe_dict[feature] = self.dataframe_dict[feature].add(feature_dataset)
					
				else:
					self.dataframe_dict[feature] = feature_dataset
					
	def get_validation_scene_split(self):
 		
 		# Get train-test scenes
 		if self.test_size != None:
			train_scenes , test_scenes = train_test_split(self.scene_list,test_size=self.test_size)

			## Update scene lists
			for sc in train_scenes:
				if sc not in self.scenes_used_for_training:
					self.scenes_used_for_training.append(sc)

			for sc in test_scenes:
				if sc not in self.scenes_used_for_testing:
					self.scenes_used_for_testing.append(sc)
			return [train_scenes,test_scenes]
		if self.k != None:
			# Create random folds for testing
			folds = []
			
			
			scenes_left = self.scene_list
			divisor = self.k
			while divisor > 1:
				t_size = float(1)/divisor
				train_scenes , test_scenes = train_test_split(scenes_left,test_size=t_size)
				folds.append(test_scenes)
				scenes_left = train_scenes
				divisor = divisor-1
				if divisor ==1:
					folds.append(train_scenes)
				
			# for c in (range(self.k-1)):
			# 	print("c")
			# 	print(c)
			# 	train_scenes , test_scenes = train_test_split(scenes_left,test_size=t_size)
			# 	folds.append(test_scenes)
				
			# 	scenes_left = train_scenes
			# 	new_k = self.k-c-1
			# 	if new_k>1:
			# 		t_size = float(1)/(new_k)
			# 	else:
			# 		folds.append(train_scenes)
			print("folds")
			for f in folds:
				print(len(f))
			return folds
			

		
	def folds_check(self,folds):
		# Check all folds have some constraints to test
		for f in folds:
			for preposition in self.test_prepositions:
				
				
				allConstraints = self.constraint_dict[preposition]
				
				Constraints = []
				
				for c in allConstraints:
					if c.scene in f:
						Constraints.append(c)
				if len(Constraints) == 0:
					return False
		return True
		
	def validation(self):
 	
	 	## Perform Repeated random sub-sampling validation
	 	## Either using k-fold or standard method
	 	for i in range(self.number_runs):
	 		self.run_count = i

			print("Run Number:" + str(i))
	 		
	 		if self.test_size != None:
	 			split = self.get_validation_scene_split()
	 			train_scenes = split[0]
	 			test_scenes =split[1]
	 			self.single_validation_test(train_scenes,test_scenes)
	 		if self.k != None:
		 		## This handles the case where test_scenes do not produce any constraints
		 		while True:
		 			folds = self.get_validation_scene_split()

		 			if self.folds_check(folds):
		 				for f in folds:
				 			test_scenes = f
				 			train_scenes = []
				 			for s in self.scene_list:
				 				if s not in test_scenes:
				 					train_scenes.append(s)
				 			self.single_validation_test(train_scenes,test_scenes)
			 			break
			 		else:
			 			print("Fold with no constraints to test. Retrying...")
			 		
		 		
		# First update value of number of runs to account for folds
		if self.k != None:
			self.total_number_runs = self.number_runs * self.k
		else:
			self.total_number_runs = self.number_runs
		## Output comparison of models and p-value
		if self.compare != None:
			other_model_p_value = dict()
			for other_model in self.Generate_Models_all_scenes.other_name_list:
				
				p_value = calculate_p_value(self.total_number_runs,self.count_our_model_wins[other_model])
				other_model_p_value[other_model] = p_value

			### Create dataframes to output
			p_value_df = pd.DataFrame(other_model_p_value,["p_value"])
			our_model_win_count = pd.DataFrame(self.count_our_model_wins,["Our model wins"])
			other_model_win_count = pd.DataFrame(self.count_other_model_wins,["Other model wins"])
			### Append dataframes into one
			new_df = p_value_df.append([our_model_win_count,other_model_win_count], sort=False)
			self.comparison_df = new_df
			

		if self.features_to_test != None:
			feature_p_value = dict()
			with_feature_better = dict()
			without_feature_better = dict()
			for feature in self.features_to_test:
				for p in self.test_prepositions + ["Average", "Overall"]:
					p_value = calculate_p_value(self.total_number_runs,self.count_with_feature_better[feature][p])
					feature_p_value[feature + ":" + p] = p_value
					with_feature_better[feature + ":" + p] = self.count_with_feature_better[feature][p]
					without_feature_better[feature + ":" + p] = self.count_without_feature_better[feature][p]

			### Create dataframes to output
			p_value_df = pd.DataFrame(feature_p_value,["p_value"])
			win_count = pd.DataFrame(with_feature_better,["With feature wins"])
			lose_count = pd.DataFrame(without_feature_better,["Without feature wins"])
			### Append dataframes into one
			new_df = p_value_df.append([win_count,lose_count], sort=False)
			self.feature_comparison_df = new_df
			

		
		# Print some info
		print("Total Runs:" + str(self.total_number_runs))
		
		if self.test_size != None:
			print("# Scenes used for testing")
			print(len(self.scenes_used_for_testing))
			print("# Scenes used for training")
			print(len(self.scenes_used_for_training))

		# Finalise by averaging scores in dataframe
		for key in self.dataframe_dict:
			self.dataframe_dict[key] = self.dataframe_dict[key].div(self.total_number_runs)
			
				
 		
	
	def output(self):
		# Handle outputting here so we're not always outputting
		self.average_dataframe = self.dataframe_dict["all_features"]
		# Reorder columns for output
		if self.features_to_test == None:
			new_column_order = self.Generate_Models_all_scenes.model_name_list
			reordered_df = self.average_dataframe[new_column_order]
			reordered_df.to_csv(self.average_csv)
		else:
			self.average_dataframe.to_csv(self.average_csv)

		self.plot_dataframe_bar_chart(self.average_dataframe,self.average_plot_pdf,"Preposition","Score",self.average_plot_title)
		if self.compare != None:

			# Output to csv
			self.comparison_df.to_csv(self.comparison_csv)
		if self.features_to_test != None:

			# Output to csv
			self.feature_comparison_df.to_csv(self.comparison_csv)
			for feature in self.features_to_test:
				dff = self.dataframe_dict[feature]
				
				dff.to_csv(self.feature_removed_average_csv[feature])

			out = dict()
			
			for feature in self.features_to_test:
				print(self.dataframe_dict[feature])
				out[feature] = self.dataframe_dict[feature][self.Generate_Models_all_scenes.our_model_name]
			out["None removed"] = self.average_dataframe[self.Generate_Models_all_scenes.our_model_name]
			df = pd.DataFrame(out,self.test_prepositions + ["Average", "Overall"])
			df.to_csv(self.scores_tables_folder+"/functional_feature_analysis.csv")
			

			output_file = self.scores_plots_folder+"/ScoresWithRemovedFeatures.pdf"
			x_label = "Preposition"
			y_label = "Score"
			plot_title = "Average Scores With Removed Features. K = "+str(self.k) + " N = " + str(self.number_runs)
			self.plot_dataframe_bar_chart(df,output_file,x_label,y_label,plot_title)
			# ax = df.plot(kind='bar', title ="Average Scores With Removed Features. K = "+str(self.k) + " N = " + str(self.number_runs),figsize=(15, 10), legend=True)
			
			# ax.set_xlabel("Preposition")
			# ax.set_ylabel("Score")
			# ax.set_yticks(np.arange(0, 1.05, 0.05))
			# ax.grid(True)
			# ax.set_axisbelow(True)
			
			# plt.legend(loc='upper center', bbox_to_anchor=(0.44, -0.35), ncol=3)
			
			# plt.savefig(self.scores_plots_folder+"/ScoresWithRemovedFeatures.pdf", bbox_inches='tight')
		
		



	def plot_dataframe_bar_chart(self,dataset,file_to_save,x_label,y_label,plot_title):
		if self.features_to_test == None:
			new_column_order = self.Generate_Models_all_scenes.model_name_list
			reordered_df = dataset[new_column_order]
		else:
			reordered_df = dataset

		

		ax = reordered_df.plot(kind='bar', width=0.85,title =plot_title,figsize=(20, 10), legend=True)
		
		ax.set_xlabel(x_label, labelpad=10)
		ax.set_ylabel(y_label)
		ax.set_yticks(np.arange(0, 1.01, 0.05))
		ax.set_ylim([0,1])
		ax.set_title(plot_title, pad=10)
		ax.grid(True)
		ax.set_axisbelow(True)
		
		plt.legend(loc='upper center', bbox_to_anchor=(0.44, -0.42), ncol=3)

		
		# plt.show()
		plt.savefig(file_to_save, bbox_inches='tight')
	


	
	def plot_bar_from_csv(self,file,file_to_save,x_label,y_label,plot_title,columns_to_drop = None):
		
		dataset = pd.read_csv(file, index_col=0)
		if columns_to_drop != None:
			dataset = dataset.drop(columns_to_drop,axis=1)
		self.plot_dataframe_bar_chart(dataset,file_to_save,x_label,y_label,plot_title)



def plot_preposition_graphs():
	scene_list = BasicInfo.get_scene_list()
	
	
	for p in preposition_list:
		M = PrepositionModels(p,scene_list)
		M.work_out_models()
		M.output_models()
		M.plot_models()


	
def calculate_p_value(N,x):
		total = 0
		for i in range(x,N+1):
			
			v = comb(N,i)*(math.pow(0.5,N))

			total += v
		return total

def test_features():
	functional_features = ["location_control","support"]
	m = MultipleRuns(constraint_dict,number_runs=100,k =2,features_to_test = functional_features)
	print("Test Features")
	m.validation()
	m.output()

def initial_test():
	m = MultipleRuns(constraint_dict)
	print("Test on all scenes")
	m.test_all_scenes()

def test_models():
	m = MultipleRuns(constraint_dict,number_runs=100,k =2,compare = "y")
	print("Test Model k = 2")
	m.validation()
	m.output()

	m = MultipleRuns(constraint_dict,number_runs=100,k =3,compare = "y")
	print("Test Model k = 3")
	m.validation()
	m.output()

def plot_all_csv():
	m = MultipleRuns(constraint_dict)
	file = m.all_csv
	out_file = m.all_plot

	# self.plot_dataframe_bar_chart(self.all_dataframe,self.all_plot,"Preposition","Score","Scores Using All Data")
	m.plot_bar_from_csv(file,out_file,"Preposition","Score","Scores Using All Data")
def plot_kfold_csv(k):
	m = MultipleRuns(constraint_dict,number_runs=100,k=k)
	file = m.average_csv
	out_file = m.average_plot_pdf

		
	m.plot_bar_from_csv(file,out_file,"Preposition","Score",m.average_plot_title)
	


def plot_feature_csv(k):
	functional_features = ["location_control","support"]
	m = MultipleRuns(constraint_dict,number_runs=100,k =k,features_to_test = functional_features)
	file = m.scores_tables_folder+"/functional_feature_analysis.csv"
	output_file = m.scores_plots_folder+"/ScoresWithRemovedFeatures.pdf"
	x_label = "Preposition"
	y_label = "Score"
	plot_title = "Average Scores With Removed Features. K = "+str(m.k) + " N = " + str(m.number_runs)
	

	m.plot_bar_from_csv(file,output_file,x_label,y_label,plot_title)


def main(constraint_dict):
	
	# plot_preposition_graphs()
	# Edit plot settings
	mpl.rcParams['font.size'] = 40
	mpl.rcParams['legend.fontsize'] = 37
	mpl.rcParams['axes.titlesize'] = 'medium'
	mpl.rcParams['axes.labelsize'] = 'medium'
	mpl.rcParams['ytick.labelsize'] = 'small'
	
	initial_test()
	test_models()
	test_features()

	# plot_all_csv()
	# plot_kfold_csv(2)
	# plot_feature_csv(2)
	
	
	
if __name__ == '__main__':
	name = "n"#raw_input("Generate new constraints? y/n  ")
	if name == "y":
		compcollection = ComparativeCollection()
		constraint_dict = compcollection.get_constraints()
	elif name == "n":
		constraint_dict = Constraint.read_from_csv()
	else:
		print("Error unrecognized input")

	scene_list = BasicInfo.get_scene_list()
	
	


	
	main(constraint_dict)
	
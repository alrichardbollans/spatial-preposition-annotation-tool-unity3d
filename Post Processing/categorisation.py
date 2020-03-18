from sklearn.neural_network import MLPClassifier

from feature_analysis import PrepositionModels, GenerateModels, Model
# from process_data import *



class GenerateCModels(GenerateModels):

	def __init__(self):
		GenerateModels.__init__(self)

class CatPrepositionModels(PrepositionModels):
	def __init__(self,preposition,train_scenes, feature_to_remove = None,polyseme = None):
		PrepositionModels.__init__(self,preposition,train_scenes, feature_to_remove = feature_to_remove,polyseme = polyseme)

		print(self.dataset)
		self.test_dataset = self.dataset[(~self.dataset.iloc[:,self.scene_index].isin(train_scenes))]
		print(self.test_dataset)

	def perceptron_model(self):
		# Next get gradient when feature predicts selection ratio
		# Reshape data first
		
		X = self.allFeatures
		Y = self.train_dataset[self.categorisation_feature_name].values.reshape(-1,1)
		
		
		clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
				
		# Fit model to data
		clf.fit(X,Y)

		self.nn_model =clf
		

		return clf

	 

class CatModel(Model):
	def __init__(self,name,weight_dict,train_scenes,test_scenes,constraint_dict=None,feature_to_remove = None,prototype_dict = None,regression_model_dict = None, regression_dimension = None,perceptron_model_dict = None):
		# Model provides method for assessing typicality
		# CatModel will provide methods for assessing categorisation
		Model.__init__(self,name,train_scenes,test_scenes,constraint_dict=constraint_dict,feature_to_remove = feature_to_remove,prototype_dict = prototype_dict,regression_model_dict = regression_model_dict, regression_dimension = regression_dimension)
		self.perceptron_model_dict = perceptron_model_dict
		
	def get_categorisation(preposition,config,threshold=None):
		

		
		c1_array = np.array(config.row)
		if self.perceptron_model_dict == None:
			
			t_value = self.get_typicality(preposition,c1_array)
			if t_value > threshold:
				cat = 1
				
			else:
				cat = 0
				
		else:
			cat = self.perceptron_model_dict[preposition].predict(c1_array)[0]

		return cat

	def get_cat_score(self,preposition):
		count = 0
		CPM = CatPrepositionModels(preposition,self.train_scenes)
		for index, row in CPM.test_dataset.iterrows():
			row[CPM.categorisation_feature_name]

	def get_catgn_user_agreement(self,preposition,threshold,user):
		scores = dict() # Scores for each preposition to be averaged later
		for preposition in preposition_list:
			scores[preposition] = 0
		
		for a1 in user1_annotations:
			if a1.task in BasicInfo.semantic_abbreviations:
				shared_annotations +=1

				c1 = Configuration(a1.scene,a1.figure,a1.ground)

				# Only use values that are not from 'context' features
				c1_array = np.array(c1.relations_row)
				t_value = self.get_typicality(preposition,c1_array)
				if t_value > threshold:
					cat = 1
					y2+=1
				else:
					cat = 0
					n2 +=1
				
				if preposition in a1.prepositions:
					y1 +=1
				else:
					n1 +=1
				
				
				if preposition in a1.prepositions and t_value > threshold:
					agreements +=1
					
				elif preposition not in a1.prepositions and t_value <= threshold:
					agreements +=1
		if shared_annotations !=0:
			
			expected_agreement = float((y1*y2 + n1*n2))/float((shared_annotations)**2)
		else:
			expected_agreement = 0

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
		
		
		return scores
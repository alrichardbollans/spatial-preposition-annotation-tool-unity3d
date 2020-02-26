

import math

from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import cdist, minkowski
from sklearn.feature_selection import RFE

from scipy.cluster.hierarchy import linkage, cut_tree, fcluster, dendrogram

import matplotlib.pyplot as plt

from feature_analysis import *
from classes import *
from process_data import BasicInfo


all_preposition_list = BasicInfo.preposition_list
polysemous_preposition_list = ['in', 'against','on', 'under',  'over' ] # list of prepositions which exist in the data
non_polysemous_prepositions = ["inside", "above", "below","on top of"]
base_polysemy_folder = "polysemy/"
polyseme_data_folder = base_polysemy_folder+'polyseme data/'
cluster_data_folder = base_polysemy_folder+'clustering/'
kmeans_folder = cluster_data_folder + 'kmeans/'
hry_folder = cluster_data_folder + 'hry/'
score_folder= base_polysemy_folder+'scores/'

class Cluster():
	def __init__(self,preposition,instances,label,alg_typ = None):
		self.preposition = preposition
		self.label = label
		self.instances = instances
		self.mean_series = self.instances.mean()
		# Convert series to appropriate dataframe
		self.means = pd.DataFrame({'values':self.mean_series.values})
		self.means = self.means.set_index(self.mean_series.index)
		self.means = self.means.transpose()
		if alg_typ == "kmeans":
			base_folder = kmeans_folder
		elif alg_typ == "hry":
			base_folder = hry_folder
		else:
			print("Error: No cluster type given")

		
		self.mean_csv = base_folder+"cluster means/clusters-"+preposition+str(self.label)+".csv"
		self.instances_csv = base_folder+"cluster instances/instances-"+preposition+str(self.label)+".csv"

		self.hr_means_csv = base_folder +"cluster means/human readable/instances-"+preposition+str(self.label)+".csv"
		
	def output(self):
		
		# self.mean_series["cardinality"] = len(self.instances.index)
		self.means.to_csv(self.mean_csv)
		self.instances.to_csv(self.instances_csv)
		feature_processer = Features()
		self.hr_means = feature_processer.convert_standard_df_to_normal(self.means)
		self.hr_means.to_csv(self.hr_means_csv)
		



class Clustering():
	all_scenes = BasicInfo.get_scene_list()
	# Number of clusters created by inspecting dendograms?
	cluster_numbers = {'on':3,'in':2,'against':3,'under':2,'over':2}
	# Thresholds for dendograms
	# color_threshold = {'on':0.75,'in':0.75,'against':0.75,'under':0.78,'over':0.75}
	# # Thresholds for assigning clusters. I think they need to be different in order to account for metric?
	# cluster_threshold = {'on':0.75,'in':0.75,'against':0.75,'under':0.8,'over':0.75}
	def __init__(self,preposition):
		self.preposition = preposition
		self.models = PrepositionModels(preposition,self.all_scenes)

		# All selected instances
		self.possible_instances = self.models.affFeatures
		self.possible_instances_relations = self.models.remove_nonrelations(self.possible_instances)
		# Dataset containing 'good' instances
		self.good_dataset = self.models.train_dataset[(self.models.train_dataset.iloc[:,self.models.ratio_index]>=0.5)]
		# Reindex df for later readability
		self.good_dataset = self.good_dataset.reset_index(drop=True)
		# All 'good' instances
		self.good_instances = self.models.remove_nonfeatures(self.good_dataset)

		self.good_instance_relations = self.models.remove_nonrelations(self.good_instances)

		self.typical_instances = self.models.typical_features
		
		# self.models.all_feature_weights()
		# self.feature_weights= self.models.read_all_feature_weights()
		self.relation_weights = self.models.read_regression_weights()
		
		self.cluster_centres_csv = kmeans_folder+"cluster centres/clusters-"+preposition+".csv"
		self.dendrogram_pdf = hry_folder+"figures/dendrogram/dendrogram-"+preposition+".pdf"
		self.elbow_pdf = kmeans_folder+"figures/elbow/"+preposition+".pdf"
		
		# The dataframe we use for clustering
		self.instances_to_cluster = self.good_instance_relations.copy()
		self.km_instances_to_cluster = self.possible_instances_relations.copy()
		## Samples are weighted by selection ratio
		self.sample_weights = self.models.aff_dataset[self.models.ratio_feature_name]#self.good_dataset[self.models.ratio_feature_name]
		## Output good instances to read
		self.good_instance_csv = cluster_data_folder + "good preposition instances/good instances - " + self.preposition+".csv"
		self.instances_to_cluster.to_csv(self.good_instance_csv)

		feature_processer = Features()
		self.hr_good_instance_csv = cluster_data_folder + "good preposition instances/human readable/good instances - " + self.preposition+".csv"
		
		self.hr_good_instances = feature_processer.convert_standard_df_to_normal(self.instances_to_cluster)

		self.hr_good_instances.to_csv(self.hr_good_instance_csv)
		

	
	def custom_metric(self,u,v):
		# weighted euclidean distance. Also weight by instances somehow?
		return minkowski(u, v, p=2,w=self.relation_weights.values)
	def DBScan_cluster(self):
		# To be improved by varying eps
		# Check if any typical instances are  being labelled as noise
		# If so, eps += 0.1 and run again
		# Else, output
		print(self.instances)
		print(self.sample_weights)
		clustering = DBSCAN(eps=3,min_samples=2,metric=self.custom_metric).fit(self.instances,sample_weight=self.sample_weights)
		print(clustering.labels_)
		print(clustering.components_)

	def work_out_hierarchy_model(self):
		
		instances = self.instances_to_cluster
		Z = linkage(instances, method='single',optimal_ordering=True)#,metric=self.custom_metric
		
		fig = plt.figure()
		fig.canvas.set_window_title(self.preposition)
		fig.suptitle("Dendrogram for '"+ self.preposition+"'", fontsize=20)
		
		# factor = self.color_threshold[self.preposition]
		if self.preposition== "on":
			thresh = 0.65*max(Z[:,2])
		else:
			thresh = 0.7*max(Z[:,2]) # Default threshold
		dendrogram(Z, color_threshold= thresh)
		
		
		plt.savefig(self.dendrogram_pdf, bbox_inches='tight')
		
		
		# Form flat clusters based on threshold
		clusters = fcluster(Z, criterion='distance', t=thresh)
		
		done_clusters = []
		for c in clusters:
			if c not in done_clusters:
				done_clusters.append(c)
				cluster_instances_index = []
				for i in range(len(clusters)):
					if c==clusters[i]:
						cluster_instances_index.append(i)
				cluster_instances = instances.iloc[cluster_instances_index,:]
				cluster = Cluster(self.preposition,cluster_instances,c, alg_typ = "hry")
				cluster.output()
		print("Number of clusters: " + str(len(done_clusters)))
		# if self.preposition == "on":
		# 	print(instances.iloc[25,:])
		# 	print(instances.iloc[26,:])
		# 	print("####")
		# 	print(instances.iloc[20,:])
		# 	print(instances.iloc[21,:])
		# 	print(instances.iloc[12,:])
		# 	plt.show()




	def work_out_kmeans_model(self,k):
		
		number_clusters = k

		# nparray = nparray.reshape(-1,1)
		
		km = KMeans(
			n_clusters=number_clusters,
			
		)
		km.fit(self.km_instances_to_cluster,sample_weight=self.sample_weights)

		return km
	

	def output_cluster_info(self,km):
		out = dict()

		for i in range(len(km.cluster_centers_)):
			out["cluster_"+str(i)] = km.cluster_centers_[i]
		


		


		df = pd.DataFrame(out,relation_keys)
		print(self.preposition)
		print(df)

		df.to_csv(self.cluster_centres_csv)

		
		k = self.cluster_numbers[self.preposition]
		for i in range(0,k):
			instances = self.km_instances_to_cluster[km.labels_ == i]
			cluster = Cluster(self.preposition,instances,i,alg_typ ="kmeans")
			cluster.output()
	
	def output_expected_kmeans_model(self):
		k = self.cluster_numbers[self.preposition]
		kmodel = self.work_out_kmeans_model(k)
		
		self.output_cluster_info(kmodel)
	def elbow(self):

		distortions = []
		inertias = []
		K = range(1,10)
		for k in K:
		    kmeanModel = self.work_out_kmeans_model(k)			    
		    # distortions.append(sum(np.min(cdist(self.km_instances_to_cluster, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / self.km_instances_to_cluster.shape[0])
		    inertias.append(kmeanModel.inertia_)

		# Plot the elbow
		plt.plot(K, inertias, 'bx-')
		plt.xlabel('k')
		plt.ylabel('Inertia')
		plt.title('The Elbow Method showing the optimal k - ' + self.preposition)
		

		plt.savefig(self.elbow_pdf, bbox_inches='tight')
		plt.close()
	

	def check_inertia_calculation(self):
		# Checks inertia calculation agrees with KMeans method
		km = self.work_out_kmeans_model(4)
		print("Our inertia")
		i = self.calculate_inertia_from_centres(km.cluster_centers_)
		print(i)
		print("proper inertia")
		print(km.inertia_)



	def distance_to_centre_squared(self,point,centre):
		# Subtract arrays point wise 
		point  = np.subtract(point,centre)
		# Square pointwise
		point = np.square(point)
		# Dot product pointwise by weights
		summ = np.sum(point)
		

		# Square root to get distance
		distance = math.sqrt(summ)

		d2 = distance * distance

		return d2

	def calculate_inertia_from_centres(self,centres):
		total_sum = 0
		# As per the kmeans source code, sample weights are scaled so average weight is 1
		weight_scaling_factor = len(self.sample_weights)/self.sample_weights.sum()

		for index, row in self.km_instances_to_cluster.iterrows():
			
			sample = row.values
			distance = -1
			closest_centre = 0
			for centre in centres:
				d = self.distance_to_centre_squared(sample,centre)
				if distance == -1:
					distance = d
					closest_centre = centre
				elif d<distance:
					distance = d
					closest_centre = centre

			weight = self.sample_weights[index]
			normalised_weight = weight * weight_scaling_factor
			weighted_distance = distance * normalised_weight
			total_sum += weighted_distance

		return total_sum
	def calculate_polysemes_inertia(self,polysemes):
		init = []
		centres = []
		for polyseme in polysemes:
			df = pd.read_csv(polyseme.mean_csv, index_col=0,names=["feature","value"])
			
			centres.append(df["value"].values)

		i = self.calculate_inertia_from_centres(centres)
		print(self.preposition)
		print("Number of clusters:" + str(len(centres)))
		return i
		
			
			


	def plot_elbow_polyseme_inertia(self):
		generated_polyseme_models = GeneratePolysemeModels(Clustering.all_scenes,Clustering.all_scenes,constraint_dict,preserve_rank=True)
		d = generated_polyseme_models.non_shared_dict
		
		polysemes = d[self.preposition]
		polysemes_inertia = self.calculate_polysemes_inertia(polysemes)

		inertias = []
		K = range(1,10)
		for k in K:
		    kmeanModel = self.work_out_kmeans_model(k)			    
		    
		    inertias.append(kmeanModel.inertia_)

		fig, axes = plt.subplots()
		
		
		
		axes.plot([len(polysemes)],[polysemes_inertia],markersize=10,markeredgewidth=2,marker=(5, 2))
		
		

		# Plot the elbow
		axes.plot(K, inertias, 'bx-')
		
		# plt.annotate('This is awesome!', 
		#              xy=(len(polysemes), polysemes_inertia),  
		#              xycoords='data',
		#              textcoords='offset points',
		#              arrowprops=dict(arrowstyle="->"))
		axes.annotate('Polysemy Inertia', xy=(len(polysemes), polysemes_inertia),  xycoords='data',
		            xytext=(len(polysemes)-1, polysemes_inertia+10),
		            
		            horizontalalignment='left', verticalalignment='bottom',
		            )
		axes.set_xlabel('k')
		axes.set_ylabel('Inertia')
		axes.set_title('Inertia from Kmeans Clusters - ' + self.preposition, pad=10)
		
		
		
		plt.savefig(self.elbow_pdf, bbox_inches='tight')
		# plt.close()
		
	

class Polyseme():
	
	def __init__(self,preposition,polyseme_name,train_scenes,eq_feature_dict =  None,greater_feature_dict =  None,less_feature_dict =  None,share_prototype =False):
		self.polyseme_name = polyseme_name
		self.preposition = preposition
		self.train_scenes = train_scenes

		self.current_scenes_property_path = None
		
		# Dictionary containing distinguishing features and their values
		self.eq_feature_dict= eq_feature_dict
		self.greater_feature_dict= greater_feature_dict
		self.less_feature_dict= less_feature_dict

		self.annotation_csv = polyseme_data_folder+'annotations/' + self.preposition+"-" +self.polyseme_name + ' .csv'
		self.prototype_csv = polyseme_data_folder+'prototypes/' + self.preposition+"-" + self.polyseme_name + ' .csv'
		self.mean_csv = polyseme_data_folder+'means/' + self.preposition+"-" + self.polyseme_name + ' .csv'

		self.regression_weights_csv = polyseme_data_folder+'regression weights/' + self.preposition+"-" + self.polyseme_name + ' .csv'
		self.plot_folder = polyseme_data_folder+'plots/'
		

		self.share_prototype = share_prototype
		self.preposition_models = PrepositionModels(self.preposition,self.train_scenes,polyseme = self)
		
		# Assign a rank/hierarchy to polysemes
		# Set rank to be sum of selection ratios (or maybe it should be average?)
		self.rank = self.get_rank()
		self.number_of_instances = self.get_number_of_instances()
		# if self.number_of_instances == 0:
		# 	print("No training instances for: " + self.polyseme_name+"-"+ self.preposition)
		# Add here that only do next bits if rank>0. Then later assign weight and prototype. This will save a lot of computing
		
			
		self.preposition_models.work_out_prototype_model()
		self.get_prototype_and_weight()
	
		
		

	def potential_instance(self,scene,figure,ground, current_scenes_property_path = None):
		#boolean checks whether the configuration could be an instance
		
		r = Relationship(scene,figure,ground)

		r.load_from_csv(current_scenes_property_path)
		if self.eq_feature_dict != None:
			for feature in self.eq_feature_dict:
				value = round(r.set_of_features[feature],6)
				condition = round(self.eq_feature_dict[feature],6)
				if value != condition:
					return False

		if self.greater_feature_dict != None:
			for feature in self.greater_feature_dict:
				
				if r.set_of_features[feature] < self.greater_feature_dict[feature]:
					return False
		if self.less_feature_dict != None:
			for feature in self.less_feature_dict:
				if r.set_of_features[feature] > self.less_feature_dict[feature]:
					return False
		return True
		
		
	def get_prototype_and_weight(self):
		
		
		

		self.weights = self.preposition_models.regression_weights
		self.prototype = self.preposition_models.prototype
		# print("unshared")
		# print(self.prototype)
		## Input dictionarys of prototype and feature weights for each preposition, stored as arrays
		if self.share_prototype:
			preposition_models = PrepositionModels(self.preposition,self.train_scenes)
			preposition_models.work_out_prototype_model()
			self.prototype = preposition_models.prototype
			# print("shared")
			# print(self.prototype)
	def get_number_of_instances(self):
		return len(self.preposition_models.aff_dataset.index)
	def get_rank(self):
		ratio_feature_name = self.preposition_models.ratio_feature_name
		mean = self.preposition_models.aff_dataset.mean(axis=0)[ratio_feature_name]
		
		self.rank = mean
		if np.isnan(self.rank):
			self.rank=0
			
		return self.rank
		
		
	def plot(self):

		self.preposition_models.plot_models()

	def output_prototype_weight(self):
		pf = pd.DataFrame(self.prototype, relation_keys)
		
		pf.to_csv(self.prototype_csv)

		wf = pd.DataFrame(self.weights, relation_keys)
		
		wf.to_csv(self.regression_weights_csv)

	def output_definition(self):
		out = dict()
		out["eq_feature_dict"] = []
		out["greater_feature_dict"] = []
		out["less_feature_dict"] = []
		for feature in relation_keys:

			if self.eq_feature_dict != None:
				if self.eq_feature_dict.has_key(feature):
				
					out["eq_feature_dict"].append(round(self.eq_feature_dict[feature],6))
				else:
					out["eq_feature_dict"].append("None")
			else:
				out["eq_feature_dict"].append("None")

			if self.greater_feature_dict != None:
				if self.greater_feature_dict.has_key(feature):
				
					out["greater_feature_dict"].append(round(self.greater_feature_dict[feature],6))
				else:
					out["greater_feature_dict"].append("None")
			else:
				out["greater_feature_dict"].append("None")

			if self.less_feature_dict != None:
				if self.less_feature_dict.has_key(feature):
				
					out["less_feature_dict"].append(round(self.less_feature_dict[feature],6))
				else:
					out["less_feature_dict"].append("None")
			else:
				out["less_feature_dict"].append("None")

		wf = pd.DataFrame(out,relation_keys)#["equality", "greater than", "less than"])

		wf.to_csv(polyseme_data_folder+"/definitions/"+self.preposition+"-"+self.polyseme_name+".csv")


	
class Polyseme_Model(Model):
	### Note we should also test the idea that polysemes affect*weight* rather than prototype. 
	#Intuitively this seems better to me.

	# Need to incorporate train_test in polyseme creation
	# Also do better with sharing prototytpes
	
	## Puts together preposition models and has various functions for testing
	def __init__(self,name,train_scenes,test_scenes,constraint_dict,polyseme_dict= None):
		Model.__init__(self,name,train_scenes,test_scenes,constraint_dict=constraint_dict)
		
		
		# Dictionary of polyseme instances for each preposition
		self.polyseme_dict = polyseme_dict
		self.test_prepositions = polysemous_preposition_list
		
		
		
		
	def get_possible_polysemes(self,preposition,scene,figure,ground, current_scenes_property_path =None):
		out = []
		if self.polyseme_dict != None:

			for polyseme in self.polyseme_dict[preposition]:
				if polyseme.potential_instance(scene,figure,ground,current_scenes_property_path):
					out.append(polyseme)
		return out

	
	def get_typicality(self,preposition,point,scene,figure,ground, current_scenes_property_path =None):
		## Works out the typicality of the given point (1D array)
		# Point taken as input is from one side of constraint inequality

		out = 0
		pps = self.get_possible_polysemes(preposition,scene,figure,ground,current_scenes_property_path)
		if len(pps) == 0:
			print("Error: No polyseme given for:")
			print(preposition)
			print(scene)
			print(figure)
			print(ground)

		for polyseme in pps:
					

			prototype_array = polyseme.prototype
			weight_array = polyseme.weights
			new = self.semantic_similarity(weight_array,point,prototype_array)
			
			new = new * (math.pow(polyseme.rank,2))
			if new > out:
				out = new
				

		
		return out
	def weighted_score(self,preposition,Constraints):
		# Calculates how well W and P satisfy the constraints, accounting for constraint weight
		counter = 0
		
		for c in Constraints:
			lhs = self.get_typicality(preposition,c.lhs,c.scene,c.f1,c.ground)
			rhs = self.get_typicality(preposition,c.rhs,c.scene,c.f2,c.ground)
			if c.is_satisfied(lhs,rhs):
				counter +=c.weight		

		return counter 
	
	def output_typicalities(self,preposition,input_feature_list):
		output_csv = base_polysemy_folder+ "instance typicalities/"+self.name+"-typicality_test-"+preposition+".csv"
		dfObj = pd.DataFrame(columns=['scene', 'figure', 'ground',self.name])
		geom_relations = Relationship.load_all(path=input_feature_list)
		geom_relations.pop(0)
		for relation in geom_relations:
			scene = relation[0]
			figure = relation[1]
			ground = relation[2]
			
			c = Configuration(scene,figure,ground,path=input_feature_list)
			
		
			# Typicality is calculated for each configuration
			# To check whether a configuration fits a particular polyseme we need to include
			value_array = np.array(c.relations_row)
			typicality = self.get_typicality(preposition,value_array,c.scene,c.figure,c.ground,current_scenes_property_path= input_feature_list)
			dfObj = dfObj.append({'scene': c.scene, 'figure': c.figure, 'ground': c.ground,'self.name': typicality}, ignore_index=True)

		
		
		dfObj.to_csv(output_csv)

		
class SalientFeature():

	def __init__(self,feature,value,gorl):
		self.feature = feature
		self.value = value
		self.gorl = gorl

class GeneratePolysemeModels():
	feature_processer = Features()

	our_model_name = "Distinct Prototype"
	
	other_model_name = "Shared Prototype"
	baseline_model_name = "Baseline Model"

	## List of all model names
	model_name_list = [our_model_name,other_model_name,baseline_model_name]
	
	## List of model names except ours
	other_name_list = [other_model_name,baseline_model_name]

	def __init__(self,train_scenes,test_scenes,constraint_dict = None, preserve_rank= False):
		# Dictionary of constraints to satisfy
		self.constraint_dict = constraint_dict
		# Variable set to true if want to generate polysemes and not edit the rank
		self.preserve_rank = preserve_rank
		
		# Scenes used to train models
		self.train_scenes = train_scenes
		# Scenes used to test models
		self.test_scenes = test_scenes

		self.baseline_model_dict = self.get_general_cases()
		self.baseline_model = Polyseme_Model("Baseline Model",self.train_scenes,self.test_scenes,self.constraint_dict,polyseme_dict= self.baseline_model_dict)

		self.non_shared_dict = self.get_non_shared_prototype_polyseme_dict()
		self.non_shared = Polyseme_Model("Distinct Prototype",self.train_scenes,self.test_scenes,self.constraint_dict,polyseme_dict= self.non_shared_dict)
		

		self.shared_dict = self.get_shared_prototype_polyseme_dict()
		self.shared = Polyseme_Model("Shared Prototype",self.train_scenes,self.test_scenes,self.constraint_dict,polyseme_dict= self.shared_dict)
	
	
		self.models = [self.non_shared,self.shared,self.baseline_model]
	
		
		
	def output_polyseme_info(self):
		d = self.non_shared_dict
		
		for preposition in d:
			out = dict()
			print("Outputting:" + preposition)
			for polyseme in d[preposition]:
				polyseme.output_prototype_weight()
				polyseme.output_definition()
				polyseme.plot()

				polyseme.preposition_models.aff_dataset.to_csv(polyseme.annotation_csv)
				polyseme.preposition_models.affRelations.mean().to_csv(polyseme.mean_csv)
				
				out[preposition + "-" + polyseme.polyseme_name] = [polyseme.get_number_of_instances(),polyseme.rank]
			
			number_df = pd.DataFrame(out,["Number", "Rank"])
			number_df.to_csv(polyseme_data_folder + "/ranks/"+preposition+ " -ranks.csv")

	def get_general_cases(self):
		d = dict()
		for preposition in all_preposition_list:
			d[preposition] = [Polyseme(preposition,"general_case",self.train_scenes)]
		return d


	def get_shared_prototype_polyseme_dict(self):
		out = dict()
		old_dict = self.non_shared_dict

		for preposition in old_dict:
			out[preposition] = []
			for polyseme in old_dict[preposition]:
				new_pol = Polyseme(polyseme.preposition,polyseme.polyseme_name,self.train_scenes,share_prototype = True,greater_feature_dict=polyseme.greater_feature_dict,eq_feature_dict=polyseme.eq_feature_dict,less_feature_dict=polyseme.less_feature_dict)
				
				out[preposition].append(new_pol)

		return out

	
	def generate_polysemes(self,preposition,salient_features):
		## Generates polysemes based on ideal meaning discusion
		## Give salient features and their threshold values
		polysemes = []
		
		g_dict = dict()
		l_dict = dict()
		for f in salient_features:
			if f.gorl == "l":
				l_dict[f.feature] = f.value
			else:
				g_dict[f.feature] = f.value
			


		## Canon
		
		p1 = Polyseme(preposition,"canon",self.train_scenes,greater_feature_dict=g_dict,less_feature_dict=l_dict)
		polysemes.append(p1)

		## Nearly canon
		x= len(salient_features) - 1
		while x>=0:
			
			name_count = 0
			for pair in list(itertools.combinations(salient_features, x)):
				name_count +=1 
				g_feature_dict = dict()
				l_feature_dict = dict()
				
					
				for f in salient_features:
					
					if f not in pair:
						
						if f.gorl == "l":
							g_feature_dict[f.feature] = f.value
						else:
							l_feature_dict[f.feature] = f.value
					if f in pair:
						
						if f.gorl == "l":
							l_feature_dict[f.feature] = f.value
						else:
							g_feature_dict[f.feature] = f.value
				if x==0:
					p_name = "far" + str(name_count)
				elif x== len(salient_features) - 1:
					p_name = "near" + str(name_count)
				else:
					p_name = "not far" + str(name_count)
				ply = Polyseme(preposition,p_name,self.train_scenes,greater_feature_dict=g_feature_dict,less_feature_dict=l_feature_dict)
				polysemes.append(ply)
			x = x - 1
		return polysemes 


	def get_non_shared_prototype_polyseme_dict(self):
		out = dict()
		

		## Non-polysemous prepositions
		# general_cases = self.get_general_cases()
		# out["inside"] = general_cases["inside"]
		
		# out["above"] = general_cases["above"]
		# out["below"] = general_cases["below"]
		# out["on top of"] = general_cases["on top of"]

		## greater and lessthan dictionaries are greater/less than OR EQUAL to
		# On
		
		contact03 = self.feature_processer.convert_normal_value_to_standardised("contact_proportion", 0.3)
		above01 = self.feature_processer.convert_normal_value_to_standardised("above_proportion", 0.1)
		above09 = self.feature_processer.convert_normal_value_to_standardised("above_proportion", 0.9)
		above099 = self.feature_processer.convert_normal_value_to_standardised("above_proportion", 0.99)
		above07 = self.feature_processer.convert_normal_value_to_standardised("above_proportion", 0.7)
		sup01 = self.feature_processer.convert_normal_value_to_standardised("support", 0.1)
		sup09 = self.feature_processer.convert_normal_value_to_standardised("support", 0.9)
		gv0 = self.feature_processer.convert_normal_value_to_standardised("ground_verticality", 0)
		gv1 = self.feature_processer.convert_normal_value_to_standardised("ground_verticality", 1)
		b0 = self.feature_processer.convert_normal_value_to_standardised("bbox_overlap_proportion", 0)
		b09 = self.feature_processer.convert_normal_value_to_standardised("bbox_overlap_proportion", 0.9)
		lc09 = self.feature_processer.convert_normal_value_to_standardised("location_control", 0.9)
		b07 = self.feature_processer.convert_normal_value_to_standardised("bbox_overlap_proportion", 0.7)
		lc075 = self.feature_processer.convert_normal_value_to_standardised("location_control", 0.75)
		gf09= self.feature_processer.convert_normal_value_to_standardised("g_covers_f", 0.9)
		bl09= self.feature_processer.convert_normal_value_to_standardised("below_proportion", 0.9)
		bl08= self.feature_processer.convert_normal_value_to_standardised("below_proportion", 0.8)
		fg09= self.feature_processer.convert_normal_value_to_standardised("f_covers_g", 0.9)
		hd01= self.feature_processer.convert_normal_value_to_standardised("horizontal_distance", 0.1)

		# print(self.feature_processer.convert_standardised_value_to_normal("location_control", 1.22))
		# print(self.feature_processer.convert_standardised_value_to_normal("location_control", 1.11))
		# On
		
		f1 = SalientFeature("above_proportion",above09,"g")
		f2 = SalientFeature("support",sup09,"g")
		f3 = SalientFeature("contact_proportion",contact03,"g")
		on_salient_features = [f1,f2,f3]
		out["on"] = self.generate_polysemes("on",on_salient_features)
		# out["on top of"] = self.generate_polysemes("on top of",on_salient_features)
		# In
		f1 = SalientFeature("bbox_overlap_proportion",b07,"g")
		f2 = SalientFeature("location_control",lc075,"g")
		
		in_salient_features = [f1,f2]
		
		out["in"] = self.generate_polysemes("in",in_salient_features)

		# # Inside
		# f1 = SalientFeature("bbox_overlap_proportion",b09,"g")
		
		
		# inside_salient_features = [f1]
		
		# out["inside"] = self.generate_polysemes("inside",inside_salient_features)

		# Under
		f1 = SalientFeature("g_covers_f",gf09,"g")
		f2 = SalientFeature("below_proportion",bl09,"g")
		
		under_salient_features = [f1,f2]
		
		out["under"] = self.generate_polysemes("under",under_salient_features)

		# # Below
		# f1 = SalientFeature("horizontal_distance",hd01,"l")
		# f2 = SalientFeature("below_proportion",bl08,"g")
		
		# below_salient_features = [f1,f2]
		
		# out["below"] = self.generate_polysemes("below",below_salient_features)
		

		# Over
		f1 = SalientFeature("f_covers_g",fg09,"g")
		f2 = SalientFeature("above_proportion",above07,"g")
		
		over_salient_features = [f1,f2]
		
		out["over"] = self.generate_polysemes("over",over_salient_features)
		
		# # above
		# f1 = SalientFeature("horizontal_distance",hd01,"l")
		# f2 = SalientFeature("above_proportion",above099,"g")
		
		# above_salient_features = [f1,f2]
		
		# out["above"] = self.generate_polysemes("above",above_salient_features)

		# Against
		f1 = SalientFeature("contact_proportion",contact03,"g")
		f2 = SalientFeature("support",sup01,"g")
		f3 = SalientFeature("above_proportion",above01,"l")
		# f5 = SalientFeature("horizontal_distance",hd01,"l")
		# f4 = SalientFeature("location_control",lc08,"g")
		
		against_salient_features = [f1,f2,f3]
		
		out["against"] = self.generate_polysemes("against",against_salient_features)
		
		general_cases = self.baseline_model_dict
		
		if self.preserve_rank:
			pass
		else:
			for prep in out:
				# print(prep)
				for poly in out[prep]:
					
					if poly.number_of_instances == 0:
						
						# In the case there are no training instances (rank=0)
						# Set the general parameters
						new_p = general_cases[prep][0]
						poly.weights = new_p.weights
						poly.prototype = new_p.prototype
						
						poly.rank = new_p.rank
						

		

		return out
		
		

class MultipleRunsPolysemyModels(MultipleRuns):
	def __init__(self,constraint_dict,number_runs=None,test_size= None,k=None,compare = None,features_to_test = None):
		MultipleRuns.__init__(self,constraint_dict,number_runs=number_runs,test_size= None,k=k,compare = compare,features_to_test = None)

 		self.all_csv = "polysemy/"+self.all_csv
 		self.all_plot = "polysemy/" + self.all_plot
 		
 		self.scores_tables_folder = "polysemy/"+"scores/tables"
 		self.scores_plots_folder = "polysemy/"+"scores/plots"
	 	
 		
 		if self.k != None:
 			self.file_tag = str(self.k) +"fold"
 			self.average_plot_title = "Scores Using Repeated K-Fold Validation. K = "+str(self.k) + " N = " + str(self.number_runs)
		
 			
	 		self.average_plot_pdf = self.scores_plots_folder +"/average" + self.file_tag+".pdf"
			self.average_csv = self.scores_tables_folder + "/averagemodel scores "+self.file_tag+".csv"
			self.comparison_csv = self.scores_tables_folder + "/repeatedcomparisons "+self.file_tag+".csv"
	
	
	def generate_models(self,train_scenes,test_scenes):
		generated_polyseme_models = GeneratePolysemeModels(train_scenes,test_scenes,self.constraint_dict)
		return generated_polyseme_models

	
def output_all_polyseme_info():
	generated_polyseme_models = GeneratePolysemeModels(Clustering.all_scenes,Clustering.all_scenes,constraint_dict,preserve_rank=True)
	generated_polyseme_models.output_polyseme_info()
	
def test_on_all_scenes():
	generated_polyseme_models = GeneratePolysemeModels(Clustering.all_scenes,Clustering.all_scenes,constraint_dict)

	p_models= generated_polyseme_models.models

	t = TestModels(p_models,"all")
	all_dataframe = t.score_dataframe.copy()
	
	# all_dataframe =all_dataframe.drop(non_polysemous_prepositions)

	all_dataframe.to_csv(score_folder+"all_test.csv")
	print(all_dataframe)
def test_model(runs,k):
	m = MultipleRunsPolysemyModels(constraint_dict,number_runs=runs,k =k,compare = "y")
	print("Test Model k = "+ str(k))
	m.validation()
	m.output()
	print(m.average_dataframe)

def test_models():
	mpl.rcParams['font.size'] = 40
	mpl.rcParams['legend.fontsize'] = 37
	mpl.rcParams['axes.titlesize'] = 'medium'
	mpl.rcParams['axes.labelsize'] = 'medium'
	mpl.rcParams['ytick.labelsize'] = 'small'

	# m = MultipleRunsPolysemyModels(constraint_dict,number_runs=100,k =2,compare = "y")
	# print("Test Model k = 2")
	# m.validation()
	# m.output()

	# m = MultipleRunsPolysemyModels(constraint_dict,number_runs=100,k =3,compare = "y")
	# print("Test Model k = 3")
	# m.validation()
	# m.output()
	test_on_all_scenes()
	# test_model(2,2)
	# test_model(10,10)

def output_typicality():
	generated_polyseme_models = GeneratePolysemeModels(Clustering.all_scenes,Clustering.all_scenes)
	
	model= generated_polyseme_models.non_shared
	for preposition in all_preposition_list:
		# Using Features.output_path as the path is the default anyway?
		model.output_typicalities(preposition,Features.output_path)
def work_out_all_kmeans_clusters():
	for preposition in polysemous_preposition_list:
		c = Clustering(preposition)
		# km = c.work_out_kmeans_model(3)
		# c.output_clusters(km)
		c.elbow()
def compare_kmeans():
	for preposition in polysemous_preposition_list:
		c = Clustering(preposition)
		
		c.plot_elbow_polyseme_inertia()
	
def work_out_all_dbsccan_clusters():
	for preposition in polysemous_preposition_list:
		print(preposition)
		c = Clustering(preposition)
		km = c.DBScan_cluster()
def work_out_all_hry_clusters():
	print("Working out hry clusters")
	for preposition in polysemous_preposition_list:
		print(preposition)
		c = Clustering(preposition)
		km = c.work_out_hierarchy_model()
def work_out_kmeans_clusters():
	print("Working out kmeans clusters")
	for preposition in polysemous_preposition_list:
		print(preposition)
		c = Clustering(preposition)
		c.output_expected_kmeans_model()

def main(constraint_dict):
	# work_out_kmeans_clusters()
	# compare_kmeans()
	# work_out_all_kmeans_clusters()
	work_out_all_hry_clusters()
	# output_all_polyseme_info()
	
	# output_typicality()
	# test_models()

if __name__ == '__main__':
	

	name = "n"#raw_input("Generate new constraints? y/n  ")
	if name == "y":
		compcollection = ComparativeCollection()
		constraint_dict = compcollection.get_constraints()
	elif name == "n":
		constraint_dict = Constraint.read_from_csv()
	else:
		print("Error unrecognized input")

	main(constraint_dict)
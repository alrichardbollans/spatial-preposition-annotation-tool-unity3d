## Outputs commonsense features to a csv file
## Update relationship class in classes.cs to add to feature list proper

import requests
# import nltk
import csv

# from nltk.corpus import wordnet



# import sys

import os

def clean_name(object_name):
		if '(' in object_name:
			clean_name = object_name[:object_name.find(".")]
		elif '_' in object_name:
			clean_name = object_name[:object_name.find("_")]
		else: 
			clean_name = object_name
		return clean_name.lower()

def extract_relation_weight(relation,obj1,obj2):
	object1_name = clean_name(obj1)
	object2_name = clean_name(obj2)
	tag = '/r/'+relation
	
	conceptlinks = requests.get('http://api.conceptnet.io/query?node=/c/en/'+object1_name+'&other=/c/en/'+ object2_name).json()
	if len(conceptlinks['edges']) == 0:
		return 0
	elif not any(edges['rel']['@id'] == tag for edges in conceptlinks['edges']):
		return 0
	else:	
		for edges in conceptlinks['edges']:
			if edges['rel']['@id'] == tag:
				return edges['weight']


def extract_path_similarity(obj1,obj2):
	object1_name = clean_name(obj1) + '.n.01'
	object2_name = clean_name(obj2) + '.n.01'

	x = wordnet.synset(object1_name)
	y = wordnet.synset(object2_name)

	return x.path_similarity(y)

with open('commonsense properties.csv', "w") as csvfile:
	outputwriter = csv.writer(csvfile)
	outputwriter.writerow(['Object','CN_ISA_CONTAINER','CN_UsedFor_Light'])


	with open('scene_info.csv', "r") as csvfile:
		reader = csv.reader(csvfile)
		datalist = list(reader)
		line = datalist[2]

		for obj in line:
			try:
				cont = extract_relation_weight('IsA',clean_name(obj),'container')
				light = extract_relation_weight('UsedFor',clean_name(obj),'light')
				if cont >=1:
					cont=1
				if light >=1:
					light =1
				outputwriter.writerow([obj,cont,light])
			except:
				print('Commonsense property error')
				print(clean_name(obj))
# scene_list = scenes.scene_list
# for obj in scenes.distinct_bodies:
# 	CN_ISA_CONTAINER = extract_isa_weight(obj,'container')
# 	CN_RelatedTo_CONTAINER = extract_relatedto_weight(obj,'container')
# 	WNPathSimilarity_Container = extract_path_similarity(obj,'container')
# 	with open('commonsense properties.csv', "a") as csvfile:
# 	    outputwriter = csv.writer(csvfile)
# 	    outputwriter.writerow([obj,CN_ISA_CONTAINER,CN_RelatedTo_CONTAINER,WNPathSimilarity_Container])

# print(wordnet.synset('mug.n.01').definition())

### Testing

# concept = requests.get('http://api.conceptnet.io/query?node=/c/en/mug&other=/c/en/container').json()

# # print(concept['edges'])


# for edges in concept['edges']:
# 	if edges['rel']['@id'] == '/r/IsA':
# 		print(edges['weight'])

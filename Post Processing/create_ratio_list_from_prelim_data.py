import pandas as pd
from preprocess_features import *

dataset = pd.read_csv('collected data/preliminary study/clean semantic annotation list.csv', index_col=0)
dataset = dataset.drop(["Task","Cam Location","Cam Rotation","Time"],axis=1)



def config_match(row1,row2):
	scene1 = row1["Scene"]
	scene2 = row2["Scene"]
	fig1 = row1["Figure"]
	fig2 = row2["Figure"]
	g1 = row1["Ground"]
	g2 = row2["Ground"]
	if scene1 == scene2 and fig1 == fig2 and g1==g2:
		
		
		return True
	else:
		return False
def get_config_count(dataf,scene,fig,ground):
	count = 0
	for index, row in dataf.iterrows():
		if row["Scene"]==scene and row["Figure"]==fig and row["Ground"]==ground:
			count +=1
	return count

# config_match("0821c204-7c5f-4394-b9c0-832d1019d28f","84c6529c-9835-42bf-b1f3-b7c5684f4a98")
def get_prepositions_dataframe(preposition):
	pdataset = pd.DataFrame()
	for index, row in dataset.iterrows():
		if row["Preposition"]==preposition:
			pdataset= pdataset.append(row, ignore_index=False)
	return pdataset



def add_row_count(pdataset):
	outdataset = pd.DataFrame()
	index = 0
	for index1, row1 in pdataset.iterrows():
		for index2, row2 in outdataset.iterrows():
			if config_match(row1,row2):
				break
		else:

			# Only executed if inner loop did not break
			n = get_config_count(pdataset,row1["Scene"],row1["Figure"],row1["Ground"])
			outdataset.at[index,"Count"] = n
			outdataset.at[index,"Scene"] = row1["Scene"]
			outdataset.at[index,"Figure"] = row1["Figure"]
			outdataset.at[index,"Ground"] = row1["Ground"]
			
			index +=1

	return outdataset



def add_ratio(outdataset):
	maxc = outdataset.loc[:,"Count"].max()
	for index2, row2 in outdataset.iterrows():
		

		
		c = outdataset.at[index2,"Count"]
		outdataset.at[index2,"Ratio"] = float(c)/maxc
	return outdataset

# def add_ratio_value(scene,fig,ground,feature):



preposition_list=['in', 'inside', 'against','on','on top of', 'under',  'below',  'over','above']



for preposition in preposition_list:
	pdataset = get_prepositions_dataframe(preposition)
	out = add_row_count(pdataset)
	out = add_ratio(out)

	print(preposition)
	print(out)
	# Need to process relations csv and add ratio onto end
	## Add path to classes please!
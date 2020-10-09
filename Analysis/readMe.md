# Info

Scripts given here are used to process and analyse the data.

Versions used:
Python 3
Pandas 1.0.3

Explanation of some folders:

# Folders

The code may be used to process multiple studies. 

The data for each study can be found in the associated folder.

The 2020 study folder contains the following:

## feature values
Contains information on feature values from configurations in all scenes. Standardised values are created by preprocess_features.py and read in compile_instances.py

## collected data
Collection of raw and cleaned annotation lists.

## Scene Data
Contains information and features extracted from the study scenes

## stats
Collection of results

The 2019 study folder contains the following:

## Scene Data
Contains information and features extracted from the study scenes

## collected data
Collection of raw and cleaned annotation lists.

## constraint data
List of constraints generated from data. This is created using compile_instances.py and read in basic_model_testing.py

## feature values
Contains information on feature values from configurations in all scenes. Standardised values are created by preprocess_features.py and read in compile_instances.py

## model info
Collection of basic model parameters when trained on all scenes

## polysemy
Collection of outputs related to polysemy model, see the readme in the folder for more info

## preposition data
Data related to specific prepositions. Giving configurations and how often they are selected with a given preposition in sv_task

## scores
Collection of scores from testing basic models


## stats
Collection of general stats giving overview of each task
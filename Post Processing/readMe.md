# Info
This file provides explanantion of folders and files in this folder

# Files
## process_data.py
Script to run for newly collected annotations.
Input: annotation and user info csv from data collection
Output: Clean annotation lists. Basic stats. User agreement calculations
Feature values are included later

## preprocess_features.py
Standardises feature values.
Input: features calculated from unity scripts.
Outputs to feature values folder.

## compile_instances.py
Run process_data and preprocess_features before this.

Input: cleaned annotation, user lists and list of feature values for configurations (see Relationship class)
Compiles annotation instances, adds feature values to them
Output: For each task: Basic stats for each preposition. For sv task writes a csv of feature values with selection information
Also has functions to generate constraints from comparative data

## feature_analysis.py
First run compile_instances.py
Input: Configuration selection info from compile_instances. Constraints from Comp task
Looks at how features relate to categorisation and likelihood of selection
Outputs prototypes and plots
Can be run after compile_instances

## polysemy_analysis.py
First run compile_instances.py
Generates polysemy models, outputs parameters and tests them.

## classes.py
Collection of classes used in above

# Folders

## collected data
Collection of raw and cleaned data from 2019 study

## constraint data
List of constraints generated from data. This is created using compile_instances.py and read in feature_analysis.py

## feature values
Contains information on feature values from configurations in all scenes. Standardised values are created by preprocess_features.py and read in compile_instances.py

## model info
Collection of baseline model parameters when trained on all scenes

## polysemy
Collection of outputs related to polysemy model, see the readme in the folder for more info

## preposition data
Data related to specific prepositions. Giving configurations and how often they are selected with a given preposition in sv_task

## scores
Collection of scores from testing models


## stats
Collection of general stats giving overview of each task
# Documentation
Rough docs to be improved!!!

## Simple Set-Up


## Requirements

Note that some scripts require the NonConvexMeshCollider Asset, which is available on the Unity Store.

## Scene Data
Scene Data folder contains calculated features from each scene and a script for extracting conceptnet properties. This folder is used later in /Post Processing/Analysis

## Assets
Assets folder contains necessary assets for game creation and running -- scripts, scenes, meshes etc..

### Feature Extraction

The feature extraction folder contains scripts associated with getting feature values from configurations in scenes.

To extract the given features for all scenes: 
* Begin by creating a folder named 'Scene Data' in your project directory
* Run the write_scene_info.cs script by clicking the menu item 'My Tools/Write Scene Info'
* Run the python script commonsense_properties.py which is found in the Scene Data folder
* Now run "My Tools/Calculate Geometric Features", this may take some time so be patient
* Finally run "My Tools/Calculate Functional Features", again this may take some time

The menu item "My Tools/Add Feature Check Script" can be used to inspect feature values within scenes

### Unity Scripts
Contains useful scripts for Unity Editor

### Scripts

Contains script used in-game

### Materials Meshes and Prefabs

Contains assets used for object models in scenes

### Scenes

Contains scenes and lighting information

### Resources

Contains resources (e.g. materials) which can be accessed in game

## Set Up
Below gives info on how to amend/build the game
### Scenes

Scenes must be set up following particular rules which are collected here but need organising (Note: comp = Comparative Task and sv = Preposition selection task):

* main scene should contain 'mainEmpty' GameObject which holds main script
* Scene file names should contain task abbreviation --  sv, pq and comp tasks are shared scenes. Currently these aren't distinguished. Main.cs handles which scenes are done for which task.
* Ground Objects in scenes are given tag "ground" or "figureground". Figures to compare with them are given figure or "figureground" tag.
* To associate a ground with a preposition for the comp task, ground are given empty objects as children with preposition tags. The tags have to correspond to given prepositions in the preposition list
* All grounds are compared against all figures
* For the screen task figures are set as above and also given a preposition tag
* Scene cameras must have "MainCamera" tag. camera is restricted to bounding box of room in scene template

Once all scenes have been created, they must be added to the build settings and various edits must be made. This is handled by the finalise_scenes.cs script which must be executed once all scenes have been created:

* finalise scenes.cs adds scenes to build (except example and template etc..) and must be run whenever scenes are added or removed. Sets player menu as first scene in index then iterates through scenes and bakes lighting. This script adds all scenes in the MainFolder directory to the buildsettings
* Also edits Main.cs to add the scenes to the scene list
* Also adds camera vision script to objects tagged 'MainCamera' and removes their audio listeners
* Bakes lighting in each scene

Also, for the feature extraction when adding new scenes, need to rerun scene_info script, then commonsense properties python script and then feature calculation (see Feature Extraction)

#### Adding Objects
Setting at rest: Physics is dealt with by MeshObject class in set_objects_at_rest.cs. Some objects names will need adding to attributes of this class if adding new objects

Make sure to not share naming of objects eg. "box" issue as may want to have different physics properties

Object names should be distinguished by using brackets as clean names are made by finding the first "("

Don't use red or green objects

#### Removing Scenes
Scenes need removing from build and removing from scene list in main.cs. To do this, open the project, delete the scenes and then run finalise_scenes.cs. Then build (to same folder name!)

### Other Assets
figure_material and ground_material need to be in Resources folder
### Notes


To set up tasks and/or change order of tasks, Main.cs must be edited. 'begin()' method describes which is the first task to do and 'change_task()' defines which tasks are done after the first. To create a new task all, new task instance must be created in Main class where its behaviour is defined in TaskScene



## Game Scripts
Describes purpose of game scripts, what they are attached to, how they should be added and possible variables to edit.

### Camera vision

#### Behaviour
Manages movement of camera.
* Camera is restricted to bounding box of room in scene template.
* Arrow keys move camera as if a human walking.
* 1 and 2 moves player up and down
* 0 key locks screen to allow looking around with cursor

#### Inclusion
 Gets attached to main camera in each scene. This is done automatically by "Finalise scenes.cs".

### Instruction

#### Behaviour
This handles the first scene shown for each task. Instructions are displayed and then the task scenes are loaded. Exact instructions are handled by "main.cs"

#### Inclusion
Is added to  canvas in instruction scene

#### Variables
Scene Gameobjects that need adding in editor: title_obj;instruction_obj;ButtonText;loadingImage; 
Name of Empty object in main scene: "mainEmpty"
Name of 'Instruction' scene: "instruction"

### Main

#### Behaviour
Script which brings everything together. Includes 'Task' and 'TaskScene' classes.

All grounds are compared against all figures

Highlighting works by storing and then changing all materials on object.

Active ground can't be selected in comp task
#### Inclusion
Script added to empty in main scene.This scene is open during all data collection with other scenes loaded on top. In this way the 'Main' instance is never destroyed

#### Variables

##### TaskScene
Task Scene class acts like the usual Scene class except more information is stored regarding configurations in the scene. Also has methods to deal with setting new examples and object highlighting.
	
##### Task
Task class distinguishes different tasks user can do.

##### Main

### Player Menu Main

#### Behaviour
Writes user data and loads first scene

#### Inclusion
Is added to  canvas in player_menu scene

### Return to Start
 Script called to return player to start menu

#### Notes
passwords for http post are exposed in main.cs and player menu main



Note: passwords for http post are exposed in main.cs and player menu main






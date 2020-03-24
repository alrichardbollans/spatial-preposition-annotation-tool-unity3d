# Documentation
Rough docs to be improved!!!

# Simple Set-Up

Below gives brief information on how to amend/build the game.

## Scenes

### Names
* `Main` class contains names of various non-testing scenes which need referencing in scripts. If names of any of the non-test scenes are changed, make sure they are also updated here. `first_scene_name` is also set here which defines which scene is opened first.

* Scene file names should contain task abbreviation --  sv, pq and comp tasks are shared scenes. Currently these aren't distinguished. Main.cs handles which scenes are done for which task.

Also, for the feature extraction when adding new scenes, need to rerun scene_info script, then commonsense properties python script and then feature calculation (see Feature Extraction)

### Adding Objects
Setting at rest: Physics is dealt with by MeshObject class in set_objects_at_rest.cs. Some objects names will need adding to attributes of this class if adding new objects

Make sure to not share naming of objects eg. "box" issue as may want to have different physics properties

Object names should be distinguished by using brackets as clean names are made by finding the first "("

Don't use red or green objects

### Object Tags
* MainCamera: There should be one camera in each scene given the tag "MainCamera"
* `ground`:
* `figure`:
* `figureground`:
* prepositions:

* Ground Objects in scenes are given tag `ground` or `figureground`. Figures to compare with them are given `figure` or `figureground` tag.

* All grounds are compared against all figures

* For the screen task, the ground is given a `ground` tag and a child object which contains a preposition tag. The correct figure (for ground-preposition pair) is given the `figure` tag.

* Scene cameras must have "MainCamera" tag. camera is restricted to bounding box of room in scene template

### Removing Scenes
Scenes need removing from build and removing from scene list in main.cs. To do this, open the project, delete the scenes and then run finalise_scenes.cs. Then build (to same folder name!)


## Other Assets
figure_material and ground_material need to be in Resources folder

## Build

Once all scenes have been created, they must be added to the build settings and various edits must be made. This is handled by the finalise_scenes.cs script which must be executed once all scenes have been created:

* finalise scenes.cs adds scenes to build (except example and template etc..) and must be run whenever scenes are added or removed. Sets player menu as first scene in index then iterates through scenes and bakes lighting. This script adds all scenes in the MainFolder directory to the buildsettings
* Also edits Main.cs to add the scenes to the scene list
* Also adds camera vision script to objects tagged 'MainCamera' and removes their audio listeners
* Bakes lighting in each scene

## Notes


To set up tasks and/or change order of tasks, Main.cs must be edited. 'begin()' method describes which is the first task to do and 'change_task()' defines which tasks are done after the first. To create a new task all, new task instance must be created in Main class where its behaviour is defined in TaskScene.

Annotations are output via a http post to a php script on the server. Associated strings are given in the `Main` class and *passwords for http post are exposed here*.



# Feature Extraction

The feature extraction folder contains scripts associated with getting feature values from configurations in scenes.

To extract the given features for all scenes: 
* Begin by creating a folder named 'Scene Data' in your project directory
* Run the write_scene_info.cs script by clicking the menu item 'My Tools/Write Scene Info'
* Run the python script commonsense_properties.py which is found in the Scene Data folder
* Now run "My Tools/Calculate Geometric Features", this may take some time so be patient
* Finally run "My Tools/Calculate Functional Features", again this may take some time

The menu item "My Tools/Add Feature Check Script" can be used to inspect feature values within scenes


## Requirements

Note that some scripts require the NonConvexMeshCollider Asset, which is available on the Unity Store.

## Scene Data
Scene Data folder contains calculated features from each scene and a script for extracting conceptnet properties. This folder is used later in analysis.

# Assets
Assets folder contains necessary assets for game creation and running -- scripts, scenes, meshes etc..

## Feature Extraction

The feature extraction folder contains scripts associated with getting feature values from configurations in scenes.

To extract the given features for all scenes: 
* Begin by creating a folder named 'Scene Data' in your project directory
* Run the write_scene_info.cs script by clicking the menu item 'My Tools/Write Scene Info'
* Run the python script commonsense_properties.py which is found in the Scene Data folder
* Now run "My Tools/Calculate Geometric Features", this may take some time so be patient
* Finally run "My Tools/Calculate Functional Features", again this may take some time

The menu item "My Tools/Add Feature Check Script" can be used to inspect feature values within scenes

## Unity Scripts
Contains useful scripts for Unity Editor

## Scripts

Contains script used in-game

## Materials Meshes and Prefabs

Contains assets used for object models in scenes

## Scenes

Contains scenes and lighting information

## Resources

Contains resources (e.g. materials) which can be accessed in game



# Game Scripts
In-game scripts are stored in Assets/Scripts and must be attached to the correct game objects in the editor.

Some of these scripts require gameobjects to be assigned in the editor. These are already done in the templates, also comments in scripts describe which objects should be attached.

## Camera vision
Manages movement of camera. Gets attached to main camera in each scene automatically by "Finalise scenes.cs".

## Instruction

This handles the first scene shown for each task. Instructions are displayed and then the task scenes are loaded. Exact instructions are handled by "main.cs". Is added to  canvas in instruction scene.

## Main

Script which brings everything together. Includes 'Task' and 'TaskScene' classes.

Script added to mainEmpty in main scene. This scene is open during all data collection with other scenes loaded on top. In this way the 'Main' instance is never destroyed

## Player Menu Main

Writes user data and loads first scene. Is added to  canvas in player_menu scene.








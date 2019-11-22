# Documentation
Rough docs to be improved!!!

## Set Up
### Scenes
main scene should contain 'mainEmpty' GameObject which holds main script
Object names should be distinguished by using brackets as clean names are made by finding the first "("

Don't use red or green objects

Scene file names should contain task abbreviation --  sv, pq and comp tasks are shared scenes. Currently these aren't distinguished. Main.cs handles which scenes are done for which task.

Ground Objects in scenes are given tag "ground" or "figureground". Figures to compare with them are given figure or "figureground" tag.

To associate a ground with a preposition for the comp task, ground are given empty objects as children with preposition tags. The tags have to correspond to given prepositions in the preposition list

All grounds are compared against all figures
// 
// For the screen task figures are set as above and given a preposition tag

// Scene cameras must have "MainCamera" tag. camera is restricted to bounding box of room in scene template

finalise scenes.cs adds scenes to build (except example and template etc..) and must be run whenever scenes are added or removed, sets player menu as first. then iterates through scenes and bakes lighting

/// This script adds all scenes in the MainFolder directory to the buildsettings
// Should be run after all scenes have been created
/// Also edits Main.cs to add the scenes to the scene list
/// Also adds camera vision script to objects tagged 'MainCamera' and removes their audio listeners
// Bakes lighting in each scene

When adding new scenes, need to rerun scene_info script, then commonsense properties python script and then feature calculation

#### Adding Objects
Setting at rest: Physics is dealt with by MeshObject class in set_objects_at_rest.cs. Some objects names will need adding to attributes of this class if adding new objects

Make sure to not share naming of objects eg. "box" issue as may want to have different physics properties

#### Removing Scenes
Scenes need removing from build and removing from scene list in main.cs. To do this, open the project, delete the scenes and then run finalise_scenes.cs. Then build (to same folder name!)

#### Panels
GameObject panel;
	public GameObject selected_figure_panel;
	public GameObject selected_figure_text_obj;
	public Text selected_figure_panel_text_component;
	public GameObject instruction_obj;
	public Text instruction_text_component;
	public static string[] task_panel_names = {"comp_screen_panel","sv_panel"}; // names of UI elements specific to tasks
	public List<GameObject> task_panels =  new List<GameObject>();

### Other Assets
figure_material and ground_material need to be in Resources folder
### Notes


To set up tasks and/or change order of tasks, Main.cs must be edited. 'begin()' method describes which is the first task to do and 'change_task()' defines which tasks are done after the first. To create a new task all, new task instance must be created in Main class where its behaviour is defined in TaskScene

## Editor Scripts

Script useful in creating scenes. Usually providing a button in the editor which runs the script.

### Set objects at rest

### Waitinggame

### Waitinggame_FFE

### Write Scene Info

### Finalise scenes
finalise scenes.cs adds scenes to build (except example and template) and must be run whenever scenes are added or removed, sets player menu as first. then iterates through scenes and bakes lighting

/// This script adds all scenes in the MainFolder directory to the buildsettings
// Should be run after all scenes have been created
/// Also edits Main.cs to add the scenes to the scene list
/// Also adds camera vision script to objects tagged 'MainCamera' and removes their audio listeners
// Bakes lighting in each scene

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
 Gets attached to main camera in each scene. This is done automatically by "Finaise scenes.cs".

### Instruction

#### Behaviour
This handles the first scene shown for each task. Instructions are displayed and then the task scenes are loaded. Exact instructions are handled by "main.cs"

#### Inclusion
Is added to ???

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
	
comp_preposition_list

fig_mat = Resources.Load("figure_material", typeof(Material)) as Material;
	Material grd_mat

##### Task
Task class distinguishes different tasks user can do.

instruction_list = instructions;
instruction_title = title;
scene_abbreviations.Add(n);
number_scenes_to_do = 10;

GameObject panel;
public GameObject selected_figure_panel;
public GameObject selected_figure_text_obj;
public Text selected_figure_panel_text_component;
public GameObject instruction_obj;
public Text instruction_text_component;
public static string[] task_panel_names = {"comp_screen_panel","sv_panel"}; // names of UI elements specific to tasks
public List<GameObject> task_panels =  new List<GameObject>();

##### Main
public GameObject confirm_text;
	public GameObject confirmQuit_text;
	public GameObject help_panel;

public GameObject None_toggle_obj;
Toggle None_toggle;


public GameObject loadingImage;

url for php script on server.
Authentication for php script on server

input_list_of_scenes: list of scenes to use in study. This is automatically populated by finalise scenes script.

sv = semantic validity
// pq =  predicational question
// comp =  comparative task
// game = game

// PlayerPref values are named as follows:
// selectedFigure = "selectedFigure";
// selectedGround
// task
// preposition
// scene

Ground Objects in scenes are given tag "ground" or "figureground". FIgures to compare with them are given figure or "figureground" tag

To associate a ground with a preposition for the comp task, ground are given empty objects as children with preposition tags. The tags have to correspond to given prepositions in the preposition list

For the screen task figures are set as above and given a preposition tag

// Scene cameras must have "MainCamera" tag

be careful using PlayerPrefs.GetString("selectedGround","");. In general objects have different names but sometimes this will not be the case in added scenes

// pay attention to task.set_task
//When naming objects in game be careful with strings inc. "sv" "pq" "comp" "game" "panel" "insruction" "toggle"
#### Notes
passwords for http post are exposed in main.cs and player menu main

### Player Menu Main
Note: passwords for http post are exposed in main.cs and player menu main
### Return to Start
// 
 Note: passwords for http post are exposed in main.cs and player menu main
// 

// 

/// 

// 

// Ground Objects in scenes are given tag "ground" or "figureground". FIgures to compare with them are given figure or "figureground" tag


// 





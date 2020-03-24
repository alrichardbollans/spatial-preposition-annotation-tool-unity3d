// Script added to mainEmpty in main scene.
// Must be called "Main.cs" and stored in "Assets/Scripts".

// This scene is open during all data collection with other scenes loaded on top. 
// In this way the 'Main' instance is never destroyed

// This script forms the base.
// Implements various tasks
// sv = semantic validity
// pq =  predicational question
// comp =  comparative task
// game = game

// Active ground can't be selected in comp task

/// Scene file names should contain task abbreviation --  sv, pq and comp tasks are shared scenes

// Highlighting works by storing and then changing all materials on object.

// Ground Objects in scenes are given tag "ground". FIgures to compare with them are set as children
// To associate a ground with a preposition for the comp task, ground are given empty objects as children with preposition tags
// For the screen task figures are set as above and given a preposition tag

// Scene cameras must have "MainCamera" tag.

// Be careful using PlayerPrefs.GetString("selectedGround","");. 
// In general objects have different names but sometimes this will 
// not be the case in added scenes.

// Pay attention to task.set_task.
// When naming objects in game be careful with strings 
// including any of the strings given by Main class.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine.EventSystems;

using System.IO;
using System.Text;
using System.Linq;

/// <summary>
/// TaskScene class acts like the usual Scene object except 
/// more information is stored regarding object configurations.
/// </summary>
public class TaskScene {

	public string name; //Scene Name
	public string task_type; 
	public List<GameObject> ground_list = new List<GameObject>();
	public List<GameObject> figure_list = new List<GameObject>();
	// Camera needed for raycasting
	GameObject[] cam_list;
	public Camera main_camera;
	public List<GameObject[]> configuration_list = new List<GameObject[]>();
	public List<List<object>> comparison_list = new List<List<object>>(); // list of ground/preposition pairs
	public string screening_preposition;
	public GameObject[] active_configuration; // Figure Ground pair
	public List<object> active_comparison; // Ground preposition pair
	// Preposition list for comp task.
	List<string> comp_preposition_list = new List<string> {"on","on top of", "in", "inside","against","over","below","above","under"};

	Material fig_mat = Resources.Load("figure_material", typeof(Material)) as Material;
	Material grd_mat = Resources.Load("ground_material", typeof(Material)) as Material;

	// Stores materials to undo highlighting.
	Material[] stored_fig_mats;
	Material[] stored_grd_mats;

	// Strings for storing values in PlayerPrefs
	static string selectedFig_playerpref = Main.selectedFig_playerpref;
	static string selectedgrd_playerpref = Main.selectedgrd_playerpref;
	static string prep_playerpref = Main.prep_playerpref;

	// Tag names.
	static string ground_tag = Main.ground_tag;
	static string figure_tag = Main.figure_tag;
	static string fig_grd_tag =  Main.fig_grd_tag;
	// Random instance for generating random integers.
	static System.Random rnd = new System.Random();

	/// <summary>
    /// Create class instance.
    /// </summary>
    /// <param name="n">Scene name.</param>
    /// <param name="type">task abbreviation.</param>
	public TaskScene(string n, string type){
		name = n;
		task_type = type;
		
	}

	/// <summary>
    /// Stores figure name in playerprefs and highlights figure.
    /// </summary>
    /// <param name="fig">Figure to set.</param>
	public void set_figure(GameObject fig){ 
		
		PlayerPrefs.SetString(selectedFig_playerpref, fig.name);

		highlight_figure(fig);
		Debug.Log("new figure:");
		Debug.Log(fig.name);
		
		
	}

	/// <summary>
    /// Stores ground name in playerprefs and highlights ground.
    /// </summary>
    /// <param name="gr">Ground to set.</param>
	public void set_ground(GameObject gr){
		
		PlayerPrefs.SetString(selectedgrd_playerpref, gr.name);
		highlight_ground(gr);
		Debug.Log("new ground:");
		Debug.Log(gr.name);
		
		
		
	}

	/// <summary>
    /// Stores preposition in playerprefs.
    /// </summary>
    /// <param name="preposition">preposition to set.</param>
	public void set_preposition(string preposition){
		
		PlayerPrefs.SetString(prep_playerpref, preposition);

	}

	
	/// <summary>
    /// Unhighlights figure and updates player prefs.
    /// </summary>
	public void deselect_figure(){
		
		// Get old figure
		string old_figure_name = PlayerPrefs.GetString(selectedFig_playerpref,"");
		// Note find objects can be heavy process (see docs if needs calling every frame)
		GameObject old_figure = GameObject.Find(old_figure_name);
		//If there was actually a figure, undo highlighting
		if (old_figure != null){
			Debug.Log("Unhighlighting figure: " + old_figure_name);
			unhighlight_figure(old_figure);
			
		}
		
		PlayerPrefs.SetString(selectedFig_playerpref, "");
		
	}

	
	/// <summary>
    /// Unhighlights ground and updates player prefs.
    /// </summary>	
	public void deselect_ground(){
		
		Debug.Log("Deselect ground is called");
		string old_grd_name = PlayerPrefs.GetString(selectedgrd_playerpref,"");
		Debug.Log("old_grd_name is " + old_grd_name);
		// Note find objects can be heavy process (see docs if needs calling every frame)
		GameObject old_grd = GameObject.Find(old_grd_name);
		//If there was actually a ground, undo highlighting
		if (old_grd != null){
			Debug.Log("Unhighlighting ground: " + old_grd_name);

			unhighlight_ground(old_grd);

		}
		// Remove fig form player prefs
		PlayerPrefs.SetString(selectedgrd_playerpref, "");
		
		// g_label.text = "Ground:";
	}
	
	/// <summary>
    /// Highlights figure.
    /// First stores figures materials then changes them all to highlight material.
    /// </summary>
    /// <param name="fig">Figure to highlight.</param>
	public void highlight_figure(GameObject fig){
		// Stores old figure materials
		stored_fig_mats = fig.GetComponent<Renderer>().materials;
		// Create a new array with same number of fig mat instances
		// All materials in array are the fig_mat for highlighting
		List<Material> new_mats_list = new List<Material>();
		foreach(Material m in stored_fig_mats){
			new_mats_list.Add(fig_mat);
		}

		Material[] new_mats =new_mats_list.ToArray();

		fig.GetComponent<Renderer>().materials = new_mats;		

	}

	
	/// <summary>
    /// Highlights ground.
    /// First stores grounds materials then changes them all to highlight material.
    /// </summary>
    /// <param name="grd">Ground to highlight.</param>
	public void highlight_ground(GameObject grd){
		// Stores old ground materials
		stored_grd_mats = grd.GetComponent<Renderer>().materials;
		// Create a new array with same number of grd mat instances
		List<Material> new_mats_list = new List<Material>();
		foreach(Material m in stored_grd_mats){
			new_mats_list.Add(grd_mat);
		}

		Material[] new_mats =new_mats_list.ToArray();

		grd.GetComponent<Renderer>().materials = new_mats;

	}
	/// <summary>
    /// Removes highlighting from ground.
    /// </summary>
    /// <param name="grd">Ground to unhighlight.</param>
	public void unhighlight_ground(GameObject grd){
		grd.GetComponent<Renderer>().materials = stored_grd_mats;
	}

	/// <summary>
    /// Removes highlighting from figure.
    /// </summary>
    /// <param name="fig">Figure to unhighlight.</param>
	public void unhighlight_figure(GameObject fig){
		fig.GetComponent<Renderer>().materials = stored_fig_mats;
	}
	

	/// <summary>
    /// Populates ground and figure lists to generate configurations to test.
    /// </summary>
	void populate_fig_ground_list(){
		GameObject[] g_list = GameObject.FindGameObjectsWithTag(ground_tag);
		GameObject[] f_list = GameObject.FindGameObjectsWithTag(figure_tag);
		GameObject[] fg_list = GameObject.FindGameObjectsWithTag(fig_grd_tag);

		foreach(GameObject gobj in g_list){
			ground_list.Add(gobj);
		}

		foreach(GameObject gobj in f_list){
			figure_list.Add(gobj);
		}

		foreach(GameObject gobj in fg_list){
			figure_list.Add(gobj);
			ground_list.Add(gobj);
		}
	}
	
	/// <summary>
    /// Loads scene and finds configurations to test.
    /// </summary>
	public IEnumerator set_scene_coroutine(){
		

		// Needs to be set up as a coroutine so that it only continues after scene is fully loaded
		SceneManager.LoadScene(name,LoadSceneMode.Additive);
		yield return null;
		SceneManager.SetActiveScene(SceneManager.GetSceneByName(name));
		// Set camera
		cam_list = GameObject.FindGameObjectsWithTag("MainCamera");
		foreach(GameObject c in cam_list){
			if (c.scene.name == name){
				main_camera = c.GetComponent<Camera>();
			}
		}
		populate_fig_ground_list();
		// Add in configurations
		if (task_type == Main.sv_abv || task_type == "pq" || task_type == Main.screen_abv){
			// Create list of grounds by tag
			
			// Add configurations to list by taking children of ground
			foreach (GameObject ground in ground_list){
				foreach(GameObject fig in figure_list){
					if(fig.name != ground.name){
						GameObject[] config = {fig,ground};
						configuration_list.Add(config);
					}
				}

				if(task_type == Main.screen_abv){
					foreach(Transform emp in ground.transform){
						string p = emp.gameObject.tag;
						if (comp_preposition_list.Contains(p)){
							screening_preposition = p;
							
						}
					}
				}
				
			}
		}

		

		if (task_type == Main.comp_abv){
			// Create list of grounds by tag
			
			
			foreach (GameObject ground in ground_list){
				add_new_comp_config(ground);
				add_new_comp_config(ground);
				add_new_comp_config(ground);
				add_new_comp_config(ground);
				add_new_comp_config(ground);
			}
			//// Next part can be uncommented if want to use later

			// 	foreach (Transform emp in ground.transform){
			// 		string p = emp.gameObject.tag;
			// 		if (comp_preposition_list.Contains(p)){
						
			// 			List<object> config = new List<object>();
			// 			config.Add(ground);
			// 			config.Add(p);
						
			// 			comparison_list.Add(config);
			// 			}
			// 		}
			// 	}
		}
		
	}
	
    /// <summary>
    /// Adds preposition-ground pair to comparison list.
    /// Preposition is randomly selected and removed from list.
    /// </summary>	
	void add_new_comp_config(GameObject ground){
		int r = rnd.Next(comp_preposition_list.Count);
		string p = comp_preposition_list[r];

		comp_preposition_list.Remove(p);

		List<object> config = new List<object>();

		config.Add(ground);
		config.Add(p);
		
		comparison_list.Add(config);
	}
	
	/// <summary>
    /// Sets new configuration to test.
    /// </summary>
    /// <returns>
	/// true if a new configuration can be set in the scene, otherwise False.
	/// </returns>
	public bool set_new_example(){
		if (task_type == Main.sv_abv || task_type == "pq" || task_type == Main.screen_abv){


			// Unselect figure and ground
			deselect_figure();
			deselect_ground();

			if (configuration_list.Contains(active_configuration)){
				// If there is an active configuration pick next configuration in list
				int i = configuration_list.IndexOf(active_configuration);
				
				// If there is a next one to pick do that and return true, else return false
				if (i+1 < configuration_list.Count){

					active_configuration = configuration_list[i+1];
					GameObject f = active_configuration[0];
				
					GameObject g = active_configuration[1];
					if(task_type != Main.screen_abv){
						set_figure(f);
					}
					
					set_ground(g);
					Debug.Log("New example:");
					Debug.Log("New Figure:" + f.name);
					Debug.Log("New Ground:" + g.name);
					if (task_type == Main.screen_abv){
						Debug.Log("setting preposition: " + screening_preposition);
						set_preposition(screening_preposition);
					}
					return true;
				}
				
				else {
					Debug.Log("No more configurations, will load next scene");
					//Load next scene
					return false;
				}
				
			}
			else if(configuration_list.Count >0) {
				// If there is no active configuration start with the first one in list
				active_configuration = configuration_list[0];
				GameObject f = active_configuration[0];
			
				GameObject g = active_configuration[1];

				if(task_type != Main.screen_abv){
						set_figure(f);
					}
				set_ground(g);
				if (task_type == Main.screen_abv){
					Debug.Log("setting preposition: " + screening_preposition);
					set_preposition(screening_preposition);
				}
				return true;
				
			}

			else{
				Debug.Log("No configurations for this task in this scene, will load next scene");
				//Load next scene
				return false;
			}
		}


		
		if (task_type == Main.comp_abv){
			deselect_figure();
			deselect_ground();
			


			if (comparison_list.Contains(active_comparison)){
					// If there is an active configuration pick next configuration in list
					int i = comparison_list.IndexOf(active_comparison);
					if (i+1 < comparison_list.Count){
						
						// int x = Random.Range(0,comparison_list.Count);
						active_comparison = comparison_list[i+1];
						GameObject g = active_comparison[0] as GameObject;
						string p = active_comparison[1] as string;

						set_ground(g);

						set_preposition(p);
						return true;	
					}
					else {
					Debug.Log("No more configurations, will load next scene");
					//Load next scene
					return false;
					}
			}
			else if(comparison_list.Count>0) {
					// If there is no active configuration start with the first one in list
					active_comparison = comparison_list[0];
					GameObject g = active_comparison[0] as GameObject;
					string p = active_comparison[1] as string;

					set_ground(g);

					set_preposition(p);
					return true;
					
				}
			else{
				Debug.Log("No configurations for this task in this scene, will load next scene");
				//Load next scene
				return false;
			}
		}

		else{
			Debug.Log("Error: Task type incorrectly set");
			Debug.Log("task_type = " + task_type);
			return false;
		}
		}
	}

	
/// <summary>
/// The main Task class.
/// Contains information on what is displayed for the task.
/// </summary>
public class Task {
	// Be careful editing below list. It is edited by a script (finalise_scenes.cs) 
	// (button in the editor)
    public static string[] input_list_of_scenes = {"compsva12","compsva13","compsva14","compsva23","compsva24","compsva25","compsvi12","compsvi13","compsvi14","compsvi14a","compsvi15","compsvi23","compsvi24","compsvi25","compsvi26","compsvi26a","compsvi3","compsvi34","compsvi46","compsvo12","compsvo12v","compsvo13v","compsvo14","compsvo14v","compsvo16","compsvo17","compsvo19","compsvo23v","compsvo26","compsvo27","compsvo28","compsvo29","compsvo34v","compsvo67","compsvo68","compsvo68a","compsvo78","compsvo79","compsvu23","compsvu34","compsvu63","compsvula","example","finish","instruction","main","player_menu","scene_template","screen0","screen1","screening_fail","test"};
	//
	public string name;
	public string instruction; //Instruction to give in each scene

	string new_instruction;  //Instruction to give in each scene
	//task name abbreviations with shared scenes
	List<string> scene_abbreviations =  new List<string>(); 

	public string[] instruction_list; // List of instructions to give before starting
	public string instruction_title; //Title for instruction scene
	
	public GameObject panel;
	public GameObject selected_figure_panel;
	public GameObject selected_figure_text_obj;
	public Text selected_figure_panel_text_component;
	public GameObject instruction_obj;
	public Text instruction_text_component;
	public static string[] task_panel_names = {"comp_screen_panel","sv_panel"}; // names of UI elements specific to tasks
	public List<GameObject> task_panels =  new List<GameObject>();


	public List<GameObject> active_objects =  new List<GameObject>(); // list of all objects in panel hieracrchy
	public List<Toggle> list_of_toggles = new List<Toggle> (); // Toggles for selecting prepositions
	public List<Toggle> preposition_toggles = new List<Toggle> ();
	
	public List<string> list_of_scenes = new List<string> (); // List of all scenes doesn't get chanegd
	public List<string> list_of_scenes_to_do = new List<string> (); // List of scenes where done scenes are removed 

	public int number_scenes_to_do;

	static string task_player_pref = Main.task_player_pref;

	/// <summary>
	/// Gets scene lists for task.
	/// Any scene name which contains any of the scene_abbreviations for the task is added.
	/// </summary>
	public void get_scenes(){
		// Adds scenes to task
		for (int n = 0; n < input_list_of_scenes.Length; ++n){
			string s;
			s = input_list_of_scenes[n];

			foreach(string abv in scene_abbreviations){
				if (s.Contains(abv) && !s.Contains("fail")){
					list_of_scenes.Add(s);
					list_of_scenes_to_do.Add(s);
					break;
				}
			}
			
		}
		
	}
	/// <summary>
	/// Instantiate Task.
	/// Sets panels and toggles for task.
	/// </summary>
	/// <param name="n">Task name.</param>
	/// <param name="instructions">List of instructions for task.</param>
	/// <param name="title">Task title.</param>
	public Task(string n,string[] instructions,string title){
		
		
		name = n;
		instruction_list = instructions;
		instruction_title = title;
		scene_abbreviations.Add(n);
		number_scenes_to_do = 10;

		// if(name == Main.sv_abv || name == Main.comp_abv || name == "pq"){
		// 	//These tasks share scenes
		// 	scene_abbreviations.Add(Main.sv_abv);
		// 	scene_abbreviations.Add("pq");
		// 	scene_abbreviations.Add(Main.comp_abv);

		// }
		GameObject[] allObjects = Object.FindObjectsOfType<GameObject>();
		
		if (task_panels.Count == 0){
			foreach (GameObject obj in allObjects){
			
			if (task_panel_names.Contains(obj.name)){
				task_panels.Add(obj);
				}
			}
			
		}
		
		get_scenes();

		// Set panel
		foreach (GameObject g in task_panels){

			if (g.name.Contains(name)){
				panel = g;

			}
		}
		foreach(GameObject g in allObjects){
			if (g.name.Contains("selected_figure")){
				selected_figure_panel = g;
			}
		}
		// Populate list of active objects by what's in panel hierarchy
		if (panel != null){
			foreach (Transform obj in panel.transform){
				
				active_objects.Add(obj.gameObject);
				foreach(Transform got in obj.transform){
					active_objects.Add(got.gameObject);
				}
			}
		}
		if (selected_figure_panel != null){
			foreach (Transform obj in selected_figure_panel.transform){
				
				active_objects.Add(obj.gameObject);
				}
		}
		
		// Assign various objects
		foreach (GameObject g in active_objects){
			

			//Assign instruction obj
			if (g.name.Contains("instruction")){
				instruction_obj = g;
				instruction_text_component = instruction_obj.GetComponent<Text>();
			}

			if(g.name.Contains("selected_figure_text")){
				
				selected_figure_text_obj = g;
				selected_figure_panel_text_component = selected_figure_text_obj.GetComponent<Text>();
			}
			//Add to toggle list
			
			Toggle t = g.GetComponent(typeof(Toggle)) as Toggle;
			if (t != null){
				
				list_of_toggles.Add(t);

				if(!t.gameObject.name.Contains("none")){
					preposition_toggles.Add(t);
				}
			}
		}
	}


		

	/// <summary>
	/// Turn off non-preposition toggles.
	/// </summary>
	public void turn_off_toggles(){
		foreach (Toggle t in list_of_toggles){
				t.isOn =false;
			}
	}

	/// <summary>
	/// Turn off preposition toggles.
	/// </summary>
	public void turn_off_preposition_toggles(){
		foreach (Toggle t in preposition_toggles){
			
				t.isOn =false;
			
			}
	}
	
	/// <summary>
	/// Prepares for task.
	/// Set necessary panels to be active.
	/// Turn off toggles.
	/// </summary>
	public void set_task(){
		
		// De/Activate Objects
		foreach (GameObject g in task_panels){
			if (g == panel){
				g.SetActive(true);
			}

			else {
				g.SetActive(false);
			}

			if(g.name.Contains("selected_figure_panel") && (name == Main.comp_abv || name == Main.screen_abv)){
				g.SetActive(true);
			}
		}

		// Turn off toggles
		turn_off_toggles();
		
		// Set player prefs
		PlayerPrefs.SetString(task_player_pref, name);

		
	}

	public void set_text(){
		string p = PlayerPrefs.GetString(Main.prep_playerpref,"");
		string f = PlayerPrefs.GetString(Main.selectedFig_playerpref,"");
		string g = PlayerPrefs.GetString(Main.selectedgrd_playerpref,"");
		
		new_instruction = instruction.Replace(":preposition:","<b>" + p + "</b>");
		new_instruction = new_instruction.Replace(":figure:","<color=green><b>" + Main.clean_name(f) + "</b></color>");
		new_instruction = new_instruction.Replace(":ground:","<color=red><b>" + Main.clean_name(g) + "</b></color>");
		string[] vowels = new  string[] {"a", "e", "i", "o", "u", "h"};
		
		if (f != ""){
			Debug.Log("setting fig text");
			string l = Main.clean_name(f)[0].ToString();
			if (vowels.Contains(l)){//Any(s => s.Equals(Main.clean_name(f)[0]))){
				
				new_instruction = new_instruction.Replace(":a:","an");
			}

			else {
				new_instruction = new_instruction.Replace(":a:","a");
			}
		}
		instruction_text_component.text = new_instruction;
		selected_figure_panel_text_component.text = "Selected Object: ";

	}

	
}
	



/// <summary>
/// The main game monobehaviour class.
/// Begins game, loads scenes, handles user inputs etc..
/// </summary>
public class Main : MonoBehaviour {
	// Strings and values reused across scripts.
	// PlayerPrefs
	public static string selectedFig_playerpref = "selectedFigure";
	public static string selectedgrd_playerpref = "selectedGround";
	public static string prep_playerpref = "preposition";
	public static string task_player_pref = "task";
	public static string scene_player_pref = "scene";
	public static string userid_player_pref = "UserID";

	// Tags
	public static string ground_tag = "ground";
	public static string figure_tag = "figure";
	public static string fig_grd_tag = "figureground";
	public static string main_camera_tag= "MainCamera";

	// Scene names
	public static string first_scene_name = "player_menu";
	public static string main_scene_name = "main";
	public static string instruction_scene_name = "instruction";
	public static string fail_scene_name = "screening_fail";
	public static string finish_scene_name = "finish";

	// Store names of any scenes that shouldn't be included in build.
	public static List<string> non_test_scenes = new List<string> {"example", "scene_template", "test"};
	public static List<string> unselectable_scene_objects = new List<string> {"wall","floor","ceiling"};

	// Directory Info
	public static string MainFolder   = "Assets/Scenes";
	
	// Server strings
    public static string auth_username = "game";
    public static string auth_password =  "6KK6w4EhgcrBQHKzgL";
	public static string appendannotation_url = "/spatial_language_study/appendannotation.php";
	public static string writeuserdata_url = "/spatial_language_study/writeuserdata.php";

	// Task Abbreviations
	public static string sv_abv = "sv";
	public static string comp_abv = "comp";
	public static string screen_abv = "screen";

	// Input keys
	static public KeyCode ShowHelpKey = KeyCode.Delete;
	static public KeyCode quitKey = KeyCode.Escape;

	Task sv_task;
	Task pq_task;
	Task comp_task;
	Task screen_task;
	static public Task task;

	TaskScene task_scene;

	public GameObject confirm_text;
	public GameObject confirmQuit_text;
	public GameObject help_panel;

	public GameObject None_toggle_obj;
	Toggle None_toggle;
	
	public GameObject loadingImage;
	public Text SceneCounter_Text;
	string SceneCountertext = "Scenes left: :int:";

	public bool fail_loaded; // bool to know if fail scene is loaded.
	
	public int number_scenes_done = 0;

	// Create random object for random number generation later
	static System.Random rnd = new System.Random();
	
	

	GameObject[] allObjects;
	// List of all objects to output later
	List<string> all_objects = new List<string> ();
	string all_objects_string;
	
	bool confirm = false;
	bool confirm_quit = false;

	/// <summary>
	/// Awake is used to initialize any variables or game state before the game starts.
	/// Awake is called only once during the lifetime of the script instance.
	/// Creates tasks. Adds listeners to toggles. Deactivates some objects.
	/// </summary>
	void Awake(){
		
		// Get list of all game objects
		allObjects = Object.FindObjectsOfType<GameObject>();
		all_objects_string = "";
		foreach(GameObject obj in allObjects){
			all_objects.Add(obj.name);
			all_objects_string += obj.name + ",";
		}
		// Set intructions
		string screen_instruction_title = "Instructions";
		string[] screen_instructions = {"Before beginning you will be given <b>two quick examples</b> to complete\n \n \nClick Next..",	"You will be shown an indoor scene and a description of an object will be provided at the bottom of the screen. \n \n Click on the object that best fits the description.\n \n You will be prompted to press enter or click accept to confirm your selection. \n \nIf you are correct you will move on to the next stage.",
		"To move around the scene: \n - Use the <b>arrow keys</b> to move around \n - <b>Hold down the '0' key</b> to use the mouse to look around \n - Use the <b>'1' and '2' keys</b> to move up and down if you need to \n - Press the <b>'Delete' key</b> for help"	};
		
		string sv_instruction_title = "Task 1 Instructions";
		string[] sv_instructions = {"In this task you will be shown some objects and asked to select words which could <b>describe the relationship between them</b>.","A <b>pair</b> of objects will be highlighted, <b>one in <color=green>green</color></b> and <b>the other in <color=red>red</color></b>. You need to select <b>all</b> the words which describe <b>how the <color=green>green object</color> relates to the <color=red>red object</color></b>.","The words you may select are: 'on', 'on top of', 'in', 'inside', 'against', 'over', 'under', 'above' and 'below'. \n\n If none of the given words apply, select <b> 'None of the above'</b>.\n\n Once you have made your selections, click 'Submit'. A new pair and/or scene will then be displayed.","Remember, you can use the arrow keys to move around and while holding down the '0' key you can use the mouse to look around.\n\n Also, use the '1' and '2' keys to move up and down if you need to."};
		

		string comp_instruction_title = "Task 2 Instructions";
		string[] comp_instructions = {"In this task you will be asked to select the object which <b>best fits</b> a given description.", "An object will be described by its relation to another object which will be <color=red><b>highlighted in red</b></color>, e.g. 'the object <b>on</b> the <color=red><b>table</b></color>'. You need to <b>click</b> on the object <b>which best fits the description</b>.\n\n If you feel that <b>no object fits</b> the given description, click 'Select None'.", "The object you select will turn <color=green><b>green</b></color>. Once you have selected an object you must press 'Enter' or click 'Accept' to confirm your selection. \n\n You <b>cannot select</b> the room, floor, ceiling or walls; but remember that you <b>can select</b> the table. \n\n If you feel that <b>no object fits</b> the given description, click 'Select None'.","All important objects in the scene will be immediately in view; but remember, you can use the arrow keys to move around and while holding down the '0' key you can use the mouse to look around.\n\n Also, use the '1' and '2' keys to move up and down if you need to."};
			
		// string game_instruction_title = "Game Instructions";
		// string[] game_instructions = {};
		
		// Instantiate tasks now lists have been created
		sv_task = new Task(sv_abv,sv_instructions,sv_instruction_title);
		pq_task = new Task("pq",sv_instructions,sv_instruction_title);
		comp_task = new Task(comp_abv,comp_instructions,comp_instruction_title);
		screen_task = new Task(screen_abv,screen_instructions,screen_instruction_title);
		
		// Set instructions (this should probably happen on instatiation?)
		comp_task.instruction = "Select the object which best fits the description:\n 'the object :preposition: the :ground:'";
	
		sv_task.instruction = "Select <b>all</b> words which could fill in the blank:\n \n   ':a: :figure: (____) the :ground:'";
		pq_task.instruction = "Is the :figure: :preposition: the :ground:?";
		
		screen_task.instruction = "Select the object which best fits the description:\n 'the object :preposition: the :ground:'";
		screen_task.number_scenes_to_do = screen_task.list_of_scenes.Count;
		
		
		
		None_toggle = None_toggle_obj.GetComponent(typeof(Toggle)) as Toggle;

		//Add listener for when the state of the Toggle changes, to take action
        None_toggle.onValueChanged.AddListener(delegate {
            NoneToggleValueChanged();
        });

        foreach(Toggle t in sv_task.preposition_toggles){
        	//Add listener for when the state of the Toggle changes, to take action
	        t.onValueChanged.AddListener(delegate {
	            PrepToggleValueChanged();
	        });
        }

		// Set states
		// reset_input_values();
		confirm_text.SetActive(false);
		confirmQuit_text.SetActive(false);
		// pointer.SetActive(false);
		

	}
	/// <summary>
	/// Sets first task. Deactivates some objects. Clears PlayerPrefs and loads instructions.
	/// </summary>
	void Start () 
		{	
		
		// Set which task to begin
		task = screen_task;

		loadingImage.SetActive(false);
		clear_object_player_prefs();
		task.set_task();
		load_instructions();
		help_panel.SetActive(false);
		
		
	}

	/// <summary>
	/// Gets clean name from object name, supposing "(" is first offending character	.
	/// </summary>
	/// <param name="name">Object name.</param>
	/// <returns>
	/// Clean name.
	/// </returns>
	static public string clean_name(string name){
		string newName;
		newName = name;
		if (name.Contains("(")){
			newName = name.Substring(0,name.LastIndexOf("("));
			
		}
		if(name.Contains("_")) {
			newName = name.Substring(0,name.LastIndexOf("_"));
			
		}
		
		newName = newName.Trim();
		return newName;
	}

	/// <summary>
	/// Checks is scene is loaded.
	/// </summary>
	/// <param name="sceneName">String. Scene name.</param>
	/// <returns>
	/// True, if scene is loaded, otherwise False.
	/// </returns>
	public bool is_scene_loaded(string sceneName){
		int countLoaded = SceneManager.sceneCount;
        Scene[] loadedScenes = new Scene[countLoaded];
        string[] loadedSceneNames = new string[countLoaded];
 
        for (int i = 0; i < countLoaded; i++)
        {
            loadedScenes[i] = SceneManager.GetSceneAt(i);
            loadedSceneNames[i] = loadedScenes[i].name;
        }

        if(loadedSceneNames.Contains(sceneName)){
        	return true;
        }
        else{
        	return false;
        }
	}
	/// <summary>
	/// Checks if player is currently being tested.
	/// </summary>
	/// <returns>
	/// True, if no scenes are loaded which indicate being outside test, otherwise False.
	/// </returns>
	public bool player_in_game(){
		if(! is_scene_loaded(instruction_scene_name) && ! is_scene_loaded(fail_scene_name) && ! is_scene_loaded(finish_scene_name)){
			return true;
		}
		else{
			return false;
		}
	}

	/// <summary>
	/// Clears some playerprefs and unloads active scene.
	/// </summary>
	public void unload_current_scene(){
		hide_confirm_click();
		// Unloads the active scene
		reset_scene_values();
		Scene active_scene = SceneManager.GetActiveScene();
		Debug.Log("Unloading Scene:" + active_scene.name);
		if(active_scene.name != main_scene_name){
			SceneManager.UnloadSceneAsync(active_scene);
		}
	}

	/// <summary>
	/// Sets some playerprefs and loads scene and sets new example.
	/// </summary>
	/// <param name="sceneName">String. Scene name.</param>
	public void load_scene(string sceneName){
			loadingImage.SetActive(true);
			// takes a scene name and loads it for the task
			
			task_scene = new TaskScene(sceneName,task.name);
			
			StartCoroutine(task_scene.set_scene_coroutine());
			
			PlayerPrefs.SetString(scene_player_pref, sceneName);

			// Update scene counter.
			int number_scenes_left = task.number_scenes_to_do - number_scenes_done;
			string count = number_scenes_left.ToString();
			string newtext = SceneCountertext.Replace(":int:",count);
			SceneCounter_Text.text = newtext;

			new_example();
		
		
	}

	/// <summary>
	/// If user has done enough scenes for task, changes task.
	/// Else, unloads current scene and loads new random scene.
	/// </summary>
	public void load_next_scene(){
		
		if (number_scenes_done == task.number_scenes_to_do || task.list_of_scenes_to_do.Count==0){
			change_task();
		}

		else{
			int r = rnd.Next(task.list_of_scenes_to_do.Count);

			// unloads current scene and loads the next scene
			
			unload_current_scene();
			string new_scene = task.list_of_scenes_to_do[r];
			load_scene(new_scene);

			task.list_of_scenes_to_do.Remove(new_scene);

			number_scenes_done += 1;
		}	

	}

	/// <summary>
	/// Runs new_example_coroutine.
	/// </summary>
	public void new_example(){

		// This needs to be a coroutine as we need to wait for task_scene lists to be populated
		StartCoroutine(new_example_coroutine());
	}

	/// <summary>
	/// Coroutine. If there are remaining examples to do in the scene, sets new example.
	/// Else, changes scene.
	/// </summary>
	public IEnumerator new_example_coroutine(){
		yield return null;
		bool x = task_scene.set_new_example();
		if (x){
			task.turn_off_toggles();
			task.set_text();
			yield return new WaitForSeconds(1);
			loadingImage.SetActive(false);
			
		}
		else {
			load_next_scene();
			
		}
	}

	/// <summary>
	/// Clears playerprefs for fig, ground and preposition. Unhighlights/deselects fig and ground.
	/// </summary>
	public void reset_input_values(){
		if(task_scene != null){
			task_scene.deselect_figure();
			task_scene.deselect_ground();
		}
		
		PlayerPrefs.SetString(prep_playerpref, "");
		

	}

	/// <summary>
	/// Clears playerprefs for task, fig, ground and preposition. Unhighlights/deselects fig and ground.
	/// </summary>
	public void reset_task_values(){
		number_scenes_done = 0;
		reset_input_values();
		PlayerPrefs.SetString(task_player_pref, "");

	}

	/// <summary>
	/// Clears playerprefs for scene, fig, ground and preposition. Unhighlights/deselects fig and ground.
	/// </summary>
	public void reset_scene_values(){
		reset_input_values();
		PlayerPrefs.SetString(scene_player_pref, "");
	}
	
	/// <summary>
	/// Clears playerprefs for fig, ground.
	/// </summary>
	public void clear_object_player_prefs(){
		// Game was loading with these set to an object which was causing unhighlighting of them
		PlayerPrefs.SetString(selectedFig_playerpref, "");
		PlayerPrefs.SetString(selectedgrd_playerpref, "");
	}

	/// <summary>
	/// Gets string for authentication from username and password.
	/// </summary>
	/// <param name="username">The username.</param>
	/// <param name="password">The password.</param>
	/// <returns>
	/// Authentication string.
	/// </returns>
	string authenticate(string username, string password)
	{
	    string auth = username + ":" + password;
	    auth = System.Convert.ToBase64String(System.Text.Encoding.GetEncoding("ISO-8859-1").GetBytes(auth));
	    auth = "Basic " + auth;
	    return auth;
	}
    
    /// <summary>
    /// Writes annotation info to file.
    /// </summary>
	IEnumerator sendselectionToFile_coroutine(){
        string authorization = authenticate(auth_username, auth_password);
	    string url = appendannotation_url;
        yield return null;
        /// Output info
		string f = PlayerPrefs.GetString(selectedFig_playerpref,"");
		string g = PlayerPrefs.GetString(selectedgrd_playerpref,"");
		string p = PlayerPrefs.GetString(prep_playerpref,"");
		string ta = PlayerPrefs.GetString(task_player_pref,"");
		string u = PlayerPrefs.GetString(userid_player_pref,"");
		string sc = PlayerPrefs.GetString(scene_player_pref,"");
		string now = System.DateTime.UtcNow.ToString("yyyyMMdd-HHMMss");
		string ID = System.Guid.NewGuid().ToString();
		string prepositions = "";
		List<string> selected_prepositions =  new List<string> ();
		string all_prepositions =   "";
		foreach (Toggle t in task.list_of_toggles){
				if(t.isOn ==true){
					selected_prepositions.Add(t.name);
					prepositions  = prepositions + t.name + ";";
				}

				all_prepositions = all_prepositions + t.name + ";";

			}

		string cam_loc = task_scene.main_camera.gameObject.transform.position.ToString();
		string cam_rot = task_scene.main_camera.gameObject.transform.rotation.ToString();


        bool successful = true; //Not yet used here

        WWWForm form = new WWWForm(); //create web form to talk with
        form.AddField("selectedFigure",f);
        form.AddField("selectedGround",g);
        form.AddField("preposition",p);
        form.AddField("prepositions",prepositions);
        form.AddField("allprepositions",all_prepositions);
        form.AddField("task",ta);
        form.AddField("UserID",u);
        form.AddField("scene",sc);
        form.AddField("now",now);
        form.AddField("ID",ID);
        // form.AddField("all_objects",all_objects_string);
        form.AddField("cam_loc",cam_loc);
        form.AddField("cam_rot",cam_rot);
        
        
        // Send the form to the php script
        // Upload to a cgi script
        using (var w = UnityWebRequest.Post(url, form))
        {
        	w.SetRequestHeader("AUTHORIZATION",authorization);
            yield return w.SendWebRequest();
            if (w.isNetworkError || w.isHttpError) {
                print(w.error);
            }
            
        }
    }
	
	/// <summary>
	/// Resets some playerpref values then sets new task, unloads current scene and loads instructions.
	/// </summary>
	public void change_task(){
		Debug.Log("Changing Task"); //Add to change task button in editor
		reset_task_values();

		if (task.name == screen_abv){
			task = sv_task;
			
		}
		
		else if (task.name == sv_abv){
			task = comp_task;
			
		}

		else if (task.name == comp_abv){

			Debug.Log("Finished");
			finish();
			return;
			
		}
		unload_current_scene();
		
		load_instructions();

		task.set_task();
	}

	/// <summary>
	/// Load instruction scene.
	/// </summary>
	public void load_instructions(){
		UnityEngine.SceneManagement.SceneManager.LoadScene(instruction_scene_name,LoadSceneMode.Additive);
		
	}
	/// <summary>
	/// Unloads current scene. Resets list_of_scenes to do. Loads fail scene.
	/// </summary>
	public void fail(){
		
		unload_current_scene();
		
		// Find scenes in listofscenes not in listofscenesto do
		var result = task.list_of_scenes.Except(task.list_of_scenes_to_do); 
		foreach(string s in result){
			task.list_of_scenes_to_do.Add(s); // Add them back into list of scnes to do
		}

		number_scenes_done = 0;
		


	
		UnityEngine.SceneManagement.SceneManager.LoadScene(fail_scene_name,LoadSceneMode.Additive);
		
	}

	/// <summary>
	/// Unloads current scene. Clears fig,ground player prefs. Loads finish scene.
	/// </summary>
	public void finish(){
		clear_object_player_prefs();
		unload_current_scene();
		UnityEngine.SceneManagement.SceneManager.LoadScene(finish_scene_name);
	}

	/// <summary>
	/// If Nonetoggle is on, turns off all preposition toggles.
	/// </summary>
	void NoneToggleValueChanged(){

		if(None_toggle.isOn){
			task.turn_off_preposition_toggles();

			None_toggle.isOn = true;
		}
	
	}

	/// <summary>
	/// If any preposition toggle is on, turns off nonetoggle.
	/// </summary>
	void PrepToggleValueChanged(){
		Debug.Log("Prep change");
		if(task.preposition_toggles.Any(x => x.isOn ==true)){
			Debug.Log("Prep on");
			None_toggle.isOn = false;
		}
	}
	
	/// <summary>
	/// Handles annotation submission.
	/// </summary>
	/// <remarks>
	/// Attached to submit button.
	/// </remarks>
	public void submit(){
		if(task.name == screen_abv){
			string f = PlayerPrefs.GetString(selectedFig_playerpref,"");
			GameObject fig = task_scene.active_configuration[0];
			
			// Check selection is correct.
			if (f==fig.name){

				/// Set new example
				new_example();
			}

			else{
				
				fail();
			}
		}
		else if(task.name == sv_abv){
			if(task.list_of_toggles.All(x => x.isOn ==false)){

			}
			else{
				StartCoroutine(sendselectionToFile_coroutine());

				// Set new example.
				new_example();
			}
		}
		else{
			StartCoroutine(sendselectionToFile_coroutine());

			/// Set new example.
			new_example();
		}
	}
	
	/// <summary>
	/// Sets object as figure for associated raycast object.
	/// </summary>
	void click_figure(RaycastHit fig){
		task_scene.set_figure(fig.transform.gameObject);
		
	}

	
	/// <summary>
	/// Shows confirm click dialogue.
	/// </summary>
	void show_confirm_click(){
		confirm = true;
		confirm_text.SetActive(true);
	}

	/// <summary>
	/// Hides confirm click dialogue.
	/// </summary>
	void hide_confirm_click(){
		confirm = false;
		confirm_text.SetActive(false);
	}

	/// <summary>
	/// If confirm click dialogue is shown, hides confirm click dialogue and submits annotation.
	/// </summary>
	/// <remarks>
	/// Attached to accept button.
	/// </remarks>
	public void accept(){
		if(confirm){
		  	hide_confirm_click();
		  	submit();
		}
		
	}

	/// <summary>
	/// Handles user clicking select none.
	/// </summary>
	/// <remarks>
	/// Attached to select none button.
	/// </remarks>
	public void select_none(){
		task_scene.deselect_figure();
		hide_confirm_click();
		submit();
	}

	/// <summary>
	/// Handles user what to do when user clicks or touches screen.
	/// </summary>
	void handle_click_touch(){
			
		// Deselect old figure object.
		task_scene.deselect_figure();
		hide_confirm_click();
		

		// Find hit object.
	    Ray ray = task_scene.main_camera.ScreenPointToRay(Input.mousePosition);
	    RaycastHit hit;
	  	// If something is hit.
		if (Physics.Raycast(ray, out hit)){
		    // The object identified by hit.transform was clicked.
		  	GameObject g;
		  	// Get current ground object from task_scene.
		  	if (task.name == screen_abv){
			  	g = task_scene.active_configuration[1];
			}
			else{
				g = task_scene.active_comparison[0] as GameObject;
			}
			  
			// If hit.transform is a selectable object, set figure and show confirm click.
		  	if (hit.transform.name != g.name && !unselectable_scene_objects.Any(x => hit.transform.name.Contains(x))){
		  		task.selected_figure_panel_text_component.text = "Selected Object: " + "<b>" + hit.transform.name + "</b>";
			  	click_figure(hit);
			  	show_confirm_click();
			}

			else{
				task.selected_figure_panel_text_component.text = "Selected Object: ";
			}
	  	}
	}

	/// <summary>
	/// Toggles displaying help panel.
	/// </summary>
	void toggle_help_panel(){
		if (Input.GetKeyDown (ShowHelpKey)){
			if(player_in_game()){
				if(!help_panel.activeSelf){
					help_panel.SetActive(true);
				}
				else{
					help_panel.SetActive(false);
				}
			}
		}
	}

	/// <summary>
	/// Checks if user is pressing quit key and takes action.
	/// </summary>
	void quit_input(){
		
		if(Input.GetKeyDown (quitKey)) {
		    confirmQuit_text.SetActive(true);
		    confirm_quit = true;

		}
	}


	/// <summary>
	/// Update is called once per frame.
	/// </summary>
	void Update(){
		// Show/hide help panel on key press.
		toggle_help_panel();

		quit_input();

		// Finish if user confirms quit.
		if (Input.GetKeyDown (KeyCode.Return) && confirm_quit == true){
		  	finish();
		  	#if UNITY_EDITOR
	        //Stop playing the scene
	        UnityEditor.EditorApplication.isPlaying = false;
		    #endif
		}

		// Remove panels if user clicks.
	    if (Input.GetMouseButtonDown(0)){
	    	confirmQuit_text.SetActive(false);
		    confirm_quit = false;

		    help_panel.SetActive(false);
	    }

	    // When player is in testing scenario..
	    // Handle clicks and "return" presses.
		if(player_in_game()){
			
			if (task.name == comp_abv || task.name == screen_abv){
				// If left mouse button pressed...
				if (Input.GetMouseButtonDown(0)){ 
					if (Input.touchCount > 0 && Input.GetTouch(0).phase == TouchPhase.Began){
			        	Debug.Log("touch");
			            // Check if finger is over a UI element
			            if (!EventSystem.current.IsPointerOverGameObject(Input.GetTouch(0).fingerId))
			            {
			                handle_click_touch();
			            }
			        }
					else if (!EventSystem.current.IsPointerOverGameObject()){
						
			            Debug.Log("No touch and not over UI");
		                handle_click_touch();
				        
				    }

				}
				// If return is pressed.
				if (Input.GetKeyDown (KeyCode.Return)){
			  		accept();
				}
			}
		}
	}
}

	   
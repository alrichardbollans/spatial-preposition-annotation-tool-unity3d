// Script added to mainEmpty in main scene.
// Must be called "Main.cs" and stored in "Assets/Scripts".

// This scene is open during all data collection with other scenes loaded on top. 
// In this way the 'Main' instance is never destroyed

// To Do:
// - Separate different tasks into different classes which inherit from Task.

using System;
using System.IO;
using System.Text;
using System.Linq;
using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using UnityEngine.EventSystems;




public class Config  
{  
     public string scene;  
     public string figure;
     public string ground;

     public Config(string scn, string fig,string gr){
     	scene= scn;
     	figure = fig;
     	ground = gr;
     	
     }
}   

/// <summary>
/// TaskScene class acts like the usual Scene object except 
/// more information is stored regarding object configurations and examples to test.
/// </summary>
public class TaskScene {

	public string name; //Scene Name
	public string task_type; 
	Main main; // Main instance.
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
	// Preposition list for typ task.
	List<string> typ_preposition_list = new List<string> {"on","on top of", "in", "inside","against","over","below","above","under"};

	static Material fig_mat = Resources.Load("figure_material", typeof(Material)) as Material;
	static Material grd_mat = Resources.Load("ground_material", typeof(Material)) as Material;

	// Stores materials to undo highlighting.
	static Material[] stored_fig_mats;
	static Material[] stored_grd_mats;

	// Dictionary of images for each preposition to use in typ_task.
	Dictionary<string,List<Texture2D>> typicality_images;
	// Pairs of images to display.
	Dictionary<string,List<List<Texture2D>>> typicality_image_pairs;

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
		// Get main game script
		GameObject main_empty;
	
		main_empty = GameObject.Find(Main.maingameobject);
		if(main_empty !=null){
			main = main_empty.GetComponent<Main>();
		}
		
	}

	/// <summary>
    /// Stores figure name in playerprefs and highlights figure.
    /// </summary>
    /// <param name="fig">Figure to set.</param>
	public void set_figure(GameObject fig){ 
		
		PlayerPrefs.SetString(selectedFig_playerpref, fig.name);

		highlight_figure(fig);
		// Debug.Log("new figure:");
		// Debug.Log(fig.name);
		
		
	}



	/// <summary>
    /// Stores ground name in playerprefs and highlights ground.
    /// </summary>
    /// <param name="gr">Ground to set.</param>
	public void set_ground(GameObject gr){
		
		PlayerPrefs.SetString(selectedgrd_playerpref, gr.name);
		highlight_ground(gr);
		// Debug.Log("new ground:");
		// Debug.Log(gr.name);
		
		
		
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
		// Debug.Log(old_figure_name);
		// Note find objects can be heavy process (see docs if needs calling every frame)
		GameObject old_figure = GameObject.Find(old_figure_name);
		//If there was actually a figure, undo highlighting
		if (old_figure != null){
			// Debug.Log("Unhighlighting figure: " + old_figure_name);
			unhighlight_figure(old_figure);
			
		}
		
		PlayerPrefs.SetString(selectedFig_playerpref, "");
		
	}

	
	/// <summary>
    /// Unhighlights ground and updates player prefs.
    /// </summary>	
	public void deselect_ground(){
		
		// Debug.Log("Deselect ground is called");
		string old_grd_name = PlayerPrefs.GetString(selectedgrd_playerpref,"");
		// Debug.Log("old_grd_name is " + old_grd_name);
		// Note find objects can be heavy process (see docs if needs calling every frame)
		GameObject old_grd = GameObject.Find(old_grd_name);
		//If there was actually a ground, undo highlighting
		if (old_grd != null){
			// Debug.Log("Unhighlighting ground: " + old_grd_name);

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
	static public void highlight_figure(GameObject fig){
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
	static public void highlight_ground(GameObject grd){
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
	public void populate_fig_ground_list(){
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
	public void set_main_camera(){
		// Set camera
		cam_list = GameObject.FindGameObjectsWithTag(Main.main_camera_tag);
		foreach(GameObject c in cam_list){

			if (c.scene.name == name){
				main_camera = c.GetComponent<Camera>();
			}
		}
	}

	public void populate_config_list(){
		if (task_type == Main.sv_abv || task_type == Main.sv_mod_abv || task_type == "pq" || task_type == Main.screen_abv){
			
			foreach (GameObject ground in ground_list){
				foreach(GameObject fig in figure_list){
					if(fig.name != ground.name){
						GameObject[] config = {fig,ground};
						configuration_list.Add(config);
					}
				}

				if(task_type == Main.screen_abv){
					// Add configurations to list by taking children of ground
					foreach(Transform emp in ground.transform){
						string p = emp.gameObject.tag;
						if (comp_preposition_list.Contains(p)){
							screening_preposition = p;
							
						}
					}
				}
				
			}
			
		}
		if(task_type == Main.typ_abv){
			// Get images for each preposition.
			foreach(string prep in typ_preposition_list){
				try{
					Texture2D[] typ_im_array =Resources.LoadAll<Texture2D>("Typ_task_folder/"+prep);
					typicality_images[prep] = typ_im_array.ToList();
				}
				catch(Exception e){
					Debug.Log("No folder for '" + prep + "' images.");
				}
				
			}
			// Images are shared for some prepositions.
			typicality_images["inside"] = typicality_images["in"];
			typicality_images["above"] = typicality_images["over"];
			typicality_images["below"] = typicality_images["under"];
			typicality_images["on top of"] = typicality_images["on"];
			
			// Now shuffle the lists.
			foreach(string prep in typ_preposition_list){
				typicality_images[prep] = typicality_images[prep].OrderBy(a => rnd.Next()).ToList();
				typicality_image_pairs[prep] = new List<List<Texture2D>>() {};
			}

			
			// Now go through lists and create pairs for task.
			foreach(string prep in typ_preposition_list){
				foreach(Texture2D img1 in typicality_images[prep]){
					Config c1 = get_config_from_img(img1);
					foreach(Texture2D img2 in typicality_images[prep]){
						Config c2 = get_config_from_img(img2);
						if(c1.scene != c2.scene){
							if(!typicality_image_pairs[prep].Any(img_pair => is_there_a_pair_match(c1, c2,img_pair))){
								List<Texture2D> new_pair = new List<Texture2D>() {img1,img2};
								typicality_image_pairs[prep].Add(new_pair);
							}
						}
						
						
					}
				}
			}
		}
		
		
	}

	/// <summary>
    /// Checks if two configs match an image pair by scenes.
    /// </summary>
	bool is_there_a_pair_match(Config c1, Config c2,List<Texture2D> img_pair){
		Config ic1 = get_config_from_img(img_pair[0]);
		Config ic2 = get_config_from_img(img_pair[1]);
		if(c1.scene == ic1.scene && c2.scene == ic2.scene){
			return true;
		}
		if(c2.scene == ic1.scene && c1.scene == ic2.scene){
			return true;
		}

		return false;
	}

	/// <summary>
    /// Once scene has been loaded, populates various list/configurations and set main camera.
    /// </summary>
	public void instantiate_after_scene_loaded(){
		Debug.Log("Instantiating task scene");
		// Set camera
		set_main_camera();
		populate_fig_ground_list();
		// Add in configurations
		populate_config_list();

		if (task_type == Main.comp_abv){
			
			
			
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
    /// Loads scene and finds configurations to test.
    /// </summary>
	public IEnumerator set_scene_coroutine(){
		
		Scene scene_to_load = SceneManager.GetSceneByName(name);
		if(!scene_to_load.isLoaded){
			// Needs to be set up as a coroutine so that it only continues after scene is fully loaded
			SceneManager.LoadScene(name,LoadSceneMode.Additive);
			yield return null;
			SceneManager.SetActiveScene(SceneManager.GetSceneByName(name));
		}
		// Update playerprefs.
		PlayerPrefs.SetString(Main.scene_player_pref, name);
			
		
		// Populate various lists.
		instantiate_after_scene_loaded();
			
		
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

	public static string ScreenShotName(string scene, string figure, string ground,string number) {
	    return string.Format("typtask_scene_{0}__figure_{1}__ground_{2}_.png", 
	                         scene, figure, ground);
	}

	public static Config get_config_from_img(Texture2D img){
		string old = img.name;
		string fig;
		string grd;
		string scene;
		
		int first_scene_character_index = old.IndexOf("scene_") +6;
		int first_fig_character_index = old.IndexOf("figure_") +7;
		int first_gr_character_index = old.IndexOf("ground_") +7;
		
		fig = get_string_from_img_file(old,first_fig_character_index);
		scene = get_string_from_img_file(old,first_scene_character_index);
		grd = get_string_from_img_file(old,first_gr_character_index);
		
		Config c = new Config(scene,fig,grd);

		return c;
	
	}

	static string get_string_from_img_file(string file_string,int first_ch_index){
		string out_string;
		int end = file_string.Length - first_ch_index;
		out_string = file_string.Substring(first_ch_index,end);
		
		out_string = out_string.Substring(0,out_string.IndexOf("_"));
		
		return out_string;
	}

	bool show_new_config_pictures(){
		// Set preposition.
		int r = rnd.Next(typ_preposition_list.Count);
		string p = typ_preposition_list[r];
		// If there are no images to test get another preposition.
		while(typicality_image_pairs[p].Count == 0){
			typ_preposition_list.Remove(p);
			// If there are no more prepositions return false.
			if(typ_preposition_list.Count == 0){
				return false;
			}
			else{
				
				r = rnd.Next(typ_preposition_list.Count);
				p = typ_preposition_list[r];
			}
			
		}
		// Set preposition.
		set_preposition(p);

		// Pick an image pair for the preposition.
		int i = rnd.Next(typicality_image_pairs[p].Count);
		List<Texture2D> img_pair = typicality_image_pairs[p][i];
		
		Texture2D img1 = img_pair[0];
		Texture2D img2 = img_pair[1];

		// Set the player prefs and left object texture.
		PlayerPrefs.SetString(Main.config1_player_pref, img1.name);
		main.typ_left_image.GetComponent<RawImage>().texture = img1;
		// Set the player prefs and right object texture.
		PlayerPrefs.SetString(Main.config2_player_pref, img2.name);
		main.typ_right_image.GetComponent<RawImage>().texture = img2;

		// Remove pair from testing.
		typicality_image_pairs[p].Remove(img_pair);
		
		return true;
	}
	/// <summary>
    /// Sets new configuration to test.
    /// </summary>
    /// <returns>
	/// true if a new configuration can be set in the scene, otherwise False.
	/// </returns>
	public bool set_new_example(){
		if (task_type == Main.sv_abv || task_type == Main.sv_mod_abv || task_type == "pq" || task_type == Main.screen_abv){


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


		
		else if (task_type == Main.comp_abv){
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

		else if (task_type== Main.typ_abv){
			if(Main.number_typ_configs_done<Main.number_typ_configs_to_do){
				Main.number_typ_configs_done +=1;
				bool x = show_new_config_pictures();
				return x;
				

				// int left_to_do = 1 + Main.number_typ_configs_to_do - Main.number_typ_configs_done;
				// string newtext = main.SceneCountertext.Replace(":int:",left_to_do.ToString());
				// main.SceneCounter_Text.text = newtext;

				
			}
			else{
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
    public static string[] input_list_of_scenes = {"finish","instruction","main","player_menu","scene_template","screen0","screen1","screening_fail","sv_modtypa1","sv_modtypa2","sv_modtypi1","sv_modtypi2","sv_modtypi3","sv_modtypi4","sv_modtypo1","sv_modtypo2","sv_modtypo3","sv_modtypo4","sv_modtypov1","sv_modtypov2","sv_modtypov3","sv_modtypu1","sv_modtypu2","sv_modtypu3","sv_modtypu4","test"};
	//
	public string name;
	Main main;
	public string instruction; //Instruction to give in each scene

	string new_instruction;  //Instruction to give in each scene
	//task name abbreviations with shared scenes
	List<string> scene_abbreviations =  new List<string>(); 

	public string[] instruction_list; // List of instructions to give before starting
	public string instruction_title; //Title for instruction scene
	
	public GameObject panel;
	public Text selected_figure_text;
	public Text instruction_text_component;
	List<GameObject> task_panels=  new List<GameObject>();


	public List<GameObject> active_objects =  new List<GameObject>(); // list of all objects in panel hieracrchy
	public List<Toggle> list_of_toggles = new List<Toggle> (); // Toggles for selecting prepositions
	public List<Toggle> preposition_toggles = new List<Toggle> ();
	
	public List<string> list_of_scenes = new List<string> (); // List of all scenes doesn't get chanegd
	public List<string> list_of_scenes_to_do = new List<string> (); // List of scenes where done scenes are removed 
	public int number_scenes_to_do=10;

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
	/// <param name="m">Main game manager instance.</param>
	public Task(string n,Main m){

		name = n;
		main = m;

		scene_abbreviations.Add(n);
		get_scenes();
		
		selected_figure_text = main.selected_fig_text;

		// Populate task_panels list.		
		task_panels.Add(main.sv_main_panel);
		task_panels.Add(main.comp_main_panel);
		task_panels.Add(main.typ_main_panel);
		
		if(name==Main.sv_abv){
			panel = main.sv_main_panel;
			instruction_text_component = main.sv_instruction_text;
			number_scenes_to_do = 10;
			instruction_title = "Instructions";
			string[] il = {"In this task you will be shown some objects and asked to select words which could <b>describe the relationship between them</b>.",
			"A <b>pair</b> of objects will be highlighted, <b>one in <color=green>green</color></b> and <b>the other in <color=red>red</color></b>. You need to select <b>all</b> the words which describe <b>how the <color=green>green object</color> relates to the <color=red>red object</color></b>.",
			"The words you may select are: 'on', 'on top of', 'in', 'inside', 'against', 'over', 'under', 'above' and 'below'. \n\n If none of the given words apply, select <b> 'None of the above'</b>.\n\n Once you have made your selections, click 'Submit'. A new pair and/or scene will then be displayed.",
			"Remember, you can use the arrow keys to move around and while holding down the '0' key you can use the mouse to look around.\n\n Also, use the '1' and '2' keys to move up and down if you need to."};
			instruction_list = il;
			
			instruction = "Select <b>all</b> words which could fill in the blank:\n \n   ':a: :figure: (____) the :ground:'";

		}

		if(name == Main.sv_mod_abv){
			panel = main.sv_main_panel;
			instruction_text_component = main.sv_instruction_text;
			number_scenes_to_do = 10;
			instruction_title = "Instructions";
			string[] il = {"In this task you will be shown some objects and asked to select words which could <b>describe the relationship between them</b>.",
			"A <b>pair</b> of objects will be highlighted, <b>one in <color=green>green</color></b> and <b>the other in <color=red>red</color></b>. You need to select <b>all</b> the words which describe <b>how the <color=green>green object</color> relates to the <color=red>red object</color></b>.",
			"The words you may select are: 'on', 'on top of', 'in', 'inside', 'against', 'over', 'under', 'above' and 'below'. \n\n If none of the given words apply, select <b> 'None of the above'</b>.\n\n Once you have made your selections, click 'Submit'. A new pair and/or scene will then be displayed.",
			};
			instruction_list = il;
			
			instruction = "Select <b>all</b> words which could fill in the blank:\n \n   'a <color=green><b>green object</b></color> (____) the <color=red><b>red object</b></color>'";

		}
		if(name==Main.comp_abv){
			panel = main.comp_main_panel;
			number_scenes_to_do = 10;
			instruction_text_component = main.comp_instruction_text;
			string[] il = {"In this task you will be asked to select the object which <b>best fits</b> a given description.", "An object will be described by its relation to another object which will be <color=red><b>highlighted in red</b></color>, e.g. 'the object <b>on</b> the <color=red><b>table</b></color>'. You need to <b>click</b> on the object <b>which best fits the description</b>.\n\n If you feel that <b>no object fits</b> the given description, click 'Select None'.", "The object you select will turn <color=green><b>green</b></color>. Once you have selected an object you must press 'Enter' or click 'Accept' to confirm your selection. \n\n You <b>cannot select</b> the room, floor, ceiling or walls; but remember that you <b>can select</b> the table. \n\n If you feel that <b>no object fits</b> the given description, click 'Select None'.","All important objects in the scene will be immediately in view; but remember, you can use the arrow keys to move around and while holding down the '0' key you can use the mouse to look around.\n\n Also, use the '1' and '2' keys to move up and down if you need to."};
			instruction_list = il;
			instruction_title = "Instructions";
			instruction = "Select the object which best fits the description:\n 'the object :preposition: the :ground:'";
		

		}
		if(name==Main.screen_abv){
			panel = main.comp_main_panel;
			instruction_text_component = main.comp_instruction_text;
			string[] il = {"Before beginning you will be given <b>two quick examples</b> to complete\n \n \nClick Next..",	"You will be shown an indoor scene and a description of an object will be provided at the bottom of the screen. \n \n Click on the object that best fits the description.\n \n You will be prompted to press enter or click accept to confirm your selection. \n \nIf you are correct you will move on to the next stage.",
			"To move around the scene: \n - Use the <b>arrow keys</b> to move around \n - <b>Hold down the '0' key</b> to use the mouse to look around \n - Use the <b>'1' and '2' keys</b> to move up and down if you need to \n - Press the <b>'Delete' key</b> for help"	};
			instruction_list = il;
			instruction_title = "Instructions";
			instruction = "Select the object which best fits the description:\n 'the object :preposition: the :ground:'";
			number_scenes_to_do = list_of_scenes.Count;
			

		}
		
		if(name==Main.typ_abv){
			panel = main.typ_main_panel;
			instruction_text_component = main.typ_instruction_text;
			string[] il = {"In this task you will be shown two configurations of objects and asked to select which configuration <b>best fits</b> a given description.",
			"A simple description will be given of a green object and its relationship to a red object, e.g. 'the <color=green><b>green object</b></color> <b>on</b> the <color=red><b>red object</b></color>'. You need to <b>click</b> the image <b>which best fits the description</b>.\n\n If you feel that <b>no image fits</b> the given description, click 'Select None'."};
			instruction_list = il;
			instruction_title = "Instructions";
			instruction = "Select the pair of objects which best fits the description:\n'a <color=green><b>green object</b></color> :preposition: the <color=red><b>red object</b></color>'";
			

		}

		// if(name == Main.sv_abv || name == Main.comp_abv || name == "pq"){
		// 	//These tasks share scenes
		// 	scene_abbreviations.Add(Main.sv_abv);
		// 	scene_abbreviations.Add("pq");
		// 	scene_abbreviations.Add(Main.comp_abv);

		// }

		// Populate list of active objects by what's in panel hierarchy
		add_all_descendants(panel);

		// Assign various objects
		foreach (GameObject g in active_objects){

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
	/// Add all descendants to active_objects.
	/// </summary>
	void add_all_descendants(GameObject go){
		foreach (Transform obj in go.transform){
			
			active_objects.Add(obj.gameObject);
			add_all_descendants(obj.gameObject);
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
			g.SetActive(false);
			
		}

		panel.SetActive(true);
		foreach (GameObject g in active_objects){
			g.SetActive(true);
		}

		// Hide info panel and stop camera movement for some tasks.
		if(name == Main.typ_abv || name == Main.sv_mod_abv){
			main.general_info_panel.SetActive(false);
			Main.allow_camera_movement = false;

		}
		else{
			main.general_info_panel.SetActive(true);
			
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
		selected_figure_text.text = "Selected Object: ";

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
	// Specific to typ_task.
	public static string config1_player_pref = "config1";
	public static string config2_player_pref = "config2";
	public static string selection_player_pref = "config_selection";
	



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

	// Object names
	public static string maingameobject = "mainEmpty";

	// Store names of any scenes that shouldn't be included in build.
	public static List<string> non_test_scenes = new List<string> {"example", "scene_template", "test"};
	public static List<string> unselectable_scene_objects = new List<string> {"wall","floor","ceiling"};

	// Directory Info
	public static string MainFolder   = "Assets/Scenes";
	
	// Server strings
	public static string my_url = "http://adamrichard-bollans.co.uk";
    public static string auth_username = "game";
    public static string auth_password =  "REDACTED";
	public static string appendannotation_url = my_url+"/spatial_language_study/appendannotation.php";
	public static string writeuserdata_url = my_url+"/spatial_language_study/writeuserdata.php";

	// Task Abbreviations
	public static string sv_abv = "sv";
	public static string sv_mod_abv = "sv_mod";
	public static string comp_abv = "comp";
	public static string screen_abv = "screen";
	public static string typ_abv ="typ"; // Check this string is the same in php script.

	

	// Input keys
	static public KeyCode ShowHelpKey = KeyCode.Delete;
	static public KeyCode quitKey = KeyCode.Escape;

	Task sv_task;
	Task sv_mod_task;
	Task comp_task;
	Task screen_task;
	Task typ_task;
	
	static public Task task;

	TaskScene task_scene;

	// Gameobjects to assign
	public GameObject typ_main_panel;
	public GameObject comp_main_panel;
	public GameObject sv_main_panel;
	public Text selected_fig_text;
	public Text comp_instruction_text;
	public Text sv_instruction_text;
	public Text typ_instruction_text;
	public GameObject typ_left_image;
	public GameObject typ_right_image;
	public GameObject confirm_text;
	public GameObject confirmQuit_text;
	public GameObject help_panel;
	public GameObject general_info_panel;

	public GameObject None_toggle_obj;
	Toggle None_toggle;
	
	public GameObject loadingImage;
	public Text SceneCounter_Text;
	public string SceneCountertext = "Scenes left: :int:";
	
	int number_scenes_done = 0;
	
	static public int number_typ_configs_done = 0;
	static public int number_typ_configs_to_do = 10;

	public static bool allow_camera_movement = true;

	// Objetcs to hide/show if in dev mode.
	bool dev_mode = false;
	public GameObject dev_panel;
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
	/// Populate dev_objects list.
	/// Creates tasks. Adds listeners to toggles. Deactivates some objects.
	/// </summary>
	void Awake(){
		//  Show/hide dev objects.
		if(dev_mode){
			dev_panel.SetActive(true);
		}
		else{
			dev_panel.SetActive(false);
		}
	
		if(dev_mode){
			Debug.Log("Warning: In developer mode.");
		}
		
		// Get list of all game objects
		allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		all_objects_string = "";
		foreach(GameObject obj in allObjects){
			all_objects.Add(obj.name);
			all_objects_string += obj.name + ",";
		}
		

		// Instantiate tasks now lists have been created
		sv_task = new Task(sv_abv,this);
		sv_mod_task = new Task(sv_mod_abv,this);
		comp_task = new Task(comp_abv,this);
		screen_task = new Task(screen_abv,this);
		typ_task = new Task(typ_abv,this);
	
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
		task = typ_task;

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

		// Update scene counter.
		// Update this to do typ_task.
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
		
		if (number_scenes_done == task.number_scenes_to_do){
			Debug.Log(number_scenes_done);

			Debug.Log("change because of scene number");
			change_task();
			 
		}
		if (task.list_of_scenes_to_do.Count==0){
		
			Debug.Log("change because no scene to do");
			change_task();
		}

		else if (number_typ_configs_done >= number_typ_configs_to_do){
			Debug.Log("change because of typ_task");
			change_task();
		}

		else if(task == typ_task){
			unload_current_scene();
			// Only uses on scene camera:main.
			task_scene = new TaskScene(main_scene_name,task.name);
			StartCoroutine(task_scene.set_scene_coroutine());
			new_example();
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
        string c1 = PlayerPrefs.GetString(config1_player_pref,"");
        string c2 = PlayerPrefs.GetString(config2_player_pref,"");
        string selection = PlayerPrefs.GetString(selection_player_pref,"");

        string p = PlayerPrefs.GetString(prep_playerpref,"");
		
		string f = PlayerPrefs.GetString(selectedFig_playerpref,"");
		string g = PlayerPrefs.GetString(selectedgrd_playerpref,"");
		string ta = PlayerPrefs.GetString(task_player_pref,"");
		string u = PlayerPrefs.GetString(userid_player_pref,"");
		string sc = PlayerPrefs.GetString(scene_player_pref,"");
		string now = System.DateTime.UtcNow.ToString("yyyyMMdd-HHMMss");
		string ID = System.Guid.NewGuid().ToString();
		string prepositions = "";
		// Find selected prepositions for sv task.
		List<string> selected_prepositions =  new List<string> ();
		string all_prepositions =   "";
		foreach (Toggle t in task.list_of_toggles){
			if(t.isOn ==true){
				selected_prepositions.Add(t.name);
				prepositions  = prepositions + t.name + ";";
			}

			all_prepositions = all_prepositions + t.name + ";";

		}

	
			

        bool successful = true; //Not yet used here

        //create web form to talk with.
        WWWForm form = new WWWForm(); 
        // Added to all annotations.
        form.AddField("now",now);
        form.AddField("ID",ID);
        form.AddField("task",ta);
        form.AddField("UserID",u);
        form.AddField("preposition",p);

		Debug.Log("outputting");
        if(task.name == typ_abv){

        	Debug.Log(c1);
        	Debug.Log(c2);
        	form.AddField("c1",c1);
        	form.AddField("c2",c2);
        	form.AddField("selection",selection);
        	
        }
        else{
        	string cam_loc = task_scene.main_camera.gameObject.transform.position.ToString();
        	string cam_rot = task_scene.main_camera.gameObject.transform.rotation.ToString();
        	
        	Debug.Log(f);
        	// Not included in typ_task.
        	form.AddField("selectedFigure",f);
        	Debug.Log(form.data);
        	form.AddField("selectedGround",g);
        	form.AddField("prepositions",prepositions);
        


	        form.AddField("scene",sc);

	        // form.AddField("all_objects",all_objects_string);
	        form.AddField("cam_loc",cam_loc);
	        form.AddField("cam_rot",cam_rot);
	    }
        
        
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
			task = typ_task;
			
		}
		
		else if (task.name == typ_abv){
			task = sv_mod_task;
			
		}

		else if (task.name == sv_mod_abv){

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
		// Debug.Log("Prep change");
		if(task.preposition_toggles.Any(x => x.isOn ==true)){
			// Debug.Log("Prep on");
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
		else if(task.name == sv_abv || task.name == sv_mod_abv){
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

	public void left_image_click(){
		string c1 = PlayerPrefs.GetString(config1_player_pref,"");
		PlayerPrefs.SetString(selection_player_pref,c1);
		confirm = true;
		accept();
	}

	public void right_image_click(){
		string c2 = PlayerPrefs.GetString(config2_player_pref,"");
		PlayerPrefs.SetString(selection_player_pref,c2);
		confirm = true;
		accept();
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
		reset_input_values();
		hide_confirm_click();
		submit();
	}

	/// <summary>
	/// Clears playerprefs and selection/highlighting of objects the user inputs.
	/// </summary>
	public void reset_input_values(){
		if(task_scene != null){

			if(task == comp_task){
				task_scene.deselect_figure();
			
			}	
			
		}
		if(task == typ_task){
			PlayerPrefs.SetString(selection_player_pref, "");

		}
		
		if(task == sv_task){
			task.turn_off_preposition_toggles();
			
		}


	}

	/// <summary>
	/// Clears playerprefs for task, fig, ground and preposition. Unhighlights/deselects fig and ground.
	/// </summary>
	public void reset_task_values(){
		number_scenes_done = 0;
		number_typ_configs_done = 0;
		reset_input_values();
		PlayerPrefs.SetString(task_player_pref, "");
		PlayerPrefs.SetString(prep_playerpref, "");

	}

	/// <summary>
	/// Clears playerprefs for scene, fig, ground and preposition. Unhighlights/deselects fig and ground.
	/// </summary>
	public void reset_scene_values(){
		clear_any_object_selections();
		reset_input_values();
		PlayerPrefs.SetString(scene_player_pref, "");
	}
	
	/// <summary>
	/// Clears playerprefs and highlighting for fig, ground.
	/// </summary>
	public void clear_any_object_selections(){
		if(task_scene != null){
			task_scene.deselect_figure();
			task_scene.deselect_ground();
		}
		clear_object_player_prefs();

	}

	/// <summary>
	/// Clears playerprefs for fig, ground.
	/// </summary>
	static public void clear_object_player_prefs(){
		// Game was loading with these set to an object which was causing unhighlighting of them
		PlayerPrefs.SetString(selectedFig_playerpref, "");
		PlayerPrefs.SetString(selectedgrd_playerpref, "");
		PlayerPrefs.SetString(selection_player_pref, "");
	}

	/// <summary>
	/// Handles user what to do when user clicks or touches screen.
	/// </summary>
	void handle_click_touch(){
		Debug.Log("click");

		

		// Find hit object.
	    Ray ray = task_scene.main_camera.ScreenPointToRay(Input.mousePosition);
	    RaycastHit hit;
		// Deselect old figure object.
		task_scene.deselect_figure();
		hide_confirm_click();
		

		
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
		  		task.selected_figure_text.text = "Selected Object: " + "<b>" + hit.transform.name + "</b>";
			  	click_figure(hit);
			  	show_confirm_click();
			}

			else{
				task.selected_figure_text.text = "Selected Object: ";
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
				

				if (Input.GetMouseButtonDown(0)){
					// If left mouse button pressed...
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
				
			}
			// If return is pressed.
			if (Input.GetKeyDown (KeyCode.Return)){
		  		accept();
			}

			
		}
	}
}

	   
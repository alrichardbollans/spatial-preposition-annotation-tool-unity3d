// This script gives unity button which runs functional feature calculations

// The actual calculations are done in a runtime script WaitingGame_F....cs

// It does this by adding/removing colliders, giving the objects rigidbody, playing the scene
// writing positions (using an added in-game script) and then reading the positions and editing transforms

/// Makes use of csv files to store important variables as going between run time and editor resets variables stored by some classes :(
#if UNITY_EDITOR
using UnityEngine;

using UnityEditor;
using UnityEditor.SceneManagement;

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;




//Initialize on load means update is constantly running
[InitializeOnLoad]
public class write_support_measures : EditorWindow{
	// Lists of gameobjects and mesh objects
	static GameObject[] allObjects;
	static List<MeshObject> meshObjects = new List<MeshObject>();

	// Collects heights of objects, used in calculations? Not sure necessary anymore
	static public Dictionary<string,float> obj_height_dictionary = new  Dictionary<string,float>();

	// this gets edited by an in-game script to indicate if calculations have been written
	static public bool written = false; 
	
	// List of scenes in project
	static public List<MyScene> scene_list = new List<MyScene>();
	// Scene index for loading scene (is written and stored in csv)
	static public int scene_index = 0;

	static public List<GameObject> ground_list = new List<GameObject>();
	static  public string output_path = "Scene Data/";
	static public string calculations_file = "support_measures.csv";
	static public string csv_file = output_path + calculations_file;

	// For location control calculations in x dimension
	static public string location_control_calculations_file1 = "location_control_measures1.csv";
	static public string lc1_csv_file = output_path + location_control_calculations_file1;
	// For location control calculations in z dimension
	static public string location_control_calculations_file2 = "location_control_measures2.csv";
	static public string lc2_csv_file = output_path + location_control_calculations_file2;
	
	// Files used to store information while running, for in-game script to read
	static public string scene_index_file = ".sceneindex.csv";

	static public string ground_heights_file = ".ground_heights.csv";

	static public string groundstodo_file = ".groundstodo.csv";

	// This tells waiting_game_FF script which feature to calculate. Gets updated in waiting_game_FF script
	static public string type = "support";

	/// Makes current Update function be run constantly
	// Commenting out will allow testing on a single scene
	static write_support_measures(){
		// EditorApplication.update += Update;
	}
    

    static float get_object_height(GameObject go){
		// Adds a box collider to work out highest and lowest points
        var meshcoll = go.AddComponent<BoxCollider>();
        
        float x = meshcoll.bounds.max.y - meshcoll.bounds.min.y;

        return x;
    }
	
	static public void set_grounds_to_do(){
		
        // Write grounds down
        using(StreamWriter file = new StreamWriter(groundstodo_file)){
        	List<String> row = new List<String>();
			foreach(GameObject obj in ground_list){
			
				if(!obj.name.Contains("wall") && !obj.name.Contains("ceiling") && !obj.name.Contains("floor")){
					row.Add(obj.name);
				}
				

				
			}
			string[] row1 = row.ToArray();
			string row_csv_string = String.Join(",", row1);
			file.WriteLine(row_csv_string);
		}
	}
	
	static public void main(){
		// Called each time scene is re/loaded
		Debug.Log("Check initialiseonload is uncommented");
		written = false;
		meshObjects.Clear();
		// Populate object list
		allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		// Populate mesh list
		foreach(GameObject obj in allObjects){
            MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
            if(mf != null){
            	MeshObject mobj = new MeshObject(obj);
                meshObjects.Add(mobj);
                ground_list.Add(obj);
                mobj.remove_scripts();
            }
        }
        
        // Begin by getting object physical properties
        using(StreamWriter file = new StreamWriter(ground_heights_file)){
	        foreach(MeshObject obj in meshObjects){
	        		
	        		
	        		
			        

			        // Adds a box collider to work out highest and lowest points
			        var meshcoll = obj.go.AddComponent<BoxCollider>();
			        
			        
			        float top = meshcoll.bounds.max.y;
			        float bottom = meshcoll.bounds.min.y;
			        float h =top - bottom;

	                obj_height_dictionary[obj.go.name] = h;
	            	

	            	string hs = h.ToString("G6");
	            	string tops = top.ToString("G6");
	            	string bottoms = bottom.ToString("G6");
					

					string[] row = {obj.go.scene.name, obj.go.name, hs,tops,bottoms};
					string row_csv_string = String.Join(",", row);
					file.WriteLine(row_csv_string);
	            }

				
					
				
		}
        set_grounds_to_do();
		
		// Add wait script to an object
       
        
        var waiting_script = meshObjects[0].go.GetComponent(typeof(WaitingGame_FunctionalFeatureExtraction)) as WaitingGame_FunctionalFeatureExtraction;
		if (waiting_script == null){
		    waiting_script = meshObjects[0].go.AddComponent<WaitingGame_FunctionalFeatureExtraction>();
		    Debug.Log("Adding script to:");
		    Debug.Log(allObjects[0].name);
		}
		


		// Edit physics
        foreach(MeshObject obj in meshObjects){
        		
        		obj.prepare_physics_for_positioning();

                
            }
        
    

        
		// start game
        EditorApplication.isPlaying = true;

        
       	
        
	}

	

	static void Update(){
		// Once measures have been written update function is called which opens new scene and runs again
		
		if(written && !EditorApplication.isPlaying){
			Debug.Log("passed");
			//Set bool to false
			written = false;
			// Clear scene specific lists
			obj_height_dictionary.Clear();
			ground_list.Clear();

			foreach(MeshObject obj in meshObjects){
        		
        		obj.prepare_physics_for_game();

                
            }
            Scenes scene_info = new Scenes();
			foreach(MyScene s in scene_info.SceneList)
			{
				
				// if(!s.path == EditorApplication.currentScene){
					scene_list.Add(s);
					// Debug.Log("Adding scene:" + s.name);
				// }
			}
            
			// Get scene index
			using(StreamReader file =  new StreamReader(scene_index_file)){
				string line = file.ReadLine();
	            if(line == null){
	            	scene_index = 0;
	            }
	            else{
	            	scene_index = int.Parse(line.Split(',')[0]);
	            }
	            
					
	       		
	            Debug.Log("Scene Index:");
	            Debug.Log(scene_index);
				
			}

			// Rewrite csv with new index
	        using(StreamWriter filen = new StreamWriter(scene_index_file)){
	           	List<String> row = new List<String>();
	           	int n = scene_index + 1;
	   			row.Add(n.ToString());
	   				

	   				
	   			
	   			string[] row1 = row.ToArray();
	   			string row_csv_string = String.Join(",", row1);
	   			filen.WriteLine(row_csv_string);
       		}
		

		
            MyScene next_scene = scene_list[scene_index];
            //load new scene and then call init
            Debug.Log("Next Scene:");
            Debug.Log(next_scene.path);

			next_scene.open_set_active();

		    // scene_index +=1;

		    main();
	    }
	}
	
	[MenuItem ("My Tools/Calculate Functional Features")]

	static void init(){
		
		
		using(StreamWriter file = new StreamWriter(csv_file)){
			file.WriteLine("Support Calculations");
			file.WriteLine("Scene, Figure,Ground, Movement,Normalized Movement,Support");
			}
		string titles = "Scene, Figure,Ground, Movement,ground Movement,Normalized Movement,Horizontal Movement,Horizontal Ground Movement,Force used, LocCont";
		using(StreamWriter file = new StreamWriter(lc1_csv_file)){
			file.WriteLine("Location Control Calculation: x dimension");
			file.WriteLine(titles);
		}
		using(StreamWriter file = new StreamWriter(lc2_csv_file)){
			file.WriteLine("Location Control Calculations: z dimension");
			file.WriteLine(titles);
			}

		// // Rewrite csv with new index
        using(StreamWriter filen = new StreamWriter(scene_index_file)){
           	List<String> row = new List<String>();
           	int n = 0;
   			row.Add(n.ToString());
   				

   				
   			
   			string[] row1 = row.ToArray();
   			string row_csv_string = String.Join(",", row1);
   			filen.WriteLine(row_csv_string);
   		}
		main();
		
		

		
	}
}
#endif
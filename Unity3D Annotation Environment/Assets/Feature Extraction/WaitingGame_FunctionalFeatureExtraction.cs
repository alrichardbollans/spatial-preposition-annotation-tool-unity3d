// In game script which is used to calculate functional relations and also set objects at rest.

using UnityEngine;
using UnityEngine.SceneManagement;
using System;
using System.Linq;

using System.Collections;
using System.Collections.Generic;
using System.IO;

#if UNITY_EDITOR
using UnityEditor;

public class WaitingGame_FunctionalFeatureExtraction : MonoBehaviour
{	
	GameObject[] allObjects; 
	static List<GameObject> meshObjects = new List<GameObject>();
	public Dictionary<string,Vector3> initial_position_dictionary = new  Dictionary<string,Vector3>();
	public Dictionary<string,Quaternion> initial_rotation_dictionary = new  Dictionary<string,Quaternion>();
	
	public static GameObject ground;
    public static float ground_height;
    public static float ground_top_point;
    public static float ground_bottom_point;

    static float threshold_offset = 0.01f;

    public static Vector3 force;
    public static Vector3 xforce = new Vector3(1f,0f,0f);
    public static Vector3 zforce = new Vector3(0f,0f,1f);
    
    public static int fall_time = 2;
    public static int lc_move_time = 2;

    GameObject get_new_ground(){
    	//Get new ground from csv and update file
    	string[] grounds;
    	string new_ground_name;
    	string line;
    	GameObject go;
		using(StreamReader file =  new StreamReader(write_support_measures.groundstodo_file)){
			line = file.ReadLine();
            
            grounds = line.Split(',');
            new_ground_name = grounds[0];
            
			go = GameObject.Find(new_ground_name);

		}

		// Rewrite csv without new ground
        using(StreamWriter filen = new StreamWriter(write_support_measures.groundstodo_file)){
	       	List<String> row = new List<String>();
			foreach(String g in grounds){
				if(g != new_ground_name){
					row.Add(g);
				}
	
			}
			string[] row1 = row.ToArray();
			string row_csv_string = String.Join(",", row1);
			filen.WriteLine(row_csv_string);
   		}
		

		return go;
	}

	static void reset_grounds_to_do(){	
        // Write all grounds for scene to csv.
        using(StreamWriter file = new StreamWriter(write_support_measures.groundstodo_file)){
        	List<String> row = new List<String>();
			foreach(GameObject obj in meshObjects){
				
				if(!Main.unselectable_scene_objects.Any(x => obj.name.Contains(x))){
					row.Add(obj.name);
				}		
			}
			string[] row1 = row.ToArray();
			string row_csv_string = String.Join(",", row1);
			file.WriteLine(row_csv_string);
		}
	}
		
	// Applies velocity to ground
	static void apply_force(){
		if(write_support_measures.type.Contains("-x")){
			force = -xforce;
		}
		else if(write_support_measures.type.Contains("x")){
			force = xforce;
		}
		else if (write_support_measures.type.Contains("-z")){
			force = -zforce;
		}
		else if (write_support_measures.type.Contains("z")){
			force = zforce;
		}
		// Note, only 'floating' objects don't have a rigid body, which don't provide location control anyway
		Rigidbody r = ground.GetComponent(typeof(Rigidbody)) as Rigidbody;
		if (r != null){
			
			
			r.AddForce(force,ForceMode.VelocityChange);
		}

	}
    
	static void remove_ground(){
		// Disables ground colliders from scene so it doesn't interact with other objects
		// disable normal colliders
		foreach (Collider c in ground.GetComponents<Collider>()){
        	if(c!= null){
	        	c.enabled = false;
	        }
        }
        // disable noncnovex colliders
		var collidersTransform = ground.transform.Find("Colliders");
	    GameObject collidersGo;
	    if (collidersTransform != null){
	        collidersGo = collidersTransform.gameObject;
	        foreach(Collider c in collidersGo.GetComponents<Collider>()){
	        	if(c!= null){
		        	c.enabled = false;
		        }
	        }
	    }
	    // make the ground not fall. This is inconsiquential.
        Rigidbody r = ground.GetComponent(typeof(Rigidbody)) as Rigidbody;
		if (r != null){
			r.useGravity = false;
		}
	}

	static void add_ground_back(){
		// Adds colliders for ground back to scene.
		foreach (Collider c in ground.GetComponents<Collider>()){
        	if(c!= null){
	        	c.enabled = true;
	        }
        }
        // Enable noncnovex colliders
		var collidersTransform = ground.transform.Find("Colliders");
	    GameObject collidersGo;
	    if (collidersTransform != null){
	        collidersGo = collidersTransform.gameObject;
	        foreach(Collider c in collidersGo.GetComponents<Collider>()){
	        	if(c!= null){
		        	c.enabled = true;
		        }
	        }
	    }
        Rigidbody r = ground.GetComponent(typeof(Rigidbody)) as Rigidbody;
		if (r != null){
			r.useGravity = true;
		}
		
	}

	IEnumerator reset_positions(){
		// Reloads scene to reset positions
		Scene active_scene = SceneManager.GetActiveScene();
		SceneManager.LoadScene(active_scene.name);//,LoadSceneMode.Additive);
		yield return null;
		SceneManager.SetActiveScene(SceneManager.GetSceneByName(active_scene.name));
	}

	

	IEnumerator print_support_positions(){
		
		// Removes ground, waits for object to fall and then prints positions

		remove_ground();
		// Wait 10 seconds for objects to fall
		yield return new WaitForSeconds(fall_time);
		string[] ground_info;
		using(StreamWriter file = new StreamWriter(write_support_measures.csv_file, true)){
			
			foreach(GameObject obj in meshObjects){
				
				string line;
				float fig_top = -1f;
				float fig_bottom = -1f;
				
				float fig_com_y_initial = initial_position_dictionary[obj.name].y;

				float movement = fig_com_y_initial - get_com(obj).y;
				using(StreamReader filegh =  new StreamReader(write_support_measures.ground_heights_file)){
					while((line = filegh.ReadLine()) != null)
		               {
		               	
		                ground_info = line.Split(',');
		                if(ground_info[1] == ground.name){
		                	ground_height = float.Parse(ground_info[2]);
		                	ground_top_point = float.Parse(ground_info[3]);
		                	ground_bottom_point = float.Parse(ground_info[4]);
		                }

		                if (ground_info[1] == obj.name){
		                	
		                	fig_top = float.Parse(ground_info[3]);
		                	fig_bottom = float.Parse(ground_info[4]);
		                }
						
		           }

					
				}
				if (fig_top == -1){
					Debug.LogError("fig_top missing in WaitingGame_FunctionalFeatureExtraction.cs",obj);
				}
				// Edit ground height to better account for different scenarios
				
				else if (fig_bottom-ground_top_point>threshold_offset){
					//Do nothing: ground height stays the same
				}

				else if (fig_bottom - ground_bottom_point > threshold_offset){
					ground_height = fig_bottom - ground_bottom_point;
					// Debug.Log("1 ########");
					// Debug.Log(ground.name);
					// Debug.Log(obj.name);
					// Debug.Log(ground_height);
				}
				else if (fig_com_y_initial-ground_bottom_point > threshold_offset){
					ground_height = fig_com_y_initial - ground_bottom_point;
					// Debug.Log("2 ########");
					// Debug.Log(ground.name);
					// Debug.Log(obj.name);
					// Debug.Log(ground_height);
				}

				
				
				float normalized_movement = movement/ground_height;

				float support = normalized_movement;
				if (support < 0){
					support = 0;
				}
				

				if (support > 1){
					support = 1;
				}

				// Create relationship instance to store values
				Relationship r = new Relationship(obj.scene.name, obj.name,ground.name);
				r.relation_dictionary["support"] = support;
				r.save_to_csv();

				string m = movement.ToString("G6");
				string nm = normalized_movement.ToString("G6");
				string sup = support.ToString("G6");

				string[] row = {obj.scene.name, obj.name,ground.name, m,nm,sup};
				string row_csv_string = String.Join(",", row);
				file.WriteLine(row_csv_string);
			}
		}
		
		// adding the ground back was causing crashes as adding rigidbody back to a moving object is 
		// problematic. It is also not necessary as reloading the scene resets everything
		// add_ground_back();
		// Wait for ground to come back
		// yield return null;
		
	}

	void print_lc_positions(string writing_file){
		// Distance the ground has moved
		float ground_movement = Vector3.Distance(initial_position_dictionary[ground.name],get_com(ground));
				
		using(StreamWriter filelc = new StreamWriter(writing_file, true)){
			
			foreach(GameObject obj in meshObjects){
				
				
				// Distance the figure has moved
				float movement = Vector3.Distance(initial_position_dictionary[obj.name],get_com(obj));
				// This is the movement in the direction of the force
				float horizontal_movement = 0;
				float ground_horizontal_movement = 0;
				//If force is in x direction use x position
				if(force.x == 1 || force.x ==-1){
					horizontal_movement= initial_position_dictionary[obj.name].x - get_com(obj).x;
					ground_horizontal_movement = initial_position_dictionary[ground.name].x - get_com(ground).x;
				}
				// Same for z
				if(force.z==1 || force.z ==-1){
					horizontal_movement= initial_position_dictionary[obj.name].z - get_com(obj).z;
					ground_horizontal_movement = initial_position_dictionary[ground.name].z - get_com(ground).z;

				}
				
				// // Get absolute value
				// if(horizontal_movement < 0){
				// 	horizontal_movement = -horizontal_movement;
				// }
				
				// Normalize movement by dividing by distance travelled by ground
				float normalized_movement;
				if(ground_movement!=0){
					normalized_movement = movement/ground_movement;
				}
				else{
					normalized_movement = 0;
				}

				// Horizontal version should be more accurate as it
				// avoids falling off tables etc
				float normalized_horizontal_movement;
				if(ground_horizontal_movement!=0){
					normalized_horizontal_movement = horizontal_movement/ground_horizontal_movement;
				}
				else{
					normalized_horizontal_movement = 0;
				}

				// Cap measure
				float locCont = normalized_horizontal_movement;
				if (locCont < 0){
					locCont = -locCont;
				}
				

				if (locCont > 1){
					locCont = 1;
				}
				// Create relationship instance to store values
				Relationship r = new Relationship(obj.scene.name, obj.name,ground.name);
				r.relation_dictionary[write_support_measures.type] = locCont;
				r.save_to_csv();

				string m = movement.ToString("G6");
				string gm = ground_movement.ToString("G6");
				string nm = normalized_movement.ToString("G6");
				string ghm = ground_horizontal_movement.ToString("G6");
				string hm = horizontal_movement.ToString("G6");

				string f1 = force.x.ToString("G6");
				string f2 = force.y.ToString("G6");
				string f3 = force.z.ToString("G6");

				string f = f1 + ":" + f2 + ":" + f3;

				string[] row = {obj.scene.name, obj.name,ground.name, m,gm,nm,hm,ghm,f,locCont.ToString("G6")};
				string row_csv_string = String.Join(",", row);
				filelc.WriteLine(row_csv_string);
			}
		}
	}

	IEnumerator lc_positions(){
		
		
		// Apply force to ground
		apply_force();
		// Wait 5 seconds for objects to move
		yield return new WaitForSeconds(lc_move_time);
		string fname = "";
		if(write_support_measures.type.Contains("x")){
			fname = write_support_measures.lc1_csv_file;
		}
		if(write_support_measures.type.Contains("z")){
			fname = write_support_measures.lc2_csv_file;
		}
		
		print_lc_positions(fname);
		
	}

	IEnumerator run_pp(){
		// Runs 'print_support_positions' and then resets positions.
		// If all grounds are done, gets editor script to change scene
		
		// Debug.Log("run_pp");
		// Debug.Log(write_support_measures.type);
		ground = get_new_ground();
		if(ground != null){
			
			Debug.Log("ground: " + ground.name);
			if (write_support_measures.type == "support"){
				// Debug.Log("running support positions");
				StartCoroutine(print_support_positions());
				// Debug.Log(Time.time);

				// Wait 20 seconds for above to execute
				yield return new WaitForSeconds(fall_time+2);
				// Reset Positions
				StartCoroutine(reset_positions());
				
			}
			else if (write_support_measures.type.Contains("location_control")){
				// Debug.Log("running lc positions");
				// Run LC calculations and print positions
				StartCoroutine(lc_positions());
				// Wait to execute
				yield return new WaitForSeconds(lc_move_time+2);

				StartCoroutine(reset_positions());
			}
			
		}
		else if (write_support_measures.type == "support"){
			Debug.Log("changing type");
			write_support_measures.type = "location_control_x";
			reset_grounds_to_do();
			StartCoroutine(reset_positions());
		}
		else if (write_support_measures.type == "location_control_x"){

			write_support_measures.type = "location_control_-x";
			reset_grounds_to_do();
			StartCoroutine(reset_positions());
		}
		else if (write_support_measures.type == "location_control_-x"){

			write_support_measures.type = "location_control_z";
			reset_grounds_to_do();
			StartCoroutine(reset_positions());
		}
		else if (write_support_measures.type == "location_control_z"){

			write_support_measures.type = "location_control_-z";
			reset_grounds_to_do();
			StartCoroutine(reset_positions());
		}
		


		else{
			Debug.Log("Waiting Script Attached to:");
		    Debug.Log(gameObject.name);
			
			write_support_measures.written = true;

			Application.Quit();
			#if UNITY_EDITOR
	        //Stop playing the scene
	        UnityEditor.EditorApplication.isPlaying = false;
		    #endif
		}
			
		
	    
	    
	}
	Vector3 get_com(GameObject obj){
		Rigidbody r = obj.GetComponent(typeof(Rigidbody)) as Rigidbody;
		if (r != null){
			return r.centerOfMass + r.position;/// com is relative to rigid body position
			
			
		}
		else{
			Debug.LogError("Missing Rigidbody",obj);
			
			Vector3 x = new Vector3(-1,-1,-1);
			return x;
		}
	}

	void Start(){

		meshObjects.Clear();
		
		allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		// Populate mesh list
		foreach(GameObject obj in allObjects){
            MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
            if(!obj.name.Contains("wall") && !obj.name.Contains("ceiling") && !obj.name.Contains("floor")){
	            if(mf != null){
	        		
	                meshObjects.Add(obj);
	                // ground_list.Add(obj);
	                // Debug.Log(obj.name);
	            }
	        }
        }
        // start by getting initial positions of centre of masses
		foreach(GameObject obj in meshObjects){
			
			
			Rigidbody r = obj.GetComponent(typeof(Rigidbody)) as Rigidbody;
			if (r != null){
				initial_position_dictionary[obj.name] = get_com(obj);

				
				
			}

			// initial_position_dictionary[obj.name] = obj.transform.position;
			initial_rotation_dictionary[obj.name] = obj.transform.rotation;
		}
		// Speeds up physics simulations. Adjust Wait times accordingly
		// Setting too high causes issues
		Time.timeScale = 5f;
		
		StartCoroutine(run_pp());
		
	}
}

#endif
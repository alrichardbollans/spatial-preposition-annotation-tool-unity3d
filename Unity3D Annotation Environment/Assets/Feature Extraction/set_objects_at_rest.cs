// This script makes edits to the positions of objects in the scene so that 
// they are naturally at rest.

// It does this by adding/removing colliders, giving the objects rigidbody,
// playing the scene writing positions (using an added in-game script) 
// and then reading the positions and editing transforms.
#if UNITY_EDITOR
using UnityEngine;

using UnityEditor;
using UnityEditor.SceneManagement;

using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;


/// <summary>
//// MeshObject class to hold some extra information about gameobjects with a mesh.
/// </summary>
public class MeshObject{
	// Objects which can be simplified with a convex collider.
	public static string[] convex_objects = { "pencil",
    	"cube",
    	"apple",
    	"pear",
    	"chair",
    	"box", // Need to be careful here as there are two types of boxes
    	"wall",
    	"floor", 
    	"ceiling",
    	"book",
    	"jar",
    	"picture"
    }; 
    // Objects that need giving a sphere collider.
    public static string[] sphere_objects = { "sphere", "ball"};
    // Objects which gravity doesn't affect.
    static string[] floating_objects = {"balloon"};
	static Dictionary<string,float> mass_dictionary;
	public GameObject go;
	public float mass;
	public string clean_name;
	public Collider[] colliders;

	public MeshObject(GameObject obj){
		go =obj;
		colliders = go.GetComponents<Collider>();
		mass_dictionary = new  Dictionary<string,float>
			{
				{"pencil",0.01f},
				{"table",20f},
				{"chair",3f},
				{"spoon", 0.05f}, 
				{"bowl", 0.3f},
				{"cube", 0.1f}, 
				{"apple", 0.05f},
				{"pear", 0.05f},
				{"banana", 0.05f}, 
				{"robot", 20f}, 
				{"book", 0.3f},
				{"box", 0.3f},
				{"jar", 0.3f}, 
				{"bookshelf", 10f},
				{"board", 10f},
				{"picture", 1f},
				{"bin", 1f},
				{"floor", 100f},
				{"wall", 100f},
				{"ceiling", 50f},
				{"balloon", 0.1f},
				{"cup", 0.3f},
				{"mug", 0.3f},
				{"goblet", 0.4f},
				{"lamp", 0.3f},
				{"ball", 0.1f},
				{"plate", 0.2f},
				{"shelf", 1f},
				{"clock", 1f}
             };
        
        // Find clean name
		clean_name = Main.clean_name(obj.name);
		// Add weight
		float val;

		if(mass_dictionary.TryGetValue(clean_name, out val)){
			mass = val;
		}
		else{
			mass = 1f;
			Debug.Log("Weight Missing for: " + go.name );
		}
	}

	/// <summary>
	/// Prepare object physics for positioning --- like real life.
	/// </summary>
	public void prepare_physics_for_positioning(){
		
			if (go != null){
				remove_colliders();
				give_rigid_body();  
		        add_non_convex_collider();
			}
			else{
				Debug.Log(clean_name + "is null");
			}
		
	}

	/// <summary>
	/// Prepare object physics for game --- static, colliders for clicking etc...
	/// </summary>
	public void prepare_physics_for_game(){
		remove_colliders();
		remove_rigid_body();
		remove_non_convex_colliders();
		add_mesh_collider_for_game();
		remove_scripts();
	}

	/// <summary>
	/// If necessary, give object a rigid body.
	/// </summary>
	public void give_rigid_body(){
		
		// Make sure has rigidbody.
        Rigidbody r = go.GetComponent(typeof(Rigidbody)) as Rigidbody;
        if (r == null){
            r = go.AddComponent<Rigidbody>() as Rigidbody;
        }
        r.mass = mass;
        
        if(Main.unselectable_scene_objects.Any(x => go.name.Contains(x))){
        	r = go.GetComponent(typeof(Rigidbody)) as Rigidbody;
        	
        	Object.DestroyImmediate(r);
        	
        }

        // Make floating objects not use gravity
        if(floating_objects.Contains(clean_name)){
        	r.useGravity = false;
        }
	    

	}

	/// <summary>
	/// If one is there, remove rigid body.
	/// </summary>
	public void remove_rigid_body(){
		Rigidbody r = go.GetComponent(typeof(Rigidbody)) as Rigidbody;
		if (r != null){
		    Object.DestroyImmediate(r);
		}
	}

	/// <summary>
	//// Disable all colliders attached to object.
	/// </summary>
	public void disable_colliders(){

        foreach (Collider c in colliders){
        	if(c!= null){
	        	c.enabled = false;
	        }
        }
	}

	/// <summary>
	//// Enable all colliders attached to object.
	/// </summary>
	public void enable_colliders(){
        foreach (Collider c in colliders){
        	if(c!= null){
	        	c.enabled = true;
	        }
        }
	}

	/// <summary>
	//// Destory all colliders attached to object.
	/// </summary>
	public void remove_colliders(){
		
        foreach (Collider c in go.GetComponents<Collider>()){
        	if(c!= null){
	        	Object.DestroyImmediate(c);
	        }
        }
	}

	/// <summary>
	//// Add a mesh collider.
	/// </summary>
	public void add_convex_mesh_collider(){

		MeshFilter mf = go.GetComponent(typeof(MeshFilter)) as MeshFilter;
        if(mf != null){
            var meshcoll = go.AddComponent<MeshCollider>();
            meshcoll.enabled = true;
            meshcoll.convex = true;
        }
	}

	/// <summary>
	//// Add a sphere collider.
	/// </summary>
	public void add_sphere_collider(){

		MeshFilter mf = go.GetComponent(typeof(MeshFilter)) as MeshFilter;
        if(mf != null){
        	var sphcoll = go.GetComponent<SphereCollider>();
			if(sphcoll == null){
	            var sphcollider = go.AddComponent<SphereCollider>();
	        }
        }
	}

	/// <summary>
	//// Add a mesh collider for in game.
	/// Checks various properties to add correct collider for game, as this can cause issues.
	/// </summary>
	public void add_mesh_collider_for_game(){
		if(sphere_objects.Contains(clean_name)){
			
			add_sphere_collider();
			
		}
		else{
			var meshcoll = go.GetComponent<MeshCollider>();
			if(meshcoll == null){
				MeshFilter mf = go.GetComponent(typeof(MeshFilter)) as MeshFilter;
		        if(mf != null){
		            var mcoll = go.AddComponent<MeshCollider>();
		            mcoll.enabled = true;
		            if(convex_objects.Contains(clean_name)){
			            mcoll.convex = true;
			        }
			        else{
			        	mcoll.convex = false;
			        }
		        }
		    }

		    else{
		    	if(convex_objects.Contains(clean_name)){
			            meshcoll.convex = true;
			        }
		        else{
		        	meshcoll.convex = false;
		        }
		    }
		}
	}

	/// <summary>
	//// Add a nonconvex collider if necessary.
	/// </summary>
	/// <remarks>
	/// Relies on NonConvexCollider asset.
	/// </remarks>
	public void add_non_convex_collider(){
		// Make floor, wall and ceiling simple colliders first
		if (sphere_objects.Contains(clean_name)){
			remove_non_convex_colliders();
			remove_colliders();
			add_sphere_collider();
		}
		// This is to rectify an error in naming two different types of object "box"
		else if (convex_objects.Contains(clean_name) && !(clean_name.Contains("box") && go.scene.name == "compsvi15")){
			
			remove_non_convex_colliders();
			remove_colliders();
			add_convex_mesh_collider();
			
		}
		
		
		else{
			var non_convex_script = go.GetComponent(typeof(NonConvexMeshCollider)) as NonConvexMeshCollider;
			if (non_convex_script == null){
			    non_convex_script = go.AddComponent<NonConvexMeshCollider>();
			}
			non_convex_script.boxesPerEdge = 50;
			non_convex_script.avoidExceedingMesh = false;
			non_convex_script.Calculate();

		}

	}
	
	/// <summary>
	/// Remove all colliders from non-convex collider.
	/// </summary>
	public void remove_non_convex_colliders(){
		var collidersTransform = go.transform.Find("Colliders");
        GameObject collidersGo;
        if (collidersTransform != null){
            collidersGo = collidersTransform.gameObject;
            Object.DestroyImmediate(collidersGo);
        }
	}

	/// <summary>
	/// Removes all scripts attached to game objects which may interfere.
	/// </summary>
	public void remove_scripts(){
		var waiting_script = go.GetComponent(typeof(WaitingGame)) as WaitingGame;
		if (waiting_script != null){
		    Object.DestroyImmediate(waiting_script);
		}
		var waiting_script1 = go.GetComponent(typeof(WaitingGame_FunctionalFeatureExtraction)) as WaitingGame_FunctionalFeatureExtraction;
		if (waiting_script1 != null){
		    Object.DestroyImmediate(waiting_script1);
		}
		var non_convex_script = go.GetComponent(typeof(NonConvexMeshCollider)) as NonConvexMeshCollider;
		if (non_convex_script != null){
		    Object.DestroyImmediate(non_convex_script);
		}

		var check_script = go.GetComponent(typeof(feature_check)) as feature_check;
		if (check_script != null){
		    Object.DestroyImmediate(check_script);
		}
	}
}


/// <summary>
/// Prepares object physics then runs game so that object positions are written once fallen.
/// Then sets new object positions. User must save new scene themselves.
/// </summary>
/// <remarks>
/// Initialize on load means Update method of this class is constantly running.
/// </remarks>
[InitializeOnLoad]
public class write_transforms : EditorWindow{
	static GameObject[] allObjects;
	static List<MeshObject> meshObjects = new List<MeshObject>();
	static public bool written = false; // this gets edited by an in-game script
	static write_transforms(){
		EditorApplication.update += Update;
	}
	

	static public void main(){

		// Add wait script to an object
        Debug.Log("Passing adding script:");
        
        var waiting_script = meshObjects[0].go.GetComponent(typeof(WaitingGame)) as WaitingGame;
		if (waiting_script == null){
		    waiting_script = meshObjects[0].go.AddComponent<WaitingGame>();
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

	static void read_and_update_positions(){
		// Read positions
		string line;
		using(StreamReader file =  new StreamReader(WaitingGame.new_positions_csv)){
			while((line = file.ReadLine()) != null)
            {

                string[] values = line.Split(';');
                // Remove " and (
                // Debug.Log(values[2]);
                values[2] = values[2].Substring(1,values[2].Length-2);
                // Debug.Log(values[2]);
                string[] p_values = values[2].Split(',');

                values[3] = values[3].Substring(1,values[3].Length-2);

                string[] q_values = values[3].Split(',');
	            
	            // values[1] = values[1].Substring(2,values[1].Length-2);
	            // Debug.Log(values[1]);
				GameObject go = GameObject.Find(values[1]);
				// Make editor record changes
				Undo.RecordObject(go.transform,"setting at rest");
				// Do transforms
				float x = float.Parse(p_values[0]);
				float y = float.Parse(p_values[1]);
				float z = float.Parse(p_values[2]);
				Vector3 p = new Vector3(x,y,z);

				go.transform.position = p;

				x = float.Parse(q_values[0]);
				y = float.Parse(q_values[1]);
				z = float.Parse(q_values[2]);
				float w = float.Parse(q_values[3]);

				Quaternion q = new Quaternion(x,y,z,w);
				go.transform.rotation = q;
           }

			
		}
	}

	/// <summary>
	/// Update constantly checks and once main has completed update object properties and revert physics.
	/// </summary>
	static void Update(){
		
		if(written && !EditorApplication.isPlaying){
			//Set bool to false
			written = false;
			Debug.Log("updating");

			allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
			// Populate mesh list
			foreach(GameObject obj in allObjects){
	            MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
	            if(mf != null){
	            	MeshObject mobj = new MeshObject(obj);
	                meshObjects.Add(mobj);
	            }
	        }
	
			read_and_update_positions();

			//Revert properties
			foreach(MeshObject obj in meshObjects){
        		
        		obj.prepare_physics_for_game();    
            }

		}
	}
	[MenuItem ("My Tools/Set Objects At Rest")]

	static void init(){
		Debug.Log("Check initialiseonload is uncommented");
		written = false;
		meshObjects.Clear();
		allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		// Populate mesh list
		foreach(GameObject obj in allObjects){
            MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
            if(mf != null){
            	MeshObject mobj = new MeshObject(obj);
                meshObjects.Add(mobj);
            }
        }

		main();
		
	}
}
#endif
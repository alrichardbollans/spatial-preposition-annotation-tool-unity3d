// Script which executes geometric feature extraction.
#if UNITY_EDITOR

using UnityEngine;

using UnityEditor;
using UnityEditor.SceneManagement;

using System;
using System.Linq;
using System.IO;
using System.Collections;
using System.Collections.Generic;




public class Feature_Extraction: EditorWindow{
	
	/// <summary>
	/// Iterate through all scenes and calculate all relations.
	/// </summary>
	/// <remarks>
	/// Proviodes editor button.
	/// </remarks>
	[MenuItem ("My Tools/Calculate Geometric Features")]
	static void init(){
		Scenes scene_info = new Scenes();

		// Iterate through all scenes and calculate all relations.
      	int i;
        
        for (i = 0; i < scene_info.SceneList.Count; i ++)
        {
            
            MyScene next_scene = scene_info.SceneList[i];

            next_scene.open_set_active();
            
            
            calculate_all_relations();
           
        }
        // calculate_all_relations();

        Debug.Log("Calculating geometric features done");
	}

	/// <summary>
	/// Calculates relations for all pairs of mesh objects.
	/// </summary>
	static public void calculate_all_relations(){
        GameObject[] allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		List<GameObject> meshObjects = new List<GameObject>();
		List<Entity> entities = new List<Entity>();
		// First remove unecessary colliders and recreate allobject list
		foreach(GameObject obj in allObjects){
			MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
			if(mf != null){
			    MeshObject mobj = new MeshObject(obj);
			    
			    mobj.prepare_physics_for_game();
			}
		}
		allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		foreach(GameObject obj in allObjects){
			if(!obj.name.Contains("wall") && !obj.name.Contains("ceiling") && !obj.name.Contains("floor")){
								
                MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
                if(mf != null){
                    meshObjects.Add(obj);
                    Entity e = new Entity(obj);
                    entities.Add(e);
                }
            }
        }
		
        // Create pair instance for each pair of objects, which calculates values and saves to csv.
		foreach(Entity e1 in entities){
			
			foreach(Entity e2 in entities){
				
				Pair p = new Pair(e1,e2);	
			}
		}
	}
}
#endif




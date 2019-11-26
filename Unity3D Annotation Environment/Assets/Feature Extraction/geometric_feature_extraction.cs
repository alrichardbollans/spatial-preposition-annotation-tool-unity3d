//Script which executes geometric feature extraction


using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif

using System;
using System.Linq;
using System.IO;
using System.Collections;
using System.Collections.Generic;



#if UNITY_EDITOR

public class Feature_Extraction: EditorWindow{
	
	[MenuItem ("My Tools/Calculate Geometric Features")]

	static void init(){
		Scenes scene_info = new Scenes();


		// Iterate through all scenes
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

	public void calculate_properties(List<Entity> ents){
		foreach(Entity ent in ents){
			string name = ent.name;
			string scene = ent.scene;

			Property p = new Property(scene,name);

			p.property_dictionary["bbox_volume"] = ent.bbox_volume;
			// p.property_dictionary[""] = ent.lowest_point;

			p.save_to_csv();

		}

	}
	// Run calculate_relations for all pairs of objects
	static public void calculate_all_relations(){
        GameObject[] allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		List<GameObject> meshObjects = new List<GameObject>();
		List<Entity> entities = new List<Entity>();
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
		

		foreach(Entity e1 in entities){
			
			foreach(Entity e2 in entities){
				// Create pair instance which calculates values and saves to csv
				Pair p = new Pair(e1,e2);

				
			}
		}
	}
}
#endif




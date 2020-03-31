using UnityEngine;
using UnityEngine.UI;
using System.Linq;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;

#if UNITY_EDITOR


public class Placeholder: EditorWindow{
	static string get_strings(string orig,int first_ch_index){
		string sc;
		int end = orig.Length - first_ch_index;
		sc = orig.Substring(first_ch_index,end);
		
		sc = sc.Substring(0,sc.IndexOf(":"));
		
		return sc;
	}
	[MenuItem ("My Tools/Tests")]
	static void init(){
		// Light[] lights = UnityEngine.Object.FindObjectsOfType<Light>();
		// foreach(Light l in lights){
		//     // l.shadows= LightShadows.Soft;
		//     if(l.lightmapBakeType == LightmapBakeType.Realtime){
		//     	Debug.Log(l.name);
		//     	l.shadowBias = 0;
		//     }
		// }
		// Sprite[] sprites =Resources.LoadAll<Sprite>("Typ_task_folder");
		Texture2D[] imgs =Resources.LoadAll<Texture2D>("Typ_task_folder");
		
		Texture2D img = imgs[3];//.Where(a => a.name == "nice").First() as Texture2D;//Resources.Load <Sprite> ("typ_task_folder/nice");
		Debug.Log(img.name);
		GameObject li = GameObject.Find("RightImage");

		// li.GetComponent<RawImage>().texture = img;

		string old = img.name;
		string fig;
		string grd;
		string scene;
		
		int first_scene_character_index = old.IndexOf("scene:") +6;
		int first_fig_character_index = old.IndexOf("figure:") +7;
		int first_gr_character_index = old.IndexOf("ground:") +7;
		
		fig = get_strings(old,first_fig_character_index);
		scene = get_strings(old,first_scene_character_index);
		grd = get_strings(old,first_gr_character_index);
		
		Debug.Log(fig);
		Debug.Log(scene);
		Debug.Log(grd);


		// GameObject[] allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		// List<GameObject> meshObjects = new List<GameObject>();
		// List<Entity> entities = new List<Entity>();
		// foreach(GameObject obj in allObjects){
		// 	if(!obj.name.Contains("wall") && !obj.name.Contains("ceiling") && !obj.name.Contains("floor")){
								
  //               MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
  //               if(mf != null){
  //                   meshObjects.Add(obj);
  //                   Entity e = new Entity(obj);
  //                   entities.Add(e);
  //               }
  //           }
  //       }
		

		// foreach(Entity e1 in entities){
		// 	Debug.Log(e1.name);
		// 	Debug.Log(e1.centre_of_mass);
		// 	if(e1.name == "paper"){
		// 		Debug.Log(e1.lowest_point);
		// 		Debug.Log(e1.highest_point);
		// 		float thickness = e1.highest_point - e1.lowest_point;
		// 		Debug.Log(thickness);
		// 	}
			// foreach(Entity e2 in entities){
				
			// 	Pair p = new Pair(e1,e2);
			// 	Debug.Log(e1.name);
			// 	Debug.Log(e2.name);
			// 	Debug.Log(p.horizontal_projection_overlap_proportion);

			// }
		// }
		// Debug.Log(EditorApplication.currentScene);
		// Scenes scene_info = new Scenes();
		// foreach(MyScene s in scene_info.SceneList)
		// {
		// 	Debug.Log(s.path);
		// 	// if(s.name == )
		// }
		// GameObject[] allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		// Entity e2 = new Entity(allObjects[2]);
		// Debug.Log(e2.name);

		// // Entity e1 =  new Entity(allObjects[2]);
		// // Pair p = new Pair(e1,e2);
		// // Debug.Log(p.names);
		// // float c = p.proportion_contact();
		// // Debug.Log(c);
		// foreach(GameObject obj in allObjects){
				
				
	 //            MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
	 //            if(mf != null){
	            	
	 //            	Entity e1 =  new Entity(obj);
		// 			Pair p = new Pair(e1,e2);
		// 			Debug.Log(p.names);
					// float c = p.proportion_contact();
					// Debug.Log(c);
	 //            	// Debug.Log(obj.name);
	 //            	// // Debug.Log(mf.sharedMesh.isReadable);
	 //            	// Debug.Log(mf.mesh.vertices.Length);
	 //            // MeshCollider mc = obj.GetComponent(typeof(MeshCollider)) as MeshCollider;
	 //            // if(mc != null){
	 //            // 	Debug.Log(obj.name);
	 //            // 	// Debug.Log(mf.sharedMesh.isReadable);
	 //            // 	Debug.Log(mc.vertices.Length);
	 //            // }
		// 	    // Rigidbody rb = obj.GetComponent(typeof(Rigidbody)) as Rigidbody;
		// 	    // if(rb!= null){
		// 	    //     Vector3 com = rb.centerOfMass;

		// 	    //     // Debug.Log(com);
		// 	    //     // Debug.Log(rb.position);
		// 	    // }

		// 	    // if(rb== null){
		// 	    // 	rb = obj.AddComponent<Rigidbody>() as Rigidbody;
		// 	    //     Vector3 com = rb.centerOfMass;

		// 	    //     // Debug.Log(com);
		// 	    //     // Debug.Log(rb.position);
		// 	    // }

	            }
	        // }


		// bool t = LightEditor.Settings.isCompletelyBaked;
		
		// mi.generateSecondaryUV = true;

		 // Lightmapping.realtimeGI = true;
	// }
}
#endif


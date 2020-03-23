using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;

#if UNITY_EDITOR


public class SceneInfoButton: EditorWindow{
	
	[MenuItem ("My Tools/Write Scene Info")]
	static void init(){
		Scenes scene_info = new Scenes();
		
		foreach(MyScene s in scene_info.SceneList)
		{
			
		    
		    
		    Debug.Log("scene path:" + s.path);
		    
		    EditorSceneManager.OpenScene(s.path);
		    
		    EditorSceneManager.SetActiveScene(EditorSceneManager.GetSceneByName(s.name));
		    
		    
		    GameObject[] allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		    // Add a collider so objects are clickable and remove rigidbody
		    foreach(GameObject obj in allObjects){
		        Debug.Log(obj.name);
		        MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
		        if(mf != null){
		        	s.mesh_object_list.Add(obj.name);
		        	string cn = Main.clean_name(obj.name);
		        	if (!Scenes.allObjectcleannames.Contains(cn)){
		        		
		        		Scenes.allObjectcleannames.Add(cn);		    
		        	}
		        	        
		        }
		        
		    }
		    
		   
		}
		scene_info.save_to_csv();
		}
}
#endif


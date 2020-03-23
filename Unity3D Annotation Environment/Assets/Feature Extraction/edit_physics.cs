// Provides buttons which prepares physics for all objects in scene for either positioning or game.

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

/// <summary>
/// Class which provides button which prepares all objects in scene for positioning.
/// </summary>
public class edit_physics_for_positioning: EditorWindow{
	
	
	static GameObject[] allObjects;
	[MenuItem ("My Tools/Set Physics for Positioning")]
	static void init(){
		allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		

		foreach(GameObject obj in allObjects){
            MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
            if(mf != null){
            	MeshObject mobj = new MeshObject(obj);
                
                mobj.prepare_physics_for_positioning();
            }
        }
	}
		
}

/// <summary>
/// Class which provides button which prepares all objects in scene for game.
/// </summary>
public class edit_physics_for_game: EditorWindow{
	
	
	static GameObject[] allObjects;
	[MenuItem ("My Tools/Set Physics for Game")]
	static void init(){
		allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		

		foreach(GameObject obj in allObjects){
            MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
            if(mf != null){
            	MeshObject mobj = new MeshObject(obj);
                
                mobj.prepare_physics_for_game();
            }
        }
	}
		
}
#endif




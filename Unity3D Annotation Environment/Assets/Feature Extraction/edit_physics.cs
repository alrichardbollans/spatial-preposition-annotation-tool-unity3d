// This more or less works. Needs some editing to only select objects with meshes. Need to make saving work properly and check loading
/// Something wrong with writing values in sacve_csv
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

// To calculate closest distance use nonconvex colliders and Collider.ClosestPoint 
/// --- search points on one mesh (with least vertices) and for each vertex find the closest point on each collider of other object
// Using this find the distance and therefore the shortest distance for each point and therefore the shortest overall distance

#if UNITY_EDITOR

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




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


public class Placeholder1: EditorWindow{
	
	

	static string clean_name(string name){
		if (name.Contains("(")){
			
			return name.Substring(0,name.LastIndexOf("("));
		}
		else {
			return name;
		}
	}
	[MenuItem ("My Tools/Tests")]
	static void init(){
		// This is for geometric features
		Feature_Extraction.calculate_all_relations();
		// And for support, just need to comment out update function in support.cs
	            }
	        
}
#endif


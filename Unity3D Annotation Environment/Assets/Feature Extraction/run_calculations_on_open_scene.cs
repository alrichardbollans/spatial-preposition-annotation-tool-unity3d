// Script for calculating features on current open scene.
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
	
	[MenuItem ("My Tools/Tests")]
	static void init(){
		// This is for geometric features
		Feature_Extraction.calculate_all_relations();
		// And for support?
    }
	        
}
#endif


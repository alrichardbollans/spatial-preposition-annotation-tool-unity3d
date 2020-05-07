// Script for calculating features on current open scene.
#if UNITY_EDITOR

using UnityEngine;

using UnityEditor;
using UnityEditor.SceneManagement;

using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;



public class Placeholder1: EditorWindow{
	
	[MenuItem ("My Tools/Run Geometric FE on open scene")]
	static void init(){
		// This is for geometric features
		Feature_Extraction.calculate_all_relations();
		// And for support?
    }
	        
}
#endif


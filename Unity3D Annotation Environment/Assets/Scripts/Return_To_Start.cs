// Script added to Canvas in screening_fail

using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System;
using System.IO;

public class Return_To_Start : MonoBehaviour {
 
    /// <summary>
    /// Loads first scene.
    /// Added to Button in screening_fail.
    /// </summary>
    public void Return() 
    {
        
       UnityEngine.SceneManagement.SceneManager.LoadScene(Main.first_scene_name);
    	
    	
    }
    

  }


// Script added to Canvas in screening_fail

using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System;
using System.IO;

public class Return_To_Start : MonoBehaviour {
    GameObject main_empty;
	Main main_empty_script;

	static string first_scene_name = "player_menu";

	public void Awake(){
		main_empty = GameObject.Find("mainEmpty");
		if(main_empty !=null){
			main_empty_script = main_empty.GetComponent<Main>();
		}
			
	}
    
    /// <summary>
    /// Loads first scene.
    /// Added to Button in screening_fail.
    /// </summary>
    public void Return() 
    {
        
       UnityEngine.SceneManagement.SceneManager.LoadScene(first_scene_name);
    	
    	
    }
    

  }


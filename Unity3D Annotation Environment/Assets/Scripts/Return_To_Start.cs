// Writes user data and loads first scene

using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System;
using System.IO;

public class Return_To_Start : MonoBehaviour {
    GameObject main_empty;
	Main main_empty_script;

	

	public void Awake(){
		main_empty = GameObject.Find("mainEmpty");
		if(main_empty !=null){
			main_empty_script = main_empty.GetComponent<Main>();
		}
	
		
			
	}
    
    // Add to start button in editor
    public void Return() 
    {
        
    	
       // main_empty_script.begin();
       UnityEngine.SceneManagement.SceneManager.LoadScene("player_menu"); // Load scene at given level variable
    	
    	
    }
    

   }


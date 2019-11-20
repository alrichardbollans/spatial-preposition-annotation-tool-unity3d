// Writes user data and loads first scene

using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using System.Collections;
using System;
using System.IO;

public class Player_Menu_Main : MonoBehaviour {
    // These objects need creating in Unity and assigning in the canvas object that this script is attached to
    // public GameObject loadingImage;
  
    public GameObject nativeCheck;
    public GameObject nonnativeCheck;

    public GameObject instructions;


    // Create Permanent variables --- available in all contexts
    string userID;

  
    
   
    Toggle toggle_native;
    Toggle toggle_nonnative;
 
    Text instruction;

    // Variables to store values
    int scene_index;
    
    string now;
    
    
    

	void Start () {
        // loadingImage.SetActive(false);
        // Initialise permanent variables (Needs to be done in start function for some reason)
        userID = System.Guid.NewGuid().ToString();
        PlayerPrefs.SetString("UserID", userID); // Sets player ID to use later
 
        toggle_native = nativeCheck.GetComponent<Toggle>();
        toggle_nonnative = nonnativeCheck.GetComponent<Toggle>();
        // sceneInput = sceneMenu.GetComponent<Dropdown>();
        instruction = instructions.GetComponent<Text>();
        
        // Set toggles to false
		 // set toggle to false
		toggle_native.isOn = false;
		 // set toggle to false
		toggle_nonnative.isOn = false;

		////// Set dropdown menu to list of scenes
		// string[] fileEntries = Directory.GetFiles("Assets/Scenes");
		// var info = new DirectoryInfo("Assets/Scenes");
		// var fileInfo = info.GetFiles();

		// sceneInput.options.Clear();
		// sceneInput.options.Add(new Dropdown.OptionData("Select scene..."));
		// foreach (string option in fileEntries)
		// {//Need to edit these a bit here
		// 	if (option.Contains("scene") && !option.Contains("meta")){
		//     sceneInput.options.Add(new Dropdown.OptionData(option));
		// }
		// }


	}
    
    void set_form_values(){ // set variables to current value of form
       
        now = DateTime.UtcNow.ToString("yyyyMMdd-HHMMss");
    }

    bool check_form(){ 
        //Checks if form elements have been filled in
       

        set_form_values();

        
        if (toggle_native.isOn == false && toggle_nonnative.isOn == false) {
        	// pobInput.placeholder.color = Color.red;
        	instruction.color = Color.red;
        	instruction.text = "Please fill in the details before continuing - Are you a native speaker?";
        	return false;
        }
       
        else { return true;}
    }

    string authenticate(string username, string password)
    {
        string auth = username + ":" + password;
        auth = System.Convert.ToBase64String(System.Text.Encoding.GetEncoding("ISO-8859-1").GetBytes(auth));
        auth = "Basic " + auth;
        return auth;
    }

    IEnumerator sendUserTextToFile_then_loadscene(){
        string authorization = authenticate("game", "6KK6w4EhgcrBQHKzgL");
        string url = "/spatial_language_study/writeuserdata.php";
        yield return null;
        /// Set details
        set_form_values();
        
        
        
        
      
        


        bool successful = true; //Not yet used here

        WWWForm form = new WWWForm(); //create web form to talk with
        // form.AddField("name",user_name);
        // form.AddField("pob",pob);
        form.AddField("userid",userID);
        // form.AddField("year",year);
        form.AddField("now",now);
        // form.AddField("gender",gender);
        if (toggle_native.isOn == true && toggle_nonnative.isOn == false)
        {
            form.AddField("native","1");
        }
        else
        {
            form.AddField("native","0");
        }
        // Send the form to the php script
        // Upload to a cgi script
        using (var w = UnityWebRequest.Post(url, form))
        {
            w.SetRequestHeader("AUTHORIZATION",authorization);
            yield return w.SendWebRequest();
            if (w.isNetworkError || w.isHttpError) {
                print(w.error);
            }
            
        }

        UnityEngine.SceneManagement.SceneManager.LoadScene("main");
    }
	
    // Add to start button in editor
    public void WriteandLoadScene() 
    {
        
    	set_form_values();
    	if (check_form()){
            // loadingImage.SetActive(true); // Show 'Loading' image

            StartCoroutine(sendUserTextToFile_then_loadscene()); //Need startcoroutine (not sure yet what it does)
            
             // Load scene at given level variable
    	}
    	
    }
    

   }


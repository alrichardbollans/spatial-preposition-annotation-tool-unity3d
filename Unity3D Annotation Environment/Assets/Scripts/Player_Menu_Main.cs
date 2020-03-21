// Attached to Canvas in player_menu scene
// Writes user data and loads first scene

using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using System.Collections;
using System;
using System.IO;

public class Player_Menu_Main : MonoBehaviour {
    public GameObject nativeCheck; //Native toggle
    public GameObject nonnativeCheck; // NonNative toggle
    public GameObject instructions; // Instructions

    static string main_scene_name = "main";
    static string auth_username = "game";
    static string auth_password =  "6KK6w4EhgcrBQHKzgL";
    static string url_writeuserdata = "/spatial_language_study/writeuserdata.php";

    string userID;

    Toggle toggle_native;
    Toggle toggle_nonnative;
 
    Text instruction;

    int scene_index;
    
    string now;

    /// <summary>
    /// Initialise permanent variables.
    /// Set userID. Gets components and sets input toggles to false.
    /// </summary>
	void Start () {

        userID = System.Guid.NewGuid().ToString();
        PlayerPrefs.SetString("UserID", userID); // Sets player ID to use later
 
        toggle_native = nativeCheck.GetComponent<Toggle>();
        toggle_nonnative = nonnativeCheck.GetComponent<Toggle>();
        instruction = instructions.GetComponent<Text>();
        
        // Set toggles to false		
		toggle_native.isOn = false;
		toggle_nonnative.isOn = false;

	}
    
    /// <summary>
    /// Set now to current time.
    /// </summary>
    void set_form_values(){
       
        now = DateTime.UtcNow.ToString("yyyyMMdd-HHMMss");
    }

    /// <summary>
    /// Checks if form elements have been filled in.
    /// </summary>
    /// <returns>
    /// True if form is ready to submit, otherwise False.
    /// </returns>
    bool check_form(){ 
        
        set_form_values();
        
        if (toggle_native.isOn == false && toggle_nonnative.isOn == false) {
        	// pobInput.placeholder.color = Color.red;
        	instruction.color = Color.red;
        	instruction.text = "Please fill in the details before continuing - Are you a native speaker?";
        	return false;
        }
       
        else { return true;}
    }

    /// <summary>
    /// Gets string for authentication from username and password.
    /// </summary>
    /// <param name="username">The username.</param>
    /// <param name="password">The password.</param>
    /// <returns>
    /// Authentication string.
    /// </returns>
    string authenticate(string username, string password)
    {
        string auth = username + ":" + password;
        auth = System.Convert.ToBase64String(System.Text.Encoding.GetEncoding("ISO-8859-1").GetBytes(auth));
        auth = "Basic " + auth;
        return auth;
    }

    /// <summary>
    /// Writes user info to file then loads main scene.
    /// </summary>
    /// <returns>
    /// Authentication string.
    /// </returns>
    IEnumerator sendUserTextToFile_then_loadscene(){
        string authorization = authenticate(auth_username, auth_password);
        yield return null;
        /// Set details
        set_form_values();
        
        bool successful = true; //Not yet used here

        //create web form to talk with
        WWWForm form = new WWWForm(); 
        form.AddField("userid",userID);
        form.AddField("now",now);
        
        if (toggle_native.isOn == true && toggle_nonnative.isOn == false)
        {
            form.AddField("native","1");
        }
        else
        {
            form.AddField("native","0");
        }
        // Send the form to the php script to write to server
        // Upload to a cgi script
        using (var w = UnityWebRequest.Post(url_writeuserdata, form))
        {
            w.SetRequestHeader("AUTHORIZATION",authorization);
            yield return w.SendWebRequest();
            if (w.isNetworkError || w.isHttpError) {
                print(w.error);
            }
            
        }

        UnityEngine.SceneManagement.SceneManager.LoadScene(main_scene_name);
    }
	
    /// <summary>
    /// Starts Coroutine to write user info.
    /// Added to StartButton in player_menu.
    /// </summary>
    public void WriteandLoadScene() 
    {
        
    	set_form_values();
    	if (check_form()){

            StartCoroutine(sendUserTextToFile_then_loadscene()); //Need startcoroutine (not sure yet what it does)

    	}
    	
    }
    

   }


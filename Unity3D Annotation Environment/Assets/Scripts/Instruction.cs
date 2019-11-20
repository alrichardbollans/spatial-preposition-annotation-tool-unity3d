//This handles the first scene shown for each task
// Instructions are displayed and then the task scenes are loaded
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

using System;
using System.IO;
using System.Text;
using System.Linq;

public class Instruction : MonoBehaviour {
	public GameObject title_obj;
	public GameObject instruction_obj;
	public GameObject ButtonText;

	public GameObject loadingImage;

	Text title_obj_text_component;
	Text instruction_text_component;
	Text ButtonText_text_component;

	GameObject main_empty;
	Main main_empty_script;


	
	string[] instruction_list;
	string instruction_title;
	

	public void Awake(){
		main_empty = GameObject.Find("mainEmpty");
		if(main_empty !=null){
			main_empty_script = main_empty.GetComponent<Main>();
		}

		

		var current_task = Main.task;//.name;

		if(current_task == null )
		{
			Debug.Log("Error. No task set");
			
		}
		else {
			var abv = current_task.name;
			instruction_list = current_task.instruction_list;
			instruction_title = current_task.instruction_title;
			Debug.Log(abv);
		}

	
		
			
	}

	public void Start(){
		loadingImage.SetActive(false);

		title_obj_text_component = title_obj.GetComponent<Text>();
		instruction_text_component = instruction_obj.GetComponent<Text>();
		ButtonText_text_component = ButtonText.GetComponent<Text>();

		instruction_text_component.text = instruction_list[0];
		title_obj_text_component.text = instruction_title;

		if(instruction_list.Length == 1){
			ButtonText_text_component.text = "Begin";
		}
	}

	public void load(){
		Debug.Log("load");
		// Get 'Main' instance
		UnityEngine.SceneManagement.SceneManager.UnloadSceneAsync("instruction");
		// main_empty_script.instructions_loaded = false;
		main_empty_script.load_next_scene();
		
		// UnityEngine.SceneManagement.SceneManager.LoadScene("main");
	}
	public void Back_click(){
		string active_instruction = instruction_text_component.text;
		int i = Array.FindIndex(instruction_list,w => w==active_instruction);
		
		if(i-1>=0){
			instruction_text_component.text = instruction_list[i-1];
		}

		if(ButtonText_text_component.text == "Begin"){
			ButtonText_text_component.text = "Next";
		}
	}
	public void Next_Click(){
		string active_instruction = instruction_text_component.text;
		int i = Array.FindIndex(instruction_list,w => w==active_instruction);
		
		// int i = instruction_list.IndexOf(active_instruction);
		if (i+1 < instruction_list.Length){
			
			instruction_text_component.text = instruction_list[i+1];
			if (i+2 >= instruction_list.Length){
				ButtonText_text_component.text = "Begin";
			}
		}



		else if(ButtonText_text_component.text == "Begin")
		{	
			loadingImage.SetActive(true);
			load();
		}
		
		
	}
}
// Attached to Canvas in instruction scene
// This handles the first scene shown for each task
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
	
	/// <summary>
	/// Awake is used to initialize any variables or game state before the game starts.
	/// Awake is called only once during the lifetime of the script instance.
	/// Get instruction information from task.
	/// </summary>
	public void Awake(){
		// Get main game script
		main_empty = GameObject.Find("mainEmpty");
		if(main_empty !=null){
			main_empty_script = main_empty.GetComponent<Main>();
		}

		var current_task = Main.task;

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
	/// <summary>
	///  Start is called exactly once in the lifetime of the script.
	///	 Deactivates loading image.
	///  Sets instruction and button texts.
	/// </summary>
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

	/// <summary>
	///  Unloads instruction scene and loads next text scene.
	/// </summary>
	public void load(){
		Debug.Log("load");
		
		UnityEngine.SceneManagement.SceneManager.UnloadSceneAsync("instruction");
		main_empty_script.load_next_scene();
		
	}

	/// <summary>
	/// Shows previous instruction.
	/// Attached to BackButton object.
	/// </summary>
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

	/// <summary>
	/// Shows next instruction.
	/// Attached to NextButton object.
	/// </summary>
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
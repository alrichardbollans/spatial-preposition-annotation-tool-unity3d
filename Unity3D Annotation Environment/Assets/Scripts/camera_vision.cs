// Manages camera
// Is set to not be able to move outside of the room given in the scene template
// Is added to main camera is scenes by Finalise_scenes.cs

using UnityEngine;
using System.Collections;
 
public class camera_vision : MonoBehaviour
{
 
	/*
	Camera Vision
		Adam Richard-Bollans 01/08/2019
		Based on Extended Flycam by Desi Quintans (CowfaceGames.com), 17 August 2012.
		Based on FlyThrough.js by Slin (http://wiki.unity3d.com/index.php/FlyThrough), 17 May 2011.
 
	LICENSE
		Free as in speech, and free as in beer.
 
	FEATURES
		WASD/Arrows:    Movement
		          Q:    Climb
		          E:    Drop
                  Shift:    Move faster
                  Control:    Move slower
                  End:    Toggle cursor locking to screen (you can also press Ctrl+P to toggle play mode on and off).
	*/
 	
	public static float cameraSensitivity = 0;
	public static float normalMoveSpeed = 0.5f;
 
	private float rotationX = 0.0f;
	private float rotationY = 0.0f;

	float bound_top = 4.88f;
	float bound_bottom = 0.05f;
	float bound_x_l = -4.78f;
	float bound_x_r = 4.78f;
	float bound_z_b = -4.78f;
	float bound_z_f = 4.78f;

	Camera camera_component;
 
	void Start ()
	{
		Cursor.visible = true;
		Cursor.lockState = CursorLockMode.None;
		// camera settings
		camera_component = gameObject.GetComponent<Camera>();
		camera_component.nearClipPlane = 0.05f;
		camera_component.fieldOfView = 70f;

		//Needed so the rotation starts in correct place
		rotationY = -transform.eulerAngles.x;

		// Debug.Log(transform.eulerAngles.x);

		

	}
 	
	/// <summary>
	/// Allow mouse to move camera.
	/// </summary>
	public static void allow_mouse_look_around(){
		cameraSensitivity = 60;
		Cursor.lockState = CursorLockMode.Locked;
		
	}

	/// <summary>
	/// Disable camera movement from mouse.
	/// </summary>
	public static void stop_mouse_look_around(){
		cameraSensitivity = 0;
		Cursor.lockState = CursorLockMode.None;
		
	}

	/// <summary>
	/// Disable all camera movement.
	/// </summary>
	public static void freeze_camera(){
		normalMoveSpeed = 0f;
		cameraSensitivity = 0;
		Cursor.lockState = CursorLockMode.None;
		
	}

	/// <summary>
	/// Enable camera movement.
	/// </summary>
	public static void unfreeze_camera(){
		normalMoveSpeed = 0.5f;
		
	}

	/// <summary>
	/// Get transform to increment position by.
	/// </summary>
	/// <param name="direction">Horizontal direction to consider</param>
	/// <returns>
	/// Vector3 to transform distance by.
	/// </returns>
	Vector3 get_new_transform_plus(string direction){
		Vector3 transform_new = new Vector3(0f,0f,0f);
		
			

			if(direction == "forward"){
				
				transform_new.x = transform.forward.x;
				transform_new.z = transform.forward.z;
			}
			else{
				transform_new.x = transform.right.x;
				transform_new.z = transform.right.z;
			}

			if(transform.position.x>=bound_x_r){
				if(transform_new.x > 0){
					transform_new.x = 0f;
				}
			}

			else if(transform.position.x<=bound_x_l){
				if(transform_new.x < 0){
					transform_new.x = 0f;
				}
			}

			if(transform.position.z>=bound_z_f){
				if(transform_new.z>0){
					transform_new.z = 0f;
				}
			}

			else if(transform.position.z<=bound_z_b){
				if(transform_new.z<0){
					transform_new.z = 0f;
				}
			}

		return transform_new;
	}

	/// <summary>
	/// Get transform to increment position by.
	/// </summary>
	/// <param name="direction">Horizontal direction to consider</param>
	/// <returns>
	/// Vector3 to transform distance by.
	/// </returns>
	/// <remarks>
	/// Differs from get_new_transform_plus by how it manages bounding the camera.
	/// </remarks>
	Vector3 get_new_transform_minus(string direction){
		Vector3 transform_new = new Vector3(0f,0f,0f);
		
			

		if(direction == "forward"){
			
			transform_new.x = transform.forward.x;
			transform_new.z = transform.forward.z;
		}
		else{
			transform_new.x = transform.right.x;
			transform_new.z = transform.right.z;
		}

		if(transform.position.x>=bound_x_r){
			if(transform_new.x < 0){
				transform_new.x = 0f;
			}
		}

		if(transform.position.x<=bound_x_l){
			if(transform_new.x > 0){
				transform_new.x = 0f;
			}
		}

		if(transform.position.z>=bound_z_f){
			if(transform_new.z<0){
				transform_new.z = 0f;
			}
		}

		if(transform.position.z<=bound_z_b){
			if(transform_new.z>0){
				transform_new.z = 0f;
			}
		}


		return transform_new;
	}

	void Update ()
	{	
		if(Main.task != null){
			if(Main.task.allow_camera_movement){
				// If cameraSensitivity not zero, allow movement with mouse.
				if(cameraSensitivity !=0){
					

					

					rotationX += Input.GetAxis("Mouse X") * cameraSensitivity * Time.deltaTime;
					rotationY += Input.GetAxis("Mouse Y") * cameraSensitivity * Time.deltaTime;
					// rotationY = Mathf.Clamp (rotationY, -90, 90);
			 
					transform.rotation = Quaternion.AngleAxis(rotationX, Vector3.up);
					transform.rotation *= Quaternion.AngleAxis(rotationY, Vector3.left);
				 }
			 	
		 		// Keyboard movement
		 		// 1 and 2 move camera up and down on world axis
				if (Input.GetKey (KeyCode.Alpha2) || Input.GetKey (KeyCode.Keypad2)) {
					if (transform.position.y<bound_top){
						transform.position += Vector3.up * normalMoveSpeed * Time.deltaTime;
					}
				}
				if (Input.GetKey (KeyCode.Alpha1) || Input.GetKey (KeyCode.Keypad1)) {
					if (transform.position.y>bound_bottom){
						transform.position -= Vector3.up * normalMoveSpeed * Time.deltaTime;
					}
				}
				// Arrow keys move camera on local camera axes, only along x,z axes.
				if (Input.GetKey (KeyCode.UpArrow)) {
					Vector3 transform_new;
					transform_new = get_new_transform_plus("forward");

					transform.position += transform_new * normalMoveSpeed * Time.deltaTime;
				}
				if (Input.GetKey (KeyCode.DownArrow)) {
					Vector3 transform_new;
					transform_new = get_new_transform_minus("forward");
					

					transform.position -= transform_new * normalMoveSpeed * Time.deltaTime;
				}
				if (Input.GetKey (KeyCode.RightArrow)) {
					Vector3 transform_new;
					transform_new = get_new_transform_plus("right");

					
					transform.position += transform_new * normalMoveSpeed * Time.deltaTime;
				}
				if (Input.GetKey (KeyCode.LeftArrow)) {
					Vector3 transform_new;
					transform_new = get_new_transform_minus("right");

					
					transform.position -= transform_new * normalMoveSpeed * Time.deltaTime;
				}

				// 0 toggles allowing moust to look around.
		 		if (Input.GetKey (KeyCode.Alpha0) || Input.GetKey (KeyCode.Keypad0)){

					
					allow_mouse_look_around();
					

				}

				if (!(Input.GetKey (KeyCode.Alpha0) || Input.GetKey (KeyCode.Keypad0))){
					
					stop_mouse_look_around();
					

				}
			}
		}
		
		
		
	}
}

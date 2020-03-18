//

using UnityEngine;
using System;
using System.Linq;

using System.Collections;
using System.Collections.Generic;
using System.IO;

#if UNITY_EDITOR
using UnityEditor;

public class WaitingGame : MonoBehaviour
{	
	GameObject[] allObjects; 
	List<GameObject> meshObjects = new List<GameObject>();

	IEnumerator print_positions(){
		using(StreamWriter file1 = new StreamWriter("old positions.csv")){
			foreach(GameObject obj in meshObjects){
				string p = obj.transform.position.ToString();
				string r = obj.transform.rotation.ToString();

				string[] row = {obj.scene.name, obj.name, p,r};
				string row_csv_string = String.Join(";", row);//"\"" + string.Join("\",\"", row) + "\"";//String.Join(",", row);
				
				// Debug.Log(row_csv_string);
				file1.WriteLine(row_csv_string);
			}
		}
		// Wait 10 seconds
		yield return new WaitForSeconds(5);
		
		using(StreamWriter file = new StreamWriter("positions.csv")){
			foreach(GameObject obj in meshObjects){
				string p = obj.transform.position.ToString("G6");
				string r = obj.transform.rotation.ToString("G6");

				string[] row = {obj.scene.name, obj.name, p,r};
				string row_csv_string = String.Join(";", row);//"\"" + string.Join("\",\"", row) + "\"";//String.Join(",", row);
				
				// Debug.Log(row_csv_string);
				file.WriteLine(row_csv_string);
			}
		}
		
		Application.Quit();
		#if UNITY_EDITOR
        //Stop playing the scene
        UnityEditor.EditorApplication.isPlaying = false;
	    #endif
	    Debug.Log("Waiting Script Attached to:");
	    Debug.Log(gameObject.name);
	    write_transforms.written = true;
	}

	void run_pp(){
		// Debug.Log(Time.time);
		StartCoroutine(print_positions());
		// Debug.Log(Time.time);
	}

	void Start(){
		allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		// Populate mesh list
		foreach(GameObject obj in allObjects){
            MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
            if(mf != null){
                meshObjects.Add(obj);
            }
        }
		run_pp();

	}
}

#endif
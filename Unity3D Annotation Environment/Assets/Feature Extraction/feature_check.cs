// Script to make it easier to check features in the editor.
// Adds script to each mesh object. 
// Each object is treated as a ground and can change the given figure to see value.
#if UNITY_EDITOR
using UnityEngine;

using UnityEditor;
using UnityEditor.SceneManagement;


using System;
using System.Linq;
using System.IO;
using System.Collections;
using System.Collections.Generic;

[ExecuteInEditMode]
public class feature_check : MonoBehaviour
{
    public string support;
    public string contact_proportion;
    public string below_proportion;
    public string above_proportion;
    public string f_covers_g;
    public string g_covers_f;
    public string bbox_overlap_proportion;
    public string horizontal_distance;
    
    static GameObject[] allObjects;
	static List<GameObject> meshObjects = new List<GameObject>();
	public string[] relation_keys;

	public GameObject figure;
    
	void Awake(){
		allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		foreach(GameObject obj in allObjects){
			if(!obj.name.Contains("wall") && !obj.name.Contains("ceiling") && !obj.name.Contains("floor")){
								
                MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
                if(mf != null){
                    meshObjects.Add(obj);                    
                    
                }
            }
        }
        figure = meshObjects[0];
        Relationship r = new Relationship("","","");
        relation_keys = r.relation_keys.ToArray();
        

	}
    void Update()
    {
        
        string ground = this.gameObject.name;
        
        //    Create relationship instance to read values
    	Relationship r = new Relationship(this.gameObject.scene.name,figure.name,ground);
    	r.load_from_csv();
    	support = r.relation_dictionary["support"].ToString();
    	contact_proportion = r.relation_dictionary["contact_proportion"].ToString();
    	below_proportion = r.relation_dictionary["below_proportion"].ToString();
    	above_proportion = r.relation_dictionary["above_proportion"].ToString();
    	f_covers_g = r.relation_dictionary["f_covers_g"].ToString();
    	g_covers_f = r.relation_dictionary["g_covers_f"].ToString();
    	bbox_overlap_proportion = r.relation_dictionary["bbox_overlap_proportion"].ToString();
        horizontal_distance = r.relation_dictionary["horizontal_distance"].ToString();
    	// Debug.Log("Figure: " + t);
    	// Debug.Log(r.relation_dictionary["support"]);
    }
}

[CustomEditor(typeof(feature_check))]
[CanEditMultipleObjects]
public class feature_checkEditor : Editor 
{	
	// Each feature must be added here as below
	// Not too sure why
    SerializedProperty figure;
    SerializedProperty support;
    SerializedProperty contact_proportion;
    SerializedProperty below_proportion;
    SerializedProperty f_covers_g;
    SerializedProperty g_covers_f;
    SerializedProperty bbox_overlap_proportion;
    SerializedProperty above_proportion;
    SerializedProperty horizontal_distance;
    
    
    void OnEnable()
    {
        figure = serializedObject.FindProperty("figure");
        support = serializedObject.FindProperty("support");
        contact_proportion = serializedObject.FindProperty("contact_proportion");
        below_proportion = serializedObject.FindProperty("below_proportion");
        above_proportion = serializedObject.FindProperty("above_proportion");
        f_covers_g = serializedObject.FindProperty("f_covers_g");
        g_covers_f = serializedObject.FindProperty("g_covers_f");
        bbox_overlap_proportion = serializedObject.FindProperty("bbox_overlap_proportion");
        horizontal_distance = serializedObject.FindProperty("horizontal_distance");
            }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();
        EditorGUILayout.PropertyField(figure);
        EditorGUILayout.PropertyField(support);
        EditorGUILayout.PropertyField(contact_proportion);
        EditorGUILayout.PropertyField(below_proportion);
        EditorGUILayout.PropertyField(above_proportion);
        EditorGUILayout.PropertyField(f_covers_g);
        EditorGUILayout.PropertyField(g_covers_f);
        EditorGUILayout.PropertyField(bbox_overlap_proportion);
        EditorGUILayout.PropertyField(horizontal_distance);
        serializedObject.ApplyModifiedProperties();
    }
}


public class add_check_script : EditorWindow{
	static GameObject[] allObjects;
	static List<GameObject> meshObjects = new List<GameObject>();
	
	static public void main(){

		meshObjects.Clear();
		allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
		// Populate mesh list
		foreach(GameObject obj in allObjects){
			if(!obj.name.Contains("wall") && !obj.name.Contains("ceiling") && !obj.name.Contains("floor")){
	            MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
	            if(mf != null){
	            	
	                meshObjects.Add(obj);
	            }
	        }
        }
        foreach(GameObject go in meshObjects){
        	var check_script = go.GetComponent(typeof(feature_check)) as feature_check;
			if (check_script == null){
			    check_script = go.AddComponent<feature_check>();
			    
			}
        }
 	
        
	}

	
	[MenuItem ("My Tools/Add Feature Check Script")]

	static void init(){
		
		

		main();

		
	}
}

public class remove_check_script : EditorWindow{
	static GameObject[] allObjects;
	
	[MenuItem ("My Tools/Remove Feature Check Script")]

	static void init(){
        // Search for script and destroy.
		allObjects = UnityEngine.Object.FindObjectsOfType<GameObject>();
                
        foreach(GameObject go in allObjects){
            var check_script = go.GetComponent(typeof(feature_check)) as feature_check;
            if (check_script != null){
                UnityEngine.Object.DestroyImmediate(check_script);
            }
        }

		
	}
}

#endif




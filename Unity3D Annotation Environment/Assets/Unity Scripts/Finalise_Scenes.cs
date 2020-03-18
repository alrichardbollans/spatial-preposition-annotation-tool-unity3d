/// This script adds all scenes in the MainFolder directory to the buildsettings
// Should be run after all scenes have been created
/// Also edits Main.cs to add the scenes to the scene list
/// Also adds camera vision script to objects tagged 'MainCamera' and removes their audio listeners
// Bakes lighting in each scene

using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif

using System.IO;
using System.Collections;
using System.Collections.Generic;
 #if UNITY_EDITOR
public class Finalise_Scenes : EditorWindow
{  
    [MenuItem ("My Tools/Add Scenes To Build !!SAVE Scripts FIRST!!")]
   
   

    static void Init ()
    {   
        EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo();
        // Should add a check if main scenes contain any objects of same name
        // Add all scenes in folder to build settings
        List<EditorBuildSettingsScene> editorBuildSettingsScenes = new List<EditorBuildSettingsScene>();
        List<string> SceneList =  new List<string> ();
        List<string> ScenePathList =  new List<string> ();
        string MainFolder   = "Assets/Scenes";
        string MainScene = MainFolder + "/main.unity";
        

        DirectoryInfo d = new DirectoryInfo(@MainFolder);
        FileInfo[] Files = d.GetFiles("*.unity"); //Getting unity files
        
        foreach(FileInfo file in Files )
        {
            
            Debug.Log ("file name:" + file.Name);
            SceneList.Add(file.Name);
            
        }
        
        
       
       
        int i = 0;
        // Add player menu first
        for (i = 0; i < SceneList.Count; i ++)
        {
            if (SceneList[i].Contains("player_menu")){
                string scenePath = MainFolder + "/" + SceneList[i];
                ScenePathList.Add(scenePath);
                // Debug.Log ("i = " + i);
                // Debug.Log("scene path:" + scenePath);
                editorBuildSettingsScenes.Add(new EditorBuildSettingsScene(scenePath, true));
            }
           
        }
        // Add the rest to build menu
        for (i = 0; i < SceneList.Count; i ++)
        {
            if (!SceneList[i].Contains("example") &&!SceneList[i].Contains("template") &&!SceneList[i].Contains("test") && !SceneList[i].Contains("player menu")){
                string scenePath = MainFolder + "/" + SceneList[i];
                ScenePathList.Add(scenePath);
                // Debug.Log ("i = " + i);
                // Debug.Log("scene path:" + scenePath);
                editorBuildSettingsScenes.Add(new EditorBuildSettingsScene(scenePath, true));
            }
           
        }
       
        EditorBuildSettings.scenes = editorBuildSettingsScenes.ToArray();
    
        
        GameObject[] cameras;

        // Iterate through all scenes to make edits
        for (i = 0; i < SceneList.Count; i ++)
        {
            string scenePath = MainFolder + "/" + SceneList[i];
            Debug.Log ("i = " + i);
            Debug.Log("scene path:" + scenePath);
            string scene_name;
            scene_name = SceneList[i].Substring(0,SceneList[i].LastIndexOf(".unity"));
            
            EditorSceneManager.OpenScene(scenePath);
            
            EditorSceneManager.SetActiveScene(EditorSceneManager.GetSceneByName(scene_name));
            
            cameras = GameObject.FindGameObjectsWithTag("MainCamera");
            // Add camera vision script  and remove audio listener
            foreach(GameObject camera in cameras){
                Debug.Log ("camera = " + camera.name);
                if (camera.GetComponent<camera_vision>() == null){
                    camera.AddComponent<camera_vision>();
                }
                // Destroy(camera.GetComponent<AudioListener>());
                // Camera c;
                // c = camera.GetComponent<Camera>();
                camera.GetComponent<AudioListener>().enabled = false;

                Camera camera_component;
                camera_component = camera.GetComponent<Camera>();
                camera_component.nearClipPlane = 0.05f;
                camera_component.fieldOfView = 70f;
               

            }
            GameObject[] allObjects = Object.FindObjectsOfType<GameObject>();
            // Edit objects
            foreach(GameObject obj in allObjects){
                MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
                if(mf != null){
                    MeshObject mobj = new MeshObject(obj);
                    
                    mobj.prepare_physics_for_game();
                }
                
                var waiting_script = obj.GetComponent(typeof(WaitingGame)) as WaitingGame;
                if (waiting_script != null){
                    DestroyImmediate(waiting_script);
                }

                Rigidbody r = obj.GetComponent(typeof(Rigidbody)) as Rigidbody;
                if (r != null){
                    DestroyImmediate(r);
                }

                Animator a = obj.GetComponent(typeof(Animator)) as Animator;
                if (a != null){
                    DestroyImmediate(a);
                }
            }
            
            // Edit lighting in scene
            Light[] lights = Object.FindObjectsOfType<Light>();
            foreach(Light l in lights){
                l.shadows= LightShadows.Soft;
                // l.lightmapBakeType = LightmapBakeType.Realtime;
            }

            //Bake Lighting settings

            Lightmapping.giWorkflowMode = Lightmapping.GIWorkflowMode.OnDemand;
            LightmapEditorSettings.bakeResolution = 1f;
            Lightmapping.realtimeGI = false; // Realtime GI does not show up on WebGL build
            Lightmapping.Bake();
            
            Debug.Log ("Saving active scene");
            // EditorSceneManager.SaveOpenScenes();
            EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene());
           
        }
        

    // Edit main script to add scenes
        string ScriptFolder = "Assets/Scripts";
        string ScriptName = "Main.cs";
        string ScriptFile = ScriptFolder + "/" + ScriptName;

        string oldscenetext = "public static string[] input_list_of_scenes = {";
        string newscenetext = "    public static string[] input_list_of_scenes = {";

        string line;

        using(StreamReader file =  new StreamReader(ScriptFile)){
            while((line = file.ReadLine()) != null)
               {
                  if(line.Contains(newscenetext))
                  {
                     oldscenetext = line;
                  }
               }
            }
        

        for (i=0; i<SceneList.Count; i ++)
        {  
            string scenename = SceneList[i].Substring(0,SceneList[i].LastIndexOf(".unity"));
            newscenetext = newscenetext + '"'+ scenename + '"' + ",";
        }

        newscenetext = newscenetext.Substring(0,newscenetext.LastIndexOf(","));
        newscenetext = newscenetext + "};";

        string text = File.ReadAllText(ScriptFile);
        text = text.Replace(oldscenetext,newscenetext);
        File.WriteAllText(ScriptFile,text);
    

    }
   
}
#endif
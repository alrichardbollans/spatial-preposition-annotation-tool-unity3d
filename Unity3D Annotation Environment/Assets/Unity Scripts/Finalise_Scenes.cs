// This script finalises scenes and should be executed via menuitem once all scenes have been finished.

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
    
    /// <summary>
    /// Standardises scene name by removing ".unity".
    /// </summary>
    /// <param name="scene">The name of the scene.</param>
    /// <returns>
    /// String with ".unity" removed.
    /// </returns>
    static string simplify_scene_name(string scene){
        string simplified_scene = scene;
        if(scene.Contains(".unity")){
            simplified_scene = scene.Substring(0,scene.LastIndexOf(".unity"));
        }
        return simplified_scene;
    }

    /// <summary>
    /// Checks if scene should be added to build, based on declared non_test_scenes.
    /// </summary>
    /// <param name="scene">The name of the scene.</param>
    /// <returns>
    /// True, if should be included, otherwise False.
    /// </returns>
    static bool is_scene_to_add_to_build(string scene){
        string simplified_scene = simplify_scene_name(scene);
        
        if (!Main.non_test_scenes.Contains(simplified_scene)){
            return true;
        }
        else{
            return false;
        }
    }

    /// <summary>
    /// Gets scenepath of a given scene.
    /// </summary>
    /// <param name="scene_file">The name of the scene.</param>
    /// <returns>
    /// String path for scene.
    /// </returns>
    static string get_scene_path(string scene_file){
        string simplified_scene = simplify_scene_name(scene_file);

        return Main.MainFolder + "/" + simplified_scene+ ".unity";
        
    }

    static void make_edits_to_scene(string ith_file){
        string scenePath = get_scene_path(ith_file);
        string scene_name =  simplify_scene_name(ith_file);
        
        EditorSceneManager.OpenScene(scenePath);
        
        EditorSceneManager.SetActiveScene(EditorSceneManager.GetSceneByName(scene_name));
        
        GameObject[] cameras;
        // Edit Main camera properties
        cameras = GameObject.FindGameObjectsWithTag(Main.main_camera_tag);
        
        foreach(GameObject camera in cameras){
            
            if (camera.GetComponent<camera_vision>() == null){
                camera.AddComponent<camera_vision>();
            }
            
            camera.GetComponent<AudioListener>().enabled = false;

            Camera camera_component;
            camera_component = camera.GetComponent<Camera>();
            camera_component.nearClipPlane = 0.05f;
            camera_component.fieldOfView = 70f;
           

        }
        // Make edits to objects
        GameObject[] allObjects = Object.FindObjectsOfType<GameObject>();
        
        foreach(GameObject obj in allObjects){
            MeshFilter mf = obj.GetComponent(typeof(MeshFilter)) as MeshFilter;
            if(mf != null){
                MeshObject mobj = new MeshObject(obj);
                
                mobj.prepare_physics_for_game();
                obj.isStatic = true;
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
            if(l.lightmapBakeType == LightmapBakeType.Realtime){
                // Make shadowbias 0 so close objects recieve shadows.
                l.shadowBias = 0;
            }
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

    // Add menu item in editor.
    [MenuItem ("My Tools/Add Scenes To Build !!SAVE Scripts FIRST!!")]
   
   
    /// <summary>
    /// Adds all scenes in "Assets/Scenes" to build. Player menu is set as first scene.
    /// Iterates through all scenes to make edits.
    /// Adds camera_vision script and edit some settings of each object with MainCamera tag.
    /// For each object, removes rigid body, animator and WaitingGame script. If object is mesh object
    /// edits physics properties for in-game.
    /// Bakes lighting.
    /// Then saves scene.
    /// Edits "Main.cs" script to include scenes in scene input list.
    /// </summary>
    /// <remarks>
    /// Should add a check if main scenes contain any objects of same name.
    /// 
    /// </remarks>
    static void Init ()
    {   
        
        // Give user chance to save scenes.
        EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo();
        
        // Add all scenes in main folder to build settings.
        List<EditorBuildSettingsScene> editorBuildSettingsScenes = new List<EditorBuildSettingsScene>();
        List<string> SceneList =  new List<string> ();
        
        //Getting unity files.
        DirectoryInfo d = new DirectoryInfo(@Main.MainFolder);
        FileInfo[] Files = d.GetFiles("*.unity"); 
        
        // Create scene list of scene names.
        foreach(FileInfo file in Files )
        {
            
            Debug.Log ("file name:" + file.Name);
            SceneList.Add(file.Name);
            
        }
        
        // Add player menu first.
        string firstscenePath = get_scene_path(Main.first_scene_name);
        editorBuildSettingsScenes.Add(new EditorBuildSettingsScene(firstscenePath, true));
        
        int i = 0;
        // Add the rest to build menu.
        for (i = 0; i < SceneList.Count; i ++)
        {
            if (is_scene_to_add_to_build(SceneList[i]) && !SceneList[i].Contains(Main.first_scene_name)){
                string scenePath = get_scene_path(SceneList[i]);
                editorBuildSettingsScenes.Add(new EditorBuildSettingsScene(scenePath, true));
            }
           
        }
        // Finally, modify build settings.
        EditorBuildSettings.scenes = editorBuildSettingsScenes.ToArray();
    
        
        

        // Iterate through all scenes to make edits.
        for (i = 0; i < SceneList.Count; i ++)
        {
            make_edits_to_scene(SceneList[i]);
           
        }
        

        // Edit main script to add scenes.
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
            string scenename =  simplify_scene_name(SceneList[i]);
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
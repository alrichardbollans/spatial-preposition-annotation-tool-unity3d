
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif

using System.IO;
using System.Collections;
using System.Collections.Generic;
 #if UNITY_EDITOR
public class screenshots_for_typicality : EditorWindow
{   
    // Resolution for screenshots.
    static public int resWidth = 512; 
    static public int resHeight = 768;

    // Main camera is scene.
    static Camera MainCamera;
    static Rect new_rect1;

    static TaskScene task_scene;
    static string task_name = "sv";



    public static string ScreenShotPath(TaskScene scene, string figure, string ground) {
        string direct = Application.dataPath + "/Resources/typ_task_folder/";
        string r =  direct + TaskScene.ScreenShotName(scene.name,figure,ground);
        return r;
    }

    /// <summary>
    /// Takes screenshot from camera and saves to file.
    /// </summary>
    /// <param name="file">File to save to.</param>
    public static void take_screenshot(string file){
        RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
        MainCamera.targetTexture = rt;
        Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
        MainCamera.Render();
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);
        MainCamera.targetTexture = null;
        RenderTexture.active = null; // JC: added to avoid errors
        
        byte[] bytes = screenShot.EncodeToPNG();
        
        System.IO.File.WriteAllBytes(file, bytes);
        
        
    }
    /// <summary>
    /// Checks if Gameobject is visible from given camera.
    /// </summary>
    /// <param name="go">Gameobject to check.</param>
    /// <returns>
    /// True if object sufficiently in view, else false.
    /// </returns>
    public static bool IsObjectVisible(GameObject go)
    {   
        // Get bounds of object.
        MeshRenderer mr = go.GetComponent<MeshRenderer>();
        Bounds b = mr.bounds;

        // Create small bounds in corner of object to check
        Bounds low_bound = new Bounds(b.min, new Vector3(0.1f, 0.1f, 0.1f));
        Bounds high_bound = new Bounds(b.max, new Vector3(0.1f, 0.1f, 0.1f));

        // Chaneg camera planes to fit resolution.
        Rect old_rect;
        old_rect = MainCamera.rect;
        new_rect1 = MainCamera.rect;
        new_rect1.width = 0.5f;
        
        MainCamera.rect = new_rect1;

        bool l = GeometryUtility.TestPlanesAABB(GeometryUtility.CalculateFrustumPlanes(MainCamera), low_bound);
        bool h = GeometryUtility.TestPlanesAABB(GeometryUtility.CalculateFrustumPlanes(MainCamera), high_bound);
        // Revert camera planes.
        MainCamera.rect = old_rect;
        bool r;
        if(l && h){
            // Debug.Log("I see you");
            r= true;
        }
        else{
            // Debug.Log("Where are you");
            r=false;
        }
        return r;
    }

    static void zoom_in(float factor){
        float m = -0.1f*factor;
        
        Vector3 transform_new = new Vector3(0f,0f,m);
        MainCamera.transform.position -= transform_new;
    }

    static void zoom_out(float factor){
        float m = -0.1f*factor;
        
        Vector3 transform_new = new Vector3(0f,0f,m);
        MainCamera.transform.position += transform_new;
    }

    static void position_camera(GameObject fig, GameObject gr){
        
        
        // First zoom in while the object is visible.
        while(IsObjectVisible(fig) && IsObjectVisible(gr)){
            zoom_in(1);
            // Debug.Log("Zoom in while");
        }

        // Zoom out.
        while(!IsObjectVisible(fig) || !IsObjectVisible(gr)){
            zoom_out(1);
            // Debug.Log("Zoom out while");
        }

        // Move couple more times for providence.
        zoom_out(2f);
        // extend_fov();
        

    }
    static void extend_fov(){
        Camera camera_component;
        camera_component = MainCamera.GetComponent<Camera>();
        camera_component.fieldOfView = 90f;
    }
    static void take_screenshots_all_configs(){
        
        // Tidy this up.
        task_scene.instantiate_after_scene_loaded();

        bool x = task_scene.set_new_example();
        MainCamera  = task_scene.main_camera;

        
        

        
        while(x){
            

            GameObject f = task_scene.active_configuration[0];
            
            GameObject g = task_scene.active_configuration[1];
            
            

            position_camera(f,g);
            
            
            string newname= ScreenShotPath(task_scene, f.name, g.name);
            
            
            take_screenshot(newname);

            // Try to get a new example, assign result to x.
            x = task_scene.set_new_example();
            
        }
    }

    // Add menu item in editor.
    [MenuItem ("My Tools/Get Screenshots for typicality task")]
   
   
    /// <summary>

    /// </summary>
    /// <remarks>
    /// </remarks>
    static void Init ()
    {   
        Main.clear_object_player_prefs();
        Scenes scene_info = new Scenes();

        // Iterate through all scenes and do the things.
        int i;
       
        for (i = 0; i < scene_info.SceneList.Count; i ++)
        {
            
            MyScene next_scene = scene_info.SceneList[i];
            
            next_scene.open_set_active();

            task_scene = new TaskScene(next_scene.name,task_name);
            take_screenshots_all_configs();
            
            
   
        }
        
        // task_scene = new TaskScene("scene_template",task_name);
        // take_screenshots_all_configs();

        Debug.Log("Completed");
    }
   
}
#endif
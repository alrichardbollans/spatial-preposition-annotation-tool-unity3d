// This is useful but causing a crash..
#if UNITY_EDITOR
using UnityEngine;

using UnityEditor;
using UnityEditor.SceneManagement;


using System.IO;
using System.Collections;
using System.Collections.Generic;

public class screenshots_for_typicality : EditorWindow
{   
    // Resolution for screenshots.
    static public int resWidth = 512; 
    static public int resHeight = 768;

    // Main camera is scene.
    static Camera MainCamera;
    static Rect new_rect1;

    static public GameObject fig;

    static public GameObject grd;

    static List<GameObject[]> configuration_list = new List<GameObject[]>();
    static GameObject[] active_configuration;

    
    public string prep;

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

    static void rotate(float xfactor,float yfactor){
       
        set_main_camera();
        MainCamera.transform.rotation *= Quaternion.AngleAxis(xfactor, Vector3.up);
        MainCamera.transform.rotation *= Quaternion.AngleAxis(yfactor, Vector3.left);
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
    
    public static string ScreenShotPath(string preposition, string figure, string ground) {
        string scene_name = EditorSceneManager.GetActiveScene().name;
        string direct = Application.dataPath + "/Resources/typ_task_folder/" + preposition +"/";
        string r =  direct + TypTask.ScreenShotName(scene_name,figure,ground);
        return r;
    }

    /// <summary>
    /// Takes screenshot from camera and saves to file.
    /// </summary>
    /// <param name="file">File to save to.</param>
    public static void take_screenshot(string file){
        set_main_camera();
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

    static void set_main_camera(){
        // Set camera
        GameObject[] cam_list = GameObject.FindGameObjectsWithTag(Main.main_camera_tag);
        string scene_name = EditorSceneManager.GetActiveScene().name;
        foreach(GameObject c in cam_list){

            if (c.scene.name == scene_name){
                MainCamera = c.GetComponent<Camera>();
            }
        }
    }

    static void unhighlight(){
        Task.unhighlight_figure(fig);
        Task.unhighlight_ground(grd);
    }

    static void  populate_config_list(){
        
        GameObject[] g_list = GameObject.FindGameObjectsWithTag(Main.ground_tag);
        GameObject[] f_list = GameObject.FindGameObjectsWithTag(Main.figure_tag);
        GameObject[] fg_list = GameObject.FindGameObjectsWithTag(Main.fig_grd_tag);

        List<GameObject> grd_list = new List<GameObject>();
        List<GameObject> fig_list = new List<GameObject>();

        foreach(GameObject gobj in g_list){
            grd_list.Add(gobj);
        }

        foreach(GameObject gobj in f_list){
            fig_list.Add(gobj);
        }

        foreach(GameObject gobj in fg_list){
            fig_list.Add(gobj);
            grd_list.Add(gobj);
        }
            
        foreach (GameObject ground in grd_list){
            foreach(GameObject fig in fig_list){
                if(fig.name != ground.name){
                    GameObject[] config = {fig,ground};
                    configuration_list.Add(config);
                }
            }
            
        }
    }

    static void set_next_fig_grd(){
        populate_config_list();
        Debug.Log(configuration_list.Count);
        if (configuration_list.Contains(active_configuration)){
            Debug.Log(active_configuration);
            // If there is an active configuration pick next configuration in list
            int i = configuration_list.IndexOf(active_configuration);
            
            // If there is a next one to pick do that and return true, else return false
            if (i+1 < configuration_list.Count){

                active_configuration = configuration_list[i+1];
                             
                
            }
            else{
                active_configuration = configuration_list[0];
            }
            
        }
        else if(configuration_list.Count >0) {
            // If there is no active configuration start with the first one in list
            active_configuration = configuration_list[0];
            
        }
        
        fig = active_configuration[0];
        grd = active_configuration[1];
        
    }
    // Add menu item in editor.
    [MenuItem ("My Tools/Take Screenshots")]
   
   
    /// <summary>

    /// </summary>
    /// <remarks>
    /// </remarks>
    static void Init ()
    {   
        var window = GetWindowWithRect<screenshots_for_typicality>(new Rect(50, 50, 300, 200));
        window.Show();

        configuration_list.Clear();
        
        
        // 

    }

    void OnGUI()
    {   

        EditorGUILayout.BeginVertical();
        // EditorGUIUtility.labelWidth=50f;
        // fig = EditorGUILayout.ObjectField("Figure:",fig, typeof(GameObject), true) as GameObject;
        
        // grd =EditorGUILayout.ObjectField("Ground:",grd, typeof(GameObject), true) as GameObject;
        EditorGUIUtility.labelWidth=80f;
        // i = EditorGUILayout.IntField("Screenshots#:", i);
        prep = EditorGUILayout.TextField("Preposition:", prep);
        EditorGUILayout.EndVertical();


        if (GUILayout.Button("Highlight Next!"))
        {
            set_main_camera();
            set_next_fig_grd();

            Task.highlight_figure(fig);
            Task.highlight_ground(grd);

        }
        if (GUILayout.Button("Unhighlight!"))
        {
            unhighlight();
        }
        if (GUILayout.Button("Search!"))
        {

            position_camera(fig,grd);
        }
        if (GUILayout.Button("Rotate Left!"))
        {
            rotate(-20f,0f);
        }
        if (GUILayout.Button("Rotate Right!"))
        {
            rotate(20f,0f);
        }
        if (GUILayout.Button("Rotate Up!"))
        {
            rotate(0f,20f);
        }
        if (GUILayout.Button("Rotate Down!"))
        {
            rotate(0f,-20f);
        }
        if (GUILayout.Button("Take shot!"))
        {	
        	
            
            


            string newname= ScreenShotPath(prep, fig.name, grd.name);
            
            
            take_screenshot(newname);
        }
    }
   
}


#endif
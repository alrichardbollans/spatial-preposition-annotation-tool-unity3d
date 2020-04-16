// This is useful but causing a crash..

// using UnityEngine;
// #if UNITY_EDITOR
// using UnityEditor;
// using UnityEditor.SceneManagement;
// #endif

// using System.IO;
// using System.Collections;
// using System.Collections.Generic;
//  #if UNITY_EDITOR
// public class screenshots_for_typicality : EditorWindow
// {   
//     // Resolution for screenshots.
//     static public int resWidth = 512; 
//     static public int resHeight = 768;

//     // Main camera is scene.
//     static Camera MainCamera;
//     static Rect new_rect1;


//     Material fig_mat = Resources.Load("figure_material", typeof(Material)) as Material;
//     Material grd_mat = Resources.Load("ground_material", typeof(Material)) as Material;

//     public GameObject fig;

//     public GameObject grd;

//     public int i;
//     public string prep;

//     /// <summary>
//     /// Checks if Gameobject is visible from given camera.
//     /// </summary>
//     /// <param name="go">Gameobject to check.</param>
//     /// <returns>
//     /// True if object sufficiently in view, else false.
//     /// </returns>
//     public static bool IsObjectVisible(GameObject go)
//     {   
//         // Get bounds of object.
//         MeshRenderer mr = go.GetComponent<MeshRenderer>();
//         Bounds b = mr.bounds;

//         // Create small bounds in corner of object to check
//         Bounds low_bound = new Bounds(b.min, new Vector3(0.1f, 0.1f, 0.1f));
//         Bounds high_bound = new Bounds(b.max, new Vector3(0.1f, 0.1f, 0.1f));

//         // Chaneg camera planes to fit resolution.
//         Rect old_rect;
//         old_rect = MainCamera.rect;
//         new_rect1 = MainCamera.rect;
//         new_rect1.width = 0.5f;
        
//         MainCamera.rect = new_rect1;

//         bool l = GeometryUtility.TestPlanesAABB(GeometryUtility.CalculateFrustumPlanes(MainCamera), low_bound);
//         bool h = GeometryUtility.TestPlanesAABB(GeometryUtility.CalculateFrustumPlanes(MainCamera), high_bound);
//         // Revert camera planes.
//         MainCamera.rect = old_rect;
//         bool r;
//         if(l && h){
//             // Debug.Log("I see you");
//             r= true;
//         }
//         else{
//             // Debug.Log("Where are you");
//             r=false;
//         }
//         return r;
//     }

//     static void zoom_in(float factor){
//         float m = -0.1f*factor;
        
//         Vector3 transform_new = new Vector3(0f,0f,m);
//         MainCamera.transform.position -= transform_new;
//     }

//     static void zoom_out(float factor){
//         float m = -0.1f*factor;
        
//         Vector3 transform_new = new Vector3(0f,0f,m);
//         MainCamera.transform.position += transform_new;
//     }

//     static void rotate(float xfactor,float yfactor){
       
//         set_main_camera();
//         MainCamera.transform.rotation *= Quaternion.AngleAxis(xfactor, Vector3.up);
//         MainCamera.transform.rotation *= Quaternion.AngleAxis(yfactor, Vector3.left);
//     }

//     static void position_camera(GameObject fig, GameObject gr){
        
        
//         // First zoom in while the object is visible.
//         while(IsObjectVisible(fig) && IsObjectVisible(gr)){
//             zoom_in(1);
//             // Debug.Log("Zoom in while");
//         }

//         // Zoom out.
//         while(!IsObjectVisible(fig) || !IsObjectVisible(gr)){
//             zoom_out(1);
//             // Debug.Log("Zoom out while");
//         }

//         // Move couple more times for providence.
//         zoom_out(2f);
//         // extend_fov();
        

//     }
//     static void extend_fov(){
//         Camera camera_component;
//         camera_component = MainCamera.GetComponent<Camera>();
//         camera_component.fieldOfView = 90f;
//     }
    
//     public static string ScreenShotPath(string preposition, string figure, string ground,string number) {
//         string scene_name = EditorSceneManager.GetActiveScene().name;
//         string direct = Application.dataPath + "/Resources/typ_task_folder/" + preposition +"/";
//         string r =  direct + TaskScene.ScreenShotName(scene_name,figure,ground,number);
//         return r;
//     }

//     /// <summary>
//     /// Takes screenshot from camera and saves to file.
//     /// </summary>
//     /// <param name="file">File to save to.</param>
//     public static void take_screenshot(string file){
//         set_main_camera();
//         RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
//         MainCamera.targetTexture = rt;
//         Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
//         MainCamera.Render();
//         RenderTexture.active = rt;
//         screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);
//         MainCamera.targetTexture = null;
//         RenderTexture.active = null; // JC: added to avoid errors
        
//         byte[] bytes = screenShot.EncodeToPNG();
        
//         System.IO.File.WriteAllBytes(file, bytes);
        
        
//     }

//     static void set_main_camera(){
//         // Set camera
//         GameObject[] cam_list = GameObject.FindGameObjectsWithTag(Main.main_camera_tag);
//         string scene_name = EditorSceneManager.GetActiveScene().name;
//         foreach(GameObject c in cam_list){

//             if (c.scene.name == scene_name){
//                 MainCamera = c.GetComponent<Camera>();
//             }
//         }
//     }
//     // Add menu item in editor.
//     [MenuItem ("My Tools/Position Camera for Screenshots")]
   
   
//     /// <summary>

//     /// </summary>
//     /// <remarks>
//     /// </remarks>
//     static void Init ()
//     {   
//         var window = GetWindowWithRect<screenshots_for_typicality>(new Rect(50, 50, 300, 200));
//         window.Show();

        
        
//         // 

//     }

//     void OnGUI()
//     {   

//         EditorGUILayout.BeginVertical();
//         EditorGUIUtility.labelWidth=50f;
//         fig = EditorGUILayout.ObjectField("Figure:",fig, typeof(GameObject), true) as GameObject;
        
//         grd =EditorGUILayout.ObjectField("Ground:",grd, typeof(GameObject), true) as GameObject;
//         EditorGUIUtility.labelWidth=80f;
//         i = EditorGUILayout.IntField("Screenshots#:", i);
//         prep = EditorGUILayout.TextField("Preposition:", prep);
//         EditorGUILayout.EndVertical();

//         if(fig != null){
//             if (fig.tag!=Main.figure_tag){
//             Debug.Log("Check figure tag!");
//             }
//         }
        
//         if(grd != null){
//             if (grd.tag!=Main.ground_tag){
//                 Debug.Log("Check ground tag!");
//             }
//         }

//         if (GUILayout.Button("Search!"))
//         {
//             set_main_camera();

//             TaskScene.highlight_figure(fig);
//             TaskScene.highlight_ground(grd);

//             position_camera(fig,grd);
//         }
//         if (GUILayout.Button("Rotate Left!"))
//         {
//             rotate(-20f,0f);
//         }
//         if (GUILayout.Button("Rotate Right!"))
//         {
//             rotate(20f,0f);
//         }
//         if (GUILayout.Button("Rotate Up!"))
//         {
//             rotate(0f,20f);
//         }
//         if (GUILayout.Button("Rotate Down!"))
//         {
//             rotate(0f,-20f);
//         }
//         if (GUILayout.Button("Take shot!"))
//         {
//             string num = i.ToString();
//             Debug.Log(prep);


//             string newname= ScreenShotPath(prep, fig.name, grd.name,num);
            
            
//             take_screenshot(newname);
//         }
//     }
   
// }


// #endif
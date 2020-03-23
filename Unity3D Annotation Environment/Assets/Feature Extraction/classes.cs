#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;

using System;
using System.Linq;
using System.IO;
using System.Collections;
using System.Collections.Generic;

/// <summary>
/// Class containg scene info.
/// </summary>
public class MyScene{
	public string name;
	public string filename;
	public string path;
	public List<string> mesh_object_list = new List<string>();
	static string[] non_scenes = {"example","finish","instruction","template","main","player","screen","test"};
	public bool study_scene;
	
	static string MainFolder   = Main.MainFolder;
	
	public MyScene(string _filename){
		filename = _filename;
		name = filename.Substring(0,filename.LastIndexOf(".unity"));
		path = MainFolder + "/" + filename;

		if (non_scenes.Any(n => filename.Contains(n))){
			study_scene = false;
		}
		else{
			study_scene = true;
		}
		
	}

	/// <summary>
	/// Open scene and make it active.
	/// </summary>
	public void open_set_active(){
		EditorSceneManager.OpenScene(path);
	    
	    EditorSceneManager.SetActiveScene(EditorSceneManager.GetSceneByName(name));
	}

}

/// <summary>
/// Class to hold information about all scenes.
/// </summary>
public class Scenes{
	public List<MyScene> SceneList =  new List<MyScene> ();
    
    static string MainFolder   = Main.MainFolder;
	static string output_path = Relationship.output_path;
	static string filename = "scene_info.csv";
	string csv_file = output_path + filename;

	string titles_csv_string = "Scene, Mesh Objects";

	int number_of_scenes = 0;

	public static List<string> allObjectnames = new List<string>();
	public static List<string> allObjectcleannames = new List<string>();


    public Scenes(){
    	get_scenes_lists();
    }

    /// <summary>
    /// Get list of study scenes.
    /// </summary>
	void get_scenes_lists (){
        DirectoryInfo d = new DirectoryInfo(@MainFolder);
        FileInfo[] Files = d.GetFiles("*.unity"); //Getting unity files
        
        foreach(FileInfo file in Files )
        {
            MyScene s = new MyScene(file.Name);
            if(s.study_scene){
	            SceneList.Add(s);
	        	number_of_scenes += 1;
            }
    
        }     
    }

    /// <summary>
    /// Save scene info to csv.
    /// </summary>
    public void save_to_csv(){
    	using(StreamWriter file = new StreamWriter(csv_file)){
    		string x = "number_of_scenes: " + number_of_scenes.ToString();
    		file.WriteLine(x);
    		file.WriteLine("allobjectcleannames");

    		file.WriteLine(String.Join(",", allObjectcleannames.ToArray()));
    		file.WriteLine(titles_csv_string);
	    	foreach(MyScene s in SceneList){
				List<string> row = new List<string>();
				row.Add(s.filename);
				foreach(string mo in s.mesh_object_list){
					row.Add(mo);
				}
				
				
				string row_csv_string = String.Join(",", row.ToArray());
				file.WriteLine(row_csv_string);
				}
			}
		}



}

/// <summary>
/// Class for storing information about configuration feature values.
/// </summary>
public class Relationship{
	static string output_path = "Scene Data/";
	static string filename = "relations.csv";
	string csv_file = output_path + filename;
	static string commonsensefilename = "commonsense properties.csv";
	string commonsense_csv_file = output_path + commonsensefilename;
	// Manages relationships between entities and read/writing
	public string scene;
	public string figure;
	public string ground;
	// Need to cast results whenever querying dictionary
	public Dictionary<string,object> relation_dictionary = new  Dictionary<string,object>();
	// Master list of features. Features must be added here or they won't be saved to csv file
	// Add features to end of list
	public List<string> relation_keys = new List<string> {"horizontal_distance_normalized","ground_lightsource","ground_container","support","location_control_x","location_control_z","ground_verticality","figure_volume","ground_volume","size_ratio","shortest_distance","shortest_distance_normalized","contact_proportion","above_proportion","below_proportion","bbox_overlap_proportion","horizontal_projection_overlap_proportion","location_control_-x","location_control_-z","horizontal_distance","f_covers_g","g_covers_f"};
	List<string> titles = new List<string>{"Scene", "Figure", "Ground"};
	string titles_csv_string;

	public Relationship(string scene_, string figure_, string ground_){
		scene = scene_;
		figure = figure_;
		ground = ground_;

		// Loads values of commonsense features
		string line;
		using(StreamReader file =  new StreamReader(commonsense_csv_file)){
            while((line = file.ReadLine()) != null)
           {
            	string[] values = line.Split(',');

            	if(ground.Contains(values[0])){
            		relation_dictionary["ground_container"] = float.Parse(values[1]);
            		relation_dictionary["ground_lightsource"] = float.Parse(values[2]);
            	}
        	}
        }

		titles.AddRange(relation_keys);
		titles_csv_string = String.Join(",", titles.ToArray());

	}

	/// <summary>
	/// Load relationship values from csv.
	/// </summary>
	public void load_from_csv(){
		
		string line;
		using(StreamReader file =  new StreamReader(csv_file)){
            while((line = file.ReadLine()) != null)
               {
                string[] values = line.Split(',');

                if(values[0] == scene && values[1] == figure && values[2] == ground){
                  	foreach(string relation in relation_keys){
                  		string v = values[relation_keys.IndexOf(relation) + 3]; // Add 3 to account for scene,fig,ground
                  		if(v != "?"){
	                  		relation_dictionary[relation] = float.Parse(v);
	                  	}
	                  	else{
	                  		relation_dictionary[relation] = "?";
	                  	}

                  	}
                }
            }
        }
	}

	/// <summary>
	/// Writes relationship to csv file
	/// First creates row to write by getting values associated with relations given in relation_keys
	/// If the configuration doesn't already exist in csv it appends the row
	/// If the configuration does exist it updates values in the csv using new values in the row
	/// Note that if features are removed the row that gets written may have additional unnecessary values
	/// </summary>
	public void save_to_csv(){
		// Writes relation values to csv
		List<string> row = new List<string>();
		row.Add(scene);
		row.Add(figure);
		row.Add(ground);
		// Begin by creating a row which stores the values of each relation
		// If a value is not assigned, "?" is assigned, 
		// which later means those feature values are not changed.
		foreach(string key in relation_keys){
			object val;
			if(relation_dictionary.TryGetValue(key, out val)){
				
				// if the key exists try to cast value as string.
				string v = val as string;

				// If null is returned the object is a float so output that.
				if(v==null){
					string s = string.Format("{0:F9}",val);
					row.Add(s);
					

				}
				// if a non-null value is returned add the string to the row.
				else {
					row.Add("?");
				}

			}
			else{
				// Add default string to row.
				row.Add("?");
				// Add key value pair to dictionary.
				relation_dictionary[key] = "?";
			}
			
		}

		

		List<List<string>> line_list = new List<List<string>>();

		using (StreamReader file = new StreamReader(csv_file)){
			
			string line;
			while((line = file.ReadLine()) != null){
				string[] values = Array.ConvertAll(line.Split(','), p => p.Trim());
				

				line_list.Add(values.ToList());

			}
		}
			

		// Check if there is a match between our row and a row in the csv file
		// If there is then update the values and rewrite csv
		if (line_list.Any(rel => scene == rel[0] && figure == rel[1] && ground ==rel[2])){
			
			using(StreamWriter file = new StreamWriter(csv_file)){
				file.WriteLine(titles_csv_string);
				foreach(List<string> v in line_list){
					if(!v.Contains(titles[0])){
						if(scene == v[0] && figure == v[1] && ground == v[2]){
							
							int x = 0;
							for (x=0; x<relation_keys.Count; x ++){								
								string xth_value = relation_dictionary[relation_keys[x]].ToString();

								if(xth_value != "?"){									
									if(v.Count > x + 3 ){
										v[x+3] = xth_value;
									}
									else{
										v.Add(xth_value);										
									}
								}	
							}
						}

					string row_csv_string = String.Join(",", v.ToArray());
					
					file.WriteLine(row_csv_string);
					}
				}
			}
		}

		// If there is no match, just append line to csv.
		else{
			Debug.Log("No match found");

			using(StreamWriter file = new StreamWriter(csv_file,true)){
				string row_csv_string = String.Join(",", row.ToArray());
				file.WriteLine(row_csv_string);
				Debug.Log(row_csv_string);
			}
		}	
	}
}

/// <summary>
/// Class for storing information about object feature values.
/// Property class is currently obsolete
/// </summary>
public class Property{
	static string output_path = Relationship.output_path;
	static string filename = "properties.csv";
	string csv_file = output_path + filename;
	// Manages relationships between entities and read/writing
	public string scene;
	public string obj;
	
	// Need to cast results whenever querying dictionary
	public Dictionary<string,object> property_dictionary = new  Dictionary<string,object>();
	//
	List<string> property_keys = new List<string> {"bbox_volume","lowest_point"};
	List<string> titles = new List<string>{"Scene", "Object"};
	string titles_csv_string;

	public Property(string scene_, string obj_){
		scene = scene_;
		obj = obj_;
		

		titles.AddRange(property_keys);
		titles_csv_string = String.Join(",", titles.ToArray());

	}
	public bool match(string[] row){
		if (row[0] == scene && row[1] == obj){
			return true;
		}
		else{
			return false;
		}
	}
	public void load_from_csv(){
		string line;
		using(StreamReader file =  new StreamReader(csv_file)){
            while((line = file.ReadLine()) != null)
               {
                  string[] values = line.Split(',');

                  if(match(values)){
                  	foreach(string relation in property_keys){
                  		string v = values[property_keys.IndexOf(relation) + 2]; // Add 2 to account for scene,fig,ground
                  		if(v != "?"){
	                  		property_dictionary[relation] = float.Parse(v);
	                  	}
	                  	else{
	                  		property_dictionary[relation] = "?";
	                  	}

                  	}
                  }
               }
            }
	}


	public void save_to_csv(){
		List<string> row = new List<string>();
		row.Add(scene);
		row.Add(obj);
		

		foreach(string key in property_keys){
			object val;
			if(property_dictionary.TryGetValue(key, out val)){
				
				// if the key exists try to cast value as string

				string v = val as string;

				// If null is returned the object is a float so output that
				if(v==null){
					string s = string.Format("{0:N9}",val);
					row.Add(s);

				}
				// if a non-null value is returned add the string to the row
				else {
					row.Add("?");
					
				}
			}
			else{
				row.Add("?");
			}
		}

		List<List<string>> line_list = new List<List<string>>();
		try
		{
			using (StreamReader file = new StreamReader(csv_file)){
				Debug.Log("Reading");
				string line;
				while((line = file.ReadLine()) != null){
					string[] values = Array.ConvertAll(line.Split(','), p => p.Trim());
					// Debug.Log(values[3]);

					line_list.Add(values.ToList());
				}
			}
			

			if (line_list.Any(rel => match(rel.ToArray()))){
				Debug.Log("Found match");
				using(StreamWriter file = new StreamWriter(csv_file)){
					file.WriteLine(titles_csv_string);
					foreach(List<string> v in line_list){
						if(!v.Contains(titles[0])){
							if(match(v.ToArray())){
								Debug.Log("Found exact match");
								int x = 0;
								for (x=0; x<property_keys.Count; x ++){
									string xth_value = property_dictionary[property_keys[x]].ToString();
									// Debug.Log(property_dictionary["figure_volume"]);
									Debug.Log(property_keys[x]);


									if(xth_value != "?"){
										Debug.Log(xth_value);
										if(v.Count > x + 2 ){
											v[x+2] = xth_value;
										}
										else{
											v.Add(xth_value);
										}
									}
								}
							}
					
					

						
						string row_csv_string = String.Join(",", v.ToArray());
						Debug.Log("Replacing line");
						Debug.Log("Writing:");
						Debug.Log(row_csv_string);
						file.WriteLine(row_csv_string);
						}

					}
				}
			}

			else{
				// Just append line to csv
				using(StreamWriter file = new StreamWriter(csv_file,true)){
					string row_csv_string = String.Join(",", row.ToArray());
					file.WriteLine(row_csv_string);
				}
			}


		}
		catch(Exception ex){
			Debug.Log("Writing to csv failed");
			Debug.Log(ex.ToString());
		}
		finally
		{
			// This statement is executed before function exits.
            // ... It is reached when an exception is thrown.
            // ... It is reached after the return.
            // ... It is reached in other cases.
			
		}

		
	}
}
#endif
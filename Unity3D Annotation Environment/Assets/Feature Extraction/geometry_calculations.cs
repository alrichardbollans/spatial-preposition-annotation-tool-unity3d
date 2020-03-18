// Provides classes which calculate geometric properties between objects and saves to csv

using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class Entity{
	

	//Object name
	public string name;
	public string clean_name;
	// Name of unity scene containing object
	public string scene;
	// Unity GameObject
	public GameObject go;
	// Mesh collider attached to object
	public MeshCollider meshColl =  new MeshCollider();
	
	
	// SphereCollider possibly used for annoying meshes with spherical shape
	public SphereCollider sphColl =  new SphereCollider();
	// Gameobject holding non-convex colliders
	public GameObject non_convex_colliders;
	// The mesh the object is created from (not its instance!)
	public Mesh sharedMesh;
	// The vertices of the shared mesh transformed to world space
	// i.e. vertices of instantiated mesh
	public List<Vector3> transformed_mesh_vertices = new List<Vector3>();
	
	// Whether or not the mesh collider is marked as convex
	public bool convex;
	// If the mesh is too complex to be made convex
	public bool complex;
	// If a sphere collider is used for this object
	public bool sphere;


	// Calculated properties of the entity
	// Bounding box
	public Bounds bbox;
	// Volume of the bounding box
	public float bbox_volume;
	// Horizontal area of the bounding box
	public float bbox_area;
	// Lowest point of object
	public float lowest_point;
	// Highest point of object
	public float highest_point;
	// The centre of the bounding box
	public Vector3 centre_point;
	// Centre of mass of object
	public Vector3 centre_of_mass;
	// If the entity is vertically orientated
	public float verticality;
	
	
	// Note 'y' dimension is up and down
	public Entity(GameObject obj){
		BoxCollider boxColl =  new BoxCollider();
		
		boxColl = obj.AddComponent<BoxCollider>() as BoxCollider;
		bbox = boxColl.bounds;
		// MeshObject class has useful methods
		MeshObject mobj = new MeshObject(obj);
		mobj.prepare_physics_for_game();

		name =  obj.name;
		clean_name = mobj.clean_name;
		go = obj;
		
		scene = obj.scene.name;
		meshColl = obj.GetComponent<MeshCollider>();
		
		sphColl = obj.GetComponent<SphereCollider>();
		
		// Add a mesh collider if there isn't one
		if(meshColl == null){
				meshColl = go.AddComponent<MeshCollider>() as MeshCollider;
			}
		//Get mesh attached to mesh collider
		if(meshColl != null){
			sharedMesh = meshColl.sharedMesh;
			foreach(Vector3 v in sharedMesh.vertices){
				Vector3 x = meshColl.transform.TransformPoint(v);
				transformed_mesh_vertices.Add(x);
			}
		}
		
		
		// Save state of mesh collider
		if(MeshObject.convex_objects.Contains(clean_name)){
            convex = true;
        }
        else{
        	convex = false;
        }
		// Check if mesh is too complex to make mesh collider convex
		if(transformed_mesh_vertices.Count >= 256){
			complex = true;
			
		}
		else{
			complex = false;
		}

		if(MeshObject.sphere_objects.Contains(clean_name)){
            sphere = true;
        }
        else{
        	sphere = false;
        }
		
		
		lowest_point = bbox.min.y;
		highest_point = bbox.max.y;
		centre_point = bbox.center;
		bbox_volume = bbox.size.x * bbox.size.y * bbox.size.z;
		bbox_area = bbox.size.x * bbox.size.z;
		centre_of_mass = get_com();
		verticality = get_verticality();


		mobj.prepare_physics_for_positioning();
		var collidersTransform = go.transform.Find("Colliders");
        GameObject collidersGo;
        if (collidersTransform != null){
            
            non_convex_colliders = collidersTransform.gameObject;
        }
        
		
		}

	
	
	public Vector3 get_com(){
		Rigidbody rb = give_rigid_body();

		Vector3 com = rb.centerOfMass + rb.position; // rb.centreofmass is relative to rb.position

		remove_rigid_body();
		return com;
	}



	public Rigidbody give_rigid_body(){
		// Make sure has rigidbody
        Rigidbody r = go.GetComponent(typeof(Rigidbody)) as Rigidbody;
        if (r == null){
            r = go.AddComponent<Rigidbody>() as Rigidbody;
        }
	        
	    return r;

	}

	public void remove_rigid_body(){
		Rigidbody r = go.GetComponent(typeof(Rigidbody)) as Rigidbody;
		if (r != null){
		    UnityEngine.Object.DestroyImmediate(r);
		}
	}

	public void reset_mesh_collider(){
		if(meshColl != null){
			if(convex == true){
				meshColl.convex =true;
			}
			else{
				meshColl.convex =false;
			}
		}
	}

	public void set_mesh_collider_for_counting(){
		if(!complex){
			meshColl.convex =true;
		}
		
		
	}

	// Checks whether the entity is vertically oriented
	public float get_verticality(){
		if(bbox.size.x < 0.5 * bbox.size.y || bbox.size.z < 0.5 * bbox.size.y){
			return 1f;
		}
		else{
			return 0f;
		}
	}

	
}
public class Pair{
	// Largest distance in scenes (used for normalizing)
	static float largest_distance = 14f;
	//The default contactoffset used by unity
	static float threshold = 0.01f; 
	public Entity e1;
	public Entity e2;
	public string names;

	// Volume of bbox of figure
	public float figure_volume;
	// Volume of bbox of ground
	public float ground_volume;
	// Ratio of size (f/g)
	public float size_ratio;

	//vertices from mesh of e1 which are under threshold distance to the convex hull of e2
	public Vector3[] close_vertices;
	//Vertices of e1 above top of e2
	public Vector3[] above_vertices;
	//Vertices of e1 below bottom of e2
	public Vector3[] below_vertices;
	// Shortest distance from vertices of e1 to convex hull of e2
	public float shortest_distance = -1f;
	public float shortest_distance_normalized;
	// Horizontal distance between centre of masses
	public float horizontal_distance;
	public float horizontal_distance_normalized;
	
	// projection angle used in covering calculation
	public float projection_angle = 5f;
	public float f_covers_g;
	public float g_covers_f;
	
	// Proportion of vertices of e1 which are within a threshold distance of e2
	// This is currently problematic for low density meshes
	public float contact_proportion;
	// Proportion of vertices of e1 above top of e2
	public float above_proportion;
	// Proportion of vertices of e1 below bottom of e2
	public float below_proportion;
	// Volume of overlap of the bounding boxes
	public float bbox_overlap;
	// Proportion of bounding box of e1 overlapping bounding box of e2
	public float bbox_overlap_proportion;
	// Whether or not the ground is vertically orientated
	public float ground_verticality;
	// Absolute vertical distance between bounding boxes
	public float height_separation;
	// Angle from vertical
	public float angle_separation;


	public Pair(Entity ent1, Entity ent2){
		e1 = ent1;
		e2 = ent2;
		names = e1.name + ":" + e2.name;
		figure_volume = e1.bbox_volume;
		ground_volume = e2.bbox_volume;

		// Note we can comment out the heavier functions if they don't need recalculating
		// Consider commenting out mobj.prepare_physics_for_positioning(); in Entity class
		

		
		set_distance_properties();


		contact_proportion = get_proportion_contact();
		above_proportion = get_proportion_above();
		below_proportion = get_proportion_below();

		shortest_distance_normalized = shortest_distance/largest_distance;
		size_ratio = e1.bbox_volume/e2.bbox_volume;
		bbox_overlap = get_bbox_overlap();
		bbox_overlap_proportion = bbox_overlap/figure_volume;
		height_separation = get_height_seperation();
		horizontal_distance = get_horizontal_distance();
		horizontal_distance_normalized = horizontal_distance/largest_distance;
		f_covers_g = get_f_covers_g();
		g_covers_f = get_g_covers_f();
		
		ground_verticality = e2.verticality;
		angle_separation = get_angle_seperation();

		// Write to csv
		Relationship r = new Relationship(e1.scene,e1.name,e2.name);
		r.relation_dictionary["figure_volume"] = figure_volume;
		r.relation_dictionary["ground_volume"] = ground_volume;
		r.relation_dictionary["ground_verticality"] = ground_verticality;

		r.relation_dictionary["size_ratio"] = size_ratio;
		r.relation_dictionary["contact_proportion"] = contact_proportion;

		r.relation_dictionary["above_proportion"] = above_proportion;

		r.relation_dictionary["below_proportion"] = below_proportion;

		r.relation_dictionary["bbox_overlap_proportion"] = bbox_overlap_proportion;
		r.relation_dictionary["horizontal_distance"] = horizontal_distance;
		r.relation_dictionary["horizontal_distance_normalized"] = horizontal_distance_normalized;


		r.relation_dictionary["shortest_distance"] = shortest_distance;
		r.relation_dictionary["f_covers_g"] = f_covers_g;
		r.relation_dictionary["g_covers_f"] = g_covers_f;
		
		r.save_to_csv();
	}
	 
	// Note: e1=e2 does not give all vertices
	public void set_distance_properties(){
		// Runs heavier calculations once to collect info about meshes

		List<Vector3> cvs1 = new List<Vector3>();
		List<Vector3> avs1 = new List<Vector3>();
		List<Vector3> bvs1 = new List<Vector3>();
		
		Collider[] colliders;
		
		
		
		
		foreach(Vector3 v1 in e1.transformed_mesh_vertices){
			
			// Check if v1 is an 'above' or 'below' vertex
			if (v1.y>e2.highest_point){
				avs1.Add(v1);
			}

			if (v1.y<e2.lowest_point){
				bvs1.Add(v1);
			}

			// Find point on e2 which is closest to v1 and calculate distance
			Vector3 v2;
			float d = -1f;

			
			if(!e2.convex && !e2.sphere){
				// In this case e2 uses the non_convex_colliders
				e2.meshColl.convex =false;
				colliders = e2.non_convex_colliders.GetComponents<Collider>();
				foreach(Collider c in colliders){
					v2= c.ClosestPoint(v1);
					
					float d_new;
					d_new = Vector3.Distance(v1,v2);

					if (d == -1f){
						d = d_new;
					}
					else if(d_new <d){
						d = d_new;
					}
				}
			}

			else if(!e2.complex){
				e2.set_mesh_collider_for_counting();
				v2= e2.meshColl.ClosestPoint(v1);
				d = Vector3.Distance(v1,v2);
			}
			else if(e2.sphere){
				v2 = e2.sphColl.ClosestPoint(v1);
				d = Vector3.Distance(v1,v2);

				
			}
			else{
				// BoxCollider used in rare cases with annoying meshes
				BoxCollider boxColl =  new BoxCollider();
				if(boxColl == null){
					boxColl = e2.go.AddComponent<BoxCollider>() as BoxCollider;
				}
				v2= boxColl.ClosestPoint(v1);
				d = Vector3.Distance(v1,v2);
			}
			
			

			// Add to 'contact' vertices if under threshold
			if(d< threshold){
				cvs1.Add(v1);
					
			}
			if(shortest_distance == -1){
				shortest_distance = d;
			}
			if(d<shortest_distance){
				shortest_distance = d;
			}
			
		}

		e2.reset_mesh_collider();

		close_vertices = cvs1.ToArray();
		above_vertices = avs1.ToArray();
		below_vertices = bvs1.ToArray();
	}

	// Gets horizontal distance between centre of mass of e1 and e2
	public float get_horizontal_distance(){
		Vector2 v1;
		Vector2 v2;
		float distance;

		v1 = new Vector2(e1.centre_of_mass.x,e1.centre_of_mass.z);
		v2 = new Vector2(e2.centre_of_mass.x,e2.centre_of_mass.z);

		distance = Vector2.Distance(v1,v2);

		return distance;
	}

	public float get_height_seperation(){
		// Symmetric function
		float height_separation;
		// If there is no overlap in vertical axis return the separation. Else return 0.
		if(e2.bbox.min.y > e1.bbox.max.y){
			height_separation = e2.bbox.min.y - e1.bbox.max.y;
		}
		else if(e1.bbox.min.y > e2.bbox.max.y) {
			height_separation =  e1.bbox.min.y - e2.bbox.max.y;
		}

		
		else{
			height_separation = 0f;
		}

		return height_separation;
	}

	public float get_angle_seperation(){
		float a;
		

		a = Mathf.Atan(horizontal_distance/ height_separation);

		// Normalize by diving by pi/2
		float norm = 2*a/Mathf.PI;
		return norm;
	}

	// Takes vertices in e1 which are 'close' to e2 and divides by the total number of vertices in e1
	public float get_proportion_contact(){
		
		float number_of_close = (float)close_vertices.Length;
		float number_of_e1_vertices = (float)e1.sharedMesh.vertices.Length;
		
		float f = number_of_close / number_of_e1_vertices;
		
		return f;

	}

	public float get_proportion_above(){
		float number_of_above = (float)above_vertices.Length;
		float number_of_e1_vertices = (float)e1.sharedMesh.vertices.Length;
		
		float f = number_of_above / number_of_e1_vertices;
		
		return f;
	}

	public float get_proportion_below(){
		float number_of_below = (float)below_vertices.Length;
		float number_of_e1_vertices = (float)e1.sharedMesh.vertices.Length;
		
		float f = number_of_below / number_of_e1_vertices;
		
		return f;
	}


	public float bound_center_distance(){
		return e1.bbox.SqrDistance(e2.centre_point);
	}

	
	public float get_g_covers_f(){
		return covering(e2,e1);
		
	}

	public float get_f_covers_g(){
		return covering(e1,e2);
		
	}
	public float covering(Entity e1, Entity e2){
		projection_angle = projection_angle * Mathf.Deg2Rad;
		float height_separation;

		float projection_diff;
		float e2_xmin;
		float e2_xmax;
		float e2_zmin;
		float e2_zmax;

		height_separation = get_height_seperation();

		

		
		projection_diff = height_separation*Mathf.Tan(projection_angle);

		e2_xmin = e2.bbox.min.x - projection_diff;
		e2_xmax = e2.bbox.max.x + projection_diff;
		e2_zmin = e2.bbox.min.z - projection_diff;
		e2_zmax = e2.bbox.max.z + projection_diff;
		
		float new_e2_area = (e2_xmax - e2_xmin) * (e2_zmax - e2_zmin);
		
		float overlap_area;


		overlap_area = area_overlap(e1.bbox.min.x,e1.bbox.max.x,e1.bbox.min.z,e1.bbox.max.z,e2_xmin,e2_xmax,e2_zmin,e2_zmax);
		

		float normalized_area =   overlap_area/new_e2_area;
		return normalized_area;

	}
	public float area_overlap(float min1_x, float max1_x, float min1_z, float max1_z, float min2_x, float max2_x, float min2_z, float max2_z){
		float area;
		float x = 0f;
		float z = 0f;
		if( min1_x >= min2_x && max1_x <= max2_x){
			x = max1_x - min1_x;
		}
		else if( min2_x >= min1_x && max2_x <= max1_x){
			x = max2_x - min2_x;
		}
		else if( min1_x <= min2_x && max1_x >= min2_x && max1_x <= max2_x){
			x = max1_x - min2_x;
		}
		else if( min1_x >= min2_x && min1_x <= max2_x && max1_x >= max2_x){
			x = max2_x - min1_x;
			}

		if( min1_z >= min2_z && max1_z <= max2_z){
			z = max1_z - min1_z;
		}
		else if( min2_z >= min1_z && max2_z <= max1_z){
			z = max2_z - min2_z;
		}
		else if( min1_z <= min2_z && max1_z >= min2_z && max1_z <= max2_z){
			z = max1_z - min2_z;
		}
		else if( min1_z >= min2_z && min1_z <= max2_z && max1_z >= max2_z){
			z = max2_z - min1_z;
		}

		area = x*z;
		return area;
	}
	// Gets the volume of bounding box overlap
	public float get_bbox_overlap(){
		float x = 0f;
		float y = 0f;
		float z = 0f;
		float vol;

		if( e1.bbox.min.x >= e2.bbox.min.x && e1.bbox.max.x <= e2.bbox.max.x){
			x = e1.bbox.max.x - e1.bbox.min.x;
		}
		else if( e2.bbox.min.x >= e1.bbox.min.x && e2.bbox.max.x <= e1.bbox.max.x){
			x = e2.bbox.max.x - e2.bbox.min.x;
		}
		else if( e1.bbox.min.x <= e2.bbox.min.x && e1.bbox.max.x >= e2.bbox.min.x && e1.bbox.max.x <= e2.bbox.max.x){
			x = e1.bbox.max.x - e2.bbox.min.x;
		}
		else if( e1.bbox.min.x >= e2.bbox.min.x && e1.bbox.min.x <= e2.bbox.max.x && e1.bbox.max.x >= e2.bbox.max.x){
			x = e2.bbox.max.x - e1.bbox.min.x;
			}

		if( e1.bbox.min.y >= e2.bbox.min.y && e1.bbox.max.y <= e2.bbox.max.y){
			y = e1.bbox.max.y - e1.bbox.min.y;
		}
		else if( e2.bbox.min.y >= e1.bbox.min.y && e2.bbox.max.y <= e1.bbox.max.y){
			y = e2.bbox.max.y - e2.bbox.min.y;
		}
		else if( e1.bbox.min.y <= e2.bbox.min.y && e1.bbox.max.y >= e2.bbox.min.y && e1.bbox.max.y <= e2.bbox.max.y){
			y = e1.bbox.max.y - e2.bbox.min.y;
		}
		else if( e1.bbox.min.y >= e2.bbox.min.y && e1.bbox.min.y <= e2.bbox.max.y && e1.bbox.max.y >= e2.bbox.max.y){
			y = e2.bbox.max.y - e1.bbox.min.y;
		}

		if( e1.bbox.min.z >= e2.bbox.min.z && e1.bbox.max.z <= e2.bbox.max.z){
			z = e1.bbox.max.z - e1.bbox.min.z;
		}
		else if( e2.bbox.min.z >= e1.bbox.min.z && e2.bbox.max.z <= e1.bbox.max.z){
			z = e2.bbox.max.z - e2.bbox.min.z;
		}
		else if( e1.bbox.min.z <= e2.bbox.min.z && e1.bbox.max.z >= e2.bbox.min.z && e1.bbox.max.z <= e2.bbox.max.z){
			z = e1.bbox.max.z - e2.bbox.min.z;
		}
		else if( e1.bbox.min.z >= e2.bbox.min.z && e1.bbox.min.z <= e2.bbox.max.z && e1.bbox.max.z >= e2.bbox.max.z){
			z = e2.bbox.max.z - e1.bbox.min.z;
		}

		vol = x * y * z  ;  
		return vol;
	}
}

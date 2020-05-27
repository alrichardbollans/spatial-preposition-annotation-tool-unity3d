# Feature Calculations

Code for feature extraction appears in Feature Extraction folder of Unity3D Assets.

For geometric features geometric_feature_extraction.cs provides a button in Unity editor and handles iterating over all scenes. For functional features the corresponding script is functional_feature_extraction.cs. write_scene_info.cs also provides a button 'Write Scene Info' in the editor --- this should be ran before any feature extraction.







## Geometric Features
Each feature is calculated between a pair of objects `e1`,`e2`. Features are **not** necessarily **symmetric**.

Functions for calculations are given in geometry_calculations.cs

### Mesh Approximations
For some geometric features meshes must be approximated using "Colliders" for computational reasons. This occurs in `set_distance_properties()`. If the mesh is can be appropriately approximated by it's convex hull, a box collider or a sphere collider this is used. Else, the mesh is approximated by a collection of box colliders using the NonConvexMesh Collider script.

### Shortest Distance
Shortest distance between `e1` and `e2` is calculated by iterating over the vertices of `e1` to find which vertex is closest to an approximation of `e2`. The shortest distance is then just the distance from this vertex to the approximation of `e2`.

### Containment
`containment` (referred to as `bbox_overlap_proportion`) is calculated by first calculating the volume of overlap of the axis-aligned bounding boxes of `e1` and `e2`. This volume is then divided by the volume of the axis-aligned bounding box of `e1`.

### F_covers_g

`f_covers_g` relies on a few variables. 

Firstly `height_seperation`: if the bottom of `e2` is above the top of `e1` then the height seperation is `height_separation = bottom of e2 - top of e1`. Else, if the bottom of `e1` is above the top of `e2` then the height seperation is `height_separation =  bottom of e1 - top of e2`. Else the height seperation is 0.

Next `projection_angle`: this is set to 5 degrees and accounts for the idea that as a covering object moves further from the object it is covering, it must be larger to provide the same covering function. Imagine an umbrella moving further and further away above somebody.

The projection angle and height seperation are then used to create the horizontal area that `e1` is aiming to cover, `new_e2_area` --- the area of `e2` plus the extra. The area of overlap of this area with the horizontal area of `e1` is then calculated. This overlap area is then divided by the total horizontal area of `new_e2_area`.

### G_covers_f
`g_covers_f` is the same as `f_covers_g` but with `e1` and `e2` swapped.
### Horiztontal Distance
`horizontal_distance` is just the horizontal distance between the centre of mass of `e1` and the centre of mass of `e2`.

### Contact
To calculate `contact` (referred to as `contact_proportion`), the number of vertices of `e1` which are under some threshold distance to the approximation of `e2` are counted. This is then divided by the total number of vertices in the mesh of `e1`.

### Above_proportion
To calculate `above_proportion`, the number of vertices of `e1` which are above the highest point of `e2` are counted. This is then divided by the total number of vertices in the mesh of `e1`.
### Below_proportion
To calculate `below_proportion`, the number of vertices of `e1` which are below the highest point of `e2` are counted. This is then divided by the total number of vertices in the mesh of `e1`.

## Functional Features

As simulation is used to calculate these features, the functions for calculations are given in an in-script WaitingGame_FunctionalFeatureExtraction.cs

In order to the physics of falling/moving objects to work appropriately some objects need assigning an NonConvexMeshCollider approximation. All physics is handled in set_objects_at_rest.cs script.

### Nonconvex collider approximation
Necessary assets are provided in Unity3D Annotation Environment folder, except NonConvexMesh Collider asset which must be [purchased](https://assetstore.unity.com/packages/tools/physics/non-convex-mesh-collider-84867). This is used for better object physics.

### Support
These calculations are executed in `print_support_positions()`.

Firstly the script gets the initial height of the centre of mass of the figure. Then the ground object is removed. Then the game waits 10 seconds and gets the height of the figure centre of mass. The raw `movement` is the initial position minus the new position.

This value is then normalized by an appropriate `ground_height`. If the bottom of the figure is above the top of the ground (within a threshold), then `ground_height` is just the height of the ground object. Else, if the bottom of the figure is above the bottom of the ground(within a threshold), then `ground_height` is `fig_bottom - ground_bottom_point`. Else, if the initial centre of mass is above the bottom of the ground(within a threshold), then `ground_height` is `fig_com_y_initial - ground_bottom_point;`. In all other cases `ground_height` is just the height of the ground object.

The `normalized_movement` is the raw `movement` divided by the `ground_height`. This is capped between 0 and 1.



### Location Control
These calculations are executed in `print_lc_positions()`.

Four seperate calculations are performed for each cardinal direction. These values are then averaged in later processing.

In one cardinal direction a force is applied to the ground object. The horizontal movement in the direction of the force of the centre of mass of the figure is measured. This is then divided by the movement (in direction of force) of the centre of mass of the ground. This value is then capped between 0 and 1.

# Annotator Agreement
Annotator agreements are calculated by process_data.py in Analysis and saved in the stats folder.

In each task agreement is calculated for each pair of Native English speaking users.

Cohens kappa is calculated for each pair of users and multiplied by the shared number of annotations. This is then summed for each user pair and divided by the total number of shared annotations. So the average Cohens Kappa given is weighted by shared annotations.

In both tasks observed agreement is the number of agreements divided by the number of shared annotations.

## Preposition Selection Task
A shared annotation in this task is where the users are shown the same figure and ground in the same scene.

As usual with Cohens Kappa expected agreement between users u1,u2 (for a given preposition) is the number of times in shared annotations that u1 says yes times the number of times that u2 says yes, plus number of times u1 says no times u2 says no divided by the number of shared annotations squared: `(y1*y2 + n1*n2))/(shared_annotations)^2`



## Comparative Task
A shared annotation in this task is where the users are shown the same ground and preposition in the same scene.

As we don't have category labels here we approximate expected agreement for a pair of users slightly differently.

Firstly we calculated the probability that u1 and u2 agree by both selecting none. This is the probability u1 selects none times probability that u2 selects none:
`u1_p_none = comp_none_selections1/shared_annotations`
`u2_p_none = comp_none_selections2/shared_annotations`
			

`expected_none_agreement = float(u1_p_none * u2_p_none)`

Then to work out the probability users agree on a particular object we calculate:

`average_probability_agree_on_object = (shared_annotations * (1-u1_p_none) * (1-u2_p_none))/number_of_compared_figures`

where number_of_compared_figures is the sum of all compared figures (potential objects to select) in all shared annotations.


Expected agreement is then simply:
`expected_agreement = expected_none_agreement + average_probability_agree_on_object`


See Agreements class for code.


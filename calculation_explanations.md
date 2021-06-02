# Feature Calculations

Code for feature extraction appears in Feature Extraction folder of Unity3D Assets.

For geometric features geometric_feature_extraction.cs provides a button in Unity editor and handles iterating over all scenes. For functional features the corresponding script is functional_feature_extraction.cs. write_scene_info.cs also provides a button 'Write Scene Info' in the editor --- this should be ran before any feature extraction.

Diagrams and descriptions of the features are provided in `feature extraction.pdf`.





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



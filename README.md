## Project: Perception Pick & Place

Attached is my perception pipeline script subscribing to `/pr2/world/points`, output_*.yamls, train_svm.py, features.py, and capture_features.py.  For training the models I changed the model list in capture_features.py and doubled the number of angles being trained per model to 10.  The kernel was also changed to RBF which gave more accuracy on the test set predictions. Next I will be purusing the extra challenges but wanted to get this submission in first for peace of mind!

Pipeline in depth:

1.) I decreased the leaf_size to 0.005 for a more granular point cloud.

2.) Two pass through filters were used.  One is about the same as the exercise's along the Z axis to filter out unused data below and above a certain threshold.  The 2nd I found useful to get rid of some of the clutter beyond the table.

3.) K-Means outlier filter was run.  I experimented with a few numbers and found that a cluster of 50 with a distance of 0.04 greater than the Std of distances was adequate in filtering out the nice little sparkles around the table.

4.) A RANSAC filter was run to remove the plane from the point cloud (to eventually isolate the objects!).  A distance threshold of 0.02 was adequate in remove its contents.  I found the leaf size being smaller especially helpful here (allowed me to cut the margins on how close the points could be).

5.) Color was thrown out of the filtered point cloud we are now left with leaving only with position.  A KD-tree was constructed for quicker search and clustering.  The values used were the same as in exercise 3.

6.) Each cluster was given a random color to distinguish when published to the appropriate ROS subscriber to view in RViz.

7.) All appropriate clouds were published (the colored objects, the filtered out plane,etc..)

8.) Classification and labeling took place using the trained classifier we made previously with capture_features.py and train_svm.py.  This was done specifically by computing the histogram data (color and normals) for each object remaining and running it through our trained ML model.

9.) Object centroids were calculated

10.) All relevant data was sent to the pick and place helper function to allow the PR2 to pick up the object and place it in one of the two available boxes!!!


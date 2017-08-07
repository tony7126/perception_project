## Project: Perception Pick & Place

Attached is my perception pipeline script subscribing to `/pr2/world/points`, output_*.yamls, train_svm.py, features.py, and capture_features.py.  For training the models I changed the model list in capture_features.py and doubled the number of angles being trained per model to 10.  The kernel was also changed to RBF which gave more accuracy on the test set predictions. Next I will be purusing the extra challenges but wanted to get this submission in first for peace of mind!

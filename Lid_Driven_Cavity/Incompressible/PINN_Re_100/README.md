

<h1>Modifying the Model</h1>
The NN model is created in Lid_Driven_Cavity.py.
Modifications may be done in this file.
Files named mod_*.py are imported in Lid_Driven_Cavity.py. To keep the main code cleaner and compact, various classes, such as for domain and loss, have been defined in the mod_*.py files.
Before trining the model, make sure the variable result_dir is assigned to a directory path (result directory) where the model will be saved. 

<h1>Result</h1>
Post-processing may be done in a separate folder such as in Result_01/.
Copy the NN model class to the file model_class_01.py or in the file post_processing.py before the model is loaded. The model class must be available where the model is loaded back. Here, post+processing.py file imports model_class.py before loading the model.
Model is loaded in the file post_processing.py and data is written in the output files.
The output files are loaded by the MATLAB file plot_result.m for generating plots.

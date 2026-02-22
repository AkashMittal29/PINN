"""
Feb 14, 2025
Akash Kumar Mittal

Tensorflow Essentials 04
Tensorflow Variables in Graph Execution
"""

#%% To remove all imported libraries and delete variables
from IPython import get_ipython
get_ipython().magic('reset -sf');


#%% Importing Libraries
import tensorflow as tf


#%% Executing in graph mode
tf.compat.v1.disable_eager_execution(); 
sess = tf.compat.v1.Session(); # Creating session for graph execution


#%% Defining Tensorflow Variables
# var = tf.Variable(tf.random.normal(shape=(1, 3), mean=0.0, stddev=1.0)) # random normal initializer
initializer = tf.keras.initializers.GlorotUniform()  # Xavier/Glorot initializer (Tensorflow Built-in Initializer)
var = tf.Variable(initializer(shape=(1, 3)), 
                  dtype=tf.float32, name = 'my_var', trainable=True);

res = tf.pow(var,2);


#%% Running the session
init=tf.compat.v1.global_variables_initializer(); # This will declare all the variables we have defined
sess.run(init)     # This is necessary to initialize the variables, either with the given values defined in the variable definition.
d=(sess.run(res)); # Returns a numpy array



















#%% To remove all imported libraries and delete variables
from IPython import get_ipython
get_ipython().magic('reset -sf');


#%% Importing libraries
import tensorflow as tf
import numpy as np


#%% Executing in graph mode
tf.compat.v1.disable_eager_execution(); 


#%% Performing some operations
a = tf.constant([1,2,3], dtype=tf.float64, shape=(1,3), name='ana');
b = tf.constant(np.array([[2,3,4],[5,6,7]]), dtype=tf.int64, name='b0');
b = tf.cast(b, dtype=tf.float64, name='b1') # dtype conversion/casting

a=a+1; # This does not change the constant value, only the varaible 'a' now refers to other object (to the operation add)
c = tf.add( tf.pow(a,2,name='power'), b, name='addition' ); # broadcasting

with tf.compat.v1.Session() as sess:
    writer=tf.compat.v1.summary.FileWriter('./graph',sess.graph); # run this from the command line after execution: tensorboard --logdir=my_log_dir
    d=(sess.run(c)); # Returns a numpy array

print(d, '\n\n', d.shape, '\n\n', d.size)


#%% To get tensor by its name
print(b.name)
graph = tf.compat.v1.get_default_graph()
tensor = graph.get_tensor_by_name("b1:0")
graph.get_operations()


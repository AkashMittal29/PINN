"""
Feb 14, 2025
Akash Kumar Mittal

Tensorflow Essentials 05
Placeholder
"""

#%% To remove all imported libraries and delete variables
from IPython import get_ipython
get_ipython().magic('reset -sf');


#%% Importing Libraries
import tensorflow as tf
import numpy as np


#%% Executing in graph mode
tf.compat.v1.disable_eager_execution(); 
sess = tf.compat.v1.Session(); # Creating session for graph execution
# placeholder is NOT compatible with eager execution.


#%% Defining Tensorflow Placeholder
x = tf.compat.v1.placeholder(dtype=tf.float16,shape=(None,1),name='x'); # NOTE: placeholder is not compatible with eager execution.


#%% Operations using placeholder
y = tf.pow(x,3, name='cube');
z = tf.add(x, y, name='add');


#%% Running the session
a = np.linspace(1,10,10);
a = np.reshape(a,(a.size,1)); # converting to a shape compatible with the placeholder

res_y = sess.run(y,feed_dict={x:a}); # Returns a numpy array
res_z = sess.run(z,feed_dict={x:a}); # Returns a numpy array


















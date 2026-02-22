"""
Feb 14, 2025
Akash Kumar Mittal

Tensorflow Essentials 03
Tensorflow Variables in Eager Execution
"""

#%% To remove all imported libraries and delete variables
from IPython import get_ipython
get_ipython().magic('reset -sf');


#%% Importing Libraries
import tensorflow as tf


#%% Executing in graph mode
#tf.compat.v1.disable_eager_execution(); 


#%% Defining Tensorflow Variables
a=tf.constant([1,2,3]); # Constant
var=tf.Variable(a);     # Variable
print(tf.shape(var), tf.rank(var)); 
print(var.dtype);
print(var.numpy());


#%% Reassigning a variable
var=tf.Variable([1,2,3]);
var.assign([4,5,6]);
print(var); # shape of the new tensor is same as the shape of var before assignment.

# Assigning an existing variable with the tensor of different shape
try: 
    var.assign([1,2]); # shape is different than the present shape of var: VALUEERROR
except Exception as e:
    print(f'{type(e).__name__}: {e}');


#%% PROPERTIES OF VARIABLE
a=tf.Variable([1,2],name='my_var',trainable=False, dtype=tf.int32); 
# name is automatically assigned if not provided.
# trainable false: gradient of such a variable is not computed.
# By default all variables are trainable.

# All tensorflow functions for constants are valid for Variables as well.


#%%


















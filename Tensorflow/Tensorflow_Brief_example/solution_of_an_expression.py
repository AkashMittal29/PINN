

#%% To remove all imported libraries and delete variables
from IPython import get_ipython
get_ipython().magic('reset -sf');


#%% Importing Libraries
import tensorflow as tf
import numpy as np


#%% Executing in graph mode
tf.compat.v1.disable_eager_execution(); 
sess = tf.compat.v1.Session(); # Creating session for graph execution


#%% Defining a variable and placeholder
# NOTE: placeholder is not compatible with eager execution.
x = tf.compat.v1.placeholder(dtype=tf.float16,shape=(None,1),name='placeholder_x'); # shape: (None, 1)
y = tf.Variable([3],name='var_y',trainable=False, dtype=tf.float16); # shape: (1,): 1 dimension


#%% Defining an expression and taping for gradient evaluation
# Only TRAINABLE Variables are available for automatic watch in GradientTape
with tf.GradientTape(persistent=True) as tape_1:
    tape_1.watch([x,y]);
    z = x**3+y**2
    
    
#%% Evaluating gradient
dz_dy = tape_1.gradient(z,y);
dz_dx = tape_1.gradient(z,x);


#%% Running the session
init=tf.compat.v1.global_variables_initializer();
sess.run(init); # required for variables

a = np.linspace(1,3,3); # shape: (5,): 1 dimension
a = np.reshape(a,(a.size,1)); # converting to a shape (5,1) compatible with the placeholder

print(tape_1.watched_variables())

grad_z_y = sess.run(dz_dy,feed_dict={x:a});
print(grad_z_y)
grad_z_x = sess.run(dz_dx,feed_dict={x:a});
print(grad_z_x)

#writer=tf.compat.v1.summary.FileWriter('./graph_1',sess.graph); # run this from the command line after execution: tensorboard --logdir=my_log_dir


# EXTRA
#%% To get tensor by its name
print(x.name)
graph = tf.compat.v1.get_default_graph()
tensor = graph.get_tensor_by_name("placeholder_x:0")
graph.get_operations()









 

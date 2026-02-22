"""
Feb 14, 2025
Akash Kumar Mittal

Tensorflow Essentials 06
Automatic Differentiation (Algorithmic Differentation)
"""

#%% To remove all imported libraries and delete variables
from IPython import get_ipython
get_ipython().magic('reset -sf');


#%% Importing Libraries
import tensorflow as tf


#%% Executing in graph mode
tf.compat.v1.disable_eager_execution(); 
sess = tf.compat.v1.Session(); # Creating session for graph execution


#%% Defining Tensorflow constants/Variables/placeholders
x=tf.constant([1.0,2.0,3.0]);


#%% Taping (recording) the operations for gradient computation
# Note: tf.GradientTape works with both eager execution enabled or disabled.
# Prefer using tf.GradientTape. (tf.gradients() is also available which is only available in graph mode)
# With persistent=True, the gradient over a variable can be computed multiple times. 
# It also facilitates to compute higher order derivatives within the GradientTape.
with tf.GradientTape(persistent=True) as tape1:
    tape1.watch([x]); # if this is not included then gradient 
    # with respect to x is not available since x is constant.
    # without watch, gradient is possible only with respect to variables.
    y=x*x;
    
with tf.GradientTape(persistent=True) as tape2:
    tape2.watch([x]);  
    dy_dx=tape1.gradient(y,x);
    d2y_dx2=tape2.gradient(dy_dx,x); # With persistent=True it is possible to make higher order derivative operations


#%% Operations to obtain gradient based on taping
dy_dx_2 = tape1.gradient(y,x);
                          

#%% Running the session
tape1.watched_variables() # Is only for variables
d0, d1, d2 = sess.run([y, dy_dx_2, d2y_dx2])


















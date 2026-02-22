"""
Feb 14, 2025
Akash Kumar Mittal

Tensorflow Essentials 07
Automatic Differentiation Examples
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


#%% Example 1/4: Non-scalar target
# Defining Tensorflow Variable
x=tf.Variable([2.0], name='ex_1_x');

# Taping (recording) the operations for gradient computation
with tf.GradientTape() as tape1:
    y0=x**2;
    y1=x**3;
    
out1 = tape1.gradient({'y0':y0,'y1':y1},x); # Non-scalar target
# Here target is a dictionary of two targets y0 and y1.
# The gradient has the shape of the source (here, x).
# The gradient here is therefore the sum of the gradients of each target.

# Running the session
init=tf.compat.v1.global_variables_initializer();
sess.run(init);
d1 = sess.run(out1);
print(d1[0])
print(tape1.watched_variables())


#%% Example 2/4: Non-scalar target
# Defining Tensorflow constants/Variables/placeholders
x=tf.Variable([2.0], name='ex_2_x');

# Taping (recording) the operations for gradient computation
with tf.GradientTape() as tape2:
    y=x**3*[1.0, 2.0]; # Non-scalar target

out2 = tape2.gradient(y,x);
# x=2.0
# dy_dx=[(3x^2)*1.0,(3x^2)*2.0] = [12.0, 24.0]
# output is 2x+4x = 36.0

# # Running the session
init=tf.compat.v1.global_variables_initializer();
sess.run(init);
d2 = sess.run(out2);
print(d2[0])
print(tape2.watched_variables())


#%% Example 3/4: sigmoid
# Defining Tensorflow constants
x=tf.linspace(-10.0,10.0,200+1, name='ex_3_x'); # shape (201,) # is a constant

# Taping (recording) the operations for gradient computation
with tf.GradientTape() as tape3:
    tape3.watch(x);
    y=tf.nn.sigmoid(x); # shape (201,)
    
dy_dx=tape3.gradient(y,x); # shape (201,)

# Running the session
d3_0, d3_1 = sess.run([y, dy_dx]); # Returns numpy arrays
print(tape3.watched_variables())   # No variable included

import matplotlib.pyplot as plt
plt.figure();
plt.plot(np.linspace(-10.0,10.0,200+1), d3_0, label='y');
plt.plot(np.linspace(-10.0,10.0,200+1), d3_1, label='dy_dx')
plt.legend();
plt.xlabel('x');


#%% Example 4/4: 
# Defining Tensorflow constants
x=tf.constant([1.,2.,3.,4.]); # shape (4,)

# Taping (recording) the operations for gradient computation
with tf.GradientTape() as tape:
    tape.watch(x);
    xa=x[1:];
    xa=tf.concat([xa,[x[0]]],axis=0);
    y=x**2+xa;
    
dy_dx=tape.gradient(y,x);























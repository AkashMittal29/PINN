"""
Feb 14, 2025
Akash Kumar Mittal

Tensorflow Essentials 01
Constants and Few Operations
"""

#%% To remove all imported libraries and delete variables
from IPython import get_ipython
get_ipython().magic('reset -sf');


#%% Importing libraries
import tensorflow as tf


#%% Executing in graph mode
#tf.compat.v1.disable_eager_execution(); 


#%% Declaring Tensorflow Constants
rank_0_tensor = tf.constant(105, tf.int32); 
print(rank_0_tensor)

rank_1_tensor=tf.constant([1.,2.,3.]); 
print(rank_1_tensor);

rank_2_tensor=tf.constant([[1.0, 2., 5.7],
                           [4.,  5., 8.9]], dtype=tf.float16, shape=(2,3), name='aaa'); 
print(rank_2_tensor);

rank_3_tensor = tf.constant([
                [[0, 1, 2, 3, 4],
                 [5, 6, 7, 8, 9]],
                [[10, 11, 12, 13, 14],
                 [15, 16, 17, 18, 19]],
                [[20, 21, 22, 23, 24],
                 [25, 26, 27, 28, 29]],
                            ]); 
print(rank_3_tensor);


#%% Tensor properties
rank_4_tensor=tf.zeros(shape=(3,2,4,5),dtype=tf.float16);
print('element type: ',rank_4_tensor.dtype);
print('no. of axis: ',rank_4_tensor.ndim);          # Does not return tensor
print('shape of tensor: ',rank_4_tensor.shape);     # Does not return tensor
print('total no. of elements: ',tf.size(rank_4_tensor).numpy()); # Will not work in graph execution
rank_4_tensor.dtype==tf.float16 # True

# rank_4_tensor.ndim and .shape do not return tensor. Hence, use following instead:
print('no. of axis: ',tf.rank(rank_4_tensor));      # Returns a tensor
print('shape of tensor: ',tf.shape(rank_4_tensor)); # Returns a tensor
print(rank_4_tensor.shape.as_list());

tf.shape(rank_4_tensor)==[3, 2, 4, 5] # Returns a tensor of dtype bool


#%% INDEXING IN TENSORS
a=tf.constant([4,5,6,2,49,34,45,64,10,54]);
print('first',a[0].numpy());
print('last',a[-1].numpy());
a[-1]==54                #returns a tensor, of dtype bool
print(a[:3].numpy());    # 0 to 2 elements (excluding 3)
print(a[0:5:2].numpy()); # 0 to 5(excluding) with step 2: every 2nd element after an element

a = tf.constant([[4.0, 5.0], [10.0, 1.0]]); # 2nd rank tensor
print(a[:,0].numpy());
b=a[:,0];                # shape:(2,) : rank=1 tensor
b=tf.reshape(b,shape=(2,1)); # Reshaping the tensor
tf.rank(b).numpy()

c = a[:,0:1];            # rank is retained (useful)
tf.rank(c)
print(c.shape)


#%% MANIPULATING SHAPES
# Reshaping tensor is of great utility
# a=tf.reshape(b,shape=(1,3)); # reshape() creates a new tensor in memory
a=tf.constant(
[[[ 0,  1,  2,  3,  4,],
  [ 5,  6,  7,  8,  9,]],

 [[10, 11, 12, 13, 14,],
  [15, 16, 17, 18, 19]],

 [[20, 21, 22, 23, 24,],
  [25, 26, 27, 28, 29]]] );

print(a)
print(tf.reshape(a,(5,6))); # lowest axis is filled (here 6 elements) from the source sweeping from the lowest axis. 
print(tf.reshape(a,(3,-1))); # -1: whatever fits
print(tf.reshape(a,(-1,))); # Flattening tensor into vector (single rank tensor)


#%% DATA TYPE CASTING
a=tf.constant([1,3,5,4]); print(a.dtype);
a=tf.cast(a,dtype=tf.float16); print(a.dtype);
a=a*-1;
a=tf.cast(a,dtype=tf.uint8); print(a.dtype); # decimal places are lost
# and -ve values will become zero


#%% BROADCASTING
x=tf.constant([1,2,3]);
y=tf.constant(2);
z=tf.constant([2,2,2]);

print(tf.multiply(x,y)); # Elementwise multiplication with broadcasting
print(x*y); # Elementwise multiplication with broadcasting of y
print(x*z); # Elementwise multiplication

x=tf.reshape(x,shape=(3,1));
y=tf.reshape(x,shape=(1,3)); y=tf.multiply(y,10);
print(x+y); 
print(tf.add(x,y)); # will give a 3 by 3 matrix after broadcasting

print(tf.broadcast_to(tf.constant([1,2,3]),shape=(3,3)));
# This function does nothing special to save memory.


#%%












## Lid Driven Cavity 
## Incompressible Navier-Stokes Sloution using Neural Network

#%% To remove all imported libraries and delete variables
from IPython import get_ipython
get_ipython().magic('reset -sf');


#%% Importing Libraries
import tensorflow as tf
import numpy as np
import keras
from keras import layers, models, Input
from keras.models import Model
from keras import backend
import random
import pickle


#%% Executing in graph mode
tf.compat.v1.disable_eager_execution(); 
sess = tf.compat.v1.Session(); # Creating session for graph execution
backend.set_session(sess); # Required to tell Keras to use this session. 


#%% Setting precision for Tensorflow (used as backend by Keras)
# Default: float32
tf.keras.backend.set_floatx('float64') 


#%% CONSTANTS
global Re_ref;
Re_ref=40;


#%% Base model definition to approximate flow field (Instantiating the model)
input_tensor=Input(shape=(2,)); # input: x,y (None, 2)
out1=layers.Dense(60,activation='tanh')(input_tensor);
out1=layers.Dense(60,activation='tanh')(out1);
out1=layers.Dense(60,activation='tanh')(out1);
out2=layers.Dense(3,activation='linear')(out1); # output: u,v,p

model_field=Model(inputs=[input_tensor], outputs=[out2]);
model_field.summary();
# We can not directly tape these commands because input_tensor, out1 are Kerastensor (objects of Keras) not tensorflow tensors.


#%% Model to compute gradient using base model
# Defining custom Keras layer
class cust_layer_grad(keras.layers.Layer):
    def __init__(self, units=1, **kwargs):
        super().__init__(**kwargs);
        self.units=units;
        
    def get_config(self):
        config=super().get_config();
        config ["units"]=self.units;
        return config;
    
    def call(self, xy, model_passed): 
        # xy = [x, y]: (None,2)
        # model_passed: input (None,2)->[x,y], output (None,3)->[u,v,p]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([xy]);
            uvp = model_passed(xy);
            u  = uvp[:,0:1]; # 0:1 -> to preserve dimension (end value is excluded in Python)
            v  = uvp[:,1:2];
            p  = uvp[:,2:3];
            u2 = u*u
            v2 = v*v
            uv = u*v
            du_dxy  = tape.gradient(u, xy)
            dv_dxy  = tape.gradient(v, xy)
            du2_dxy = tape.gradient(u2,xy)
            dv2_dxy = tape.gradient(v2,xy)
            duv_dxy = tape.gradient(uv,xy)
            dp_dxy  = tape.gradient(p, xy)
            du_dx   = du_dxy[:,0:1]
            du_dy   = du_dxy[:,1:2]
            dv_dx   = dv_dxy[:,0:1]
            dv_dy   = dv_dxy[:,1:2]
            du2_dx  = du2_dxy[:,0:1]
            dv2_dy  = dv2_dxy[:,1:2]
            duv_dx  = duv_dxy[:,0:1]
            duv_dy  = duv_dxy[:,1:2]
            dp_dx   = dp_dxy[:,0:1]
            dp_dy   = dp_dxy[:,1:2]
            d2u_dxdxy = tape.gradient(du_dx,xy)
            d2u_dydxy = tape.gradient(du_dy,xy)
            d2v_dxdxy = tape.gradient(dv_dx,xy)
            d2v_dydxy = tape.gradient(dv_dy,xy)
            d2u_dx2 = d2u_dxdxy[:,0:1]
            d2u_dy2 = d2u_dydxy[:,1:2]
            d2v_dx2 = d2v_dxdxy[:,0:1]
            d2v_dy2 = d2v_dydxy[:,1:2]
        
        result=tf.concat([u, v, p, 
                          du_dx, dv_dy,
                          du2_dx, duv_dy, dp_dx, d2u_dx2, d2u_dy2,
                          duv_dx, dv2_dy, dp_dy, d2v_dx2, d2v_dy2], 
                         axis=1);
        return result;

out3=cust_layer_grad(1)(input_tensor, model_field);   
    
model=Model(inputs=[input_tensor],outputs=[out3]);   
model.summary();


#%% 



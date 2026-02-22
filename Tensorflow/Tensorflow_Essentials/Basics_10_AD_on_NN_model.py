"""
Feb 14, 2025
Akash Kumar Mittal

Automatic Differentiation and Keras Model 09
"""
#%% To remove all imported libraries and delete variables
from IPython import get_ipython
get_ipython().magic('reset -sf');


#%% Importing Libraries
import numpy
import keras
from keras import layers, Input
from keras.models import Model
import tensorflow as tf
from keras import backend
#import pickle


#%% Executing in graph mode
tf.compat.v1.disable_eager_execution();
sess=tf.compat.v1.Session();
backend.set_session(sess); # Required to tell Keras to use this session.


#%% Setting precision for Tensorflow (used as backend by Keras)
# Default: float32
tf.keras.backend.set_floatx('float64') 


#%% Creating a Keras model
input_tensor=Input(shape=(2,)); # input: x, (None, 2)
out1=layers.Dense(2,activation='linear')(input_tensor); # (None,2)
model1 = Model(inputs=[input_tensor], outputs=[out1]); 
# We can not directly tape these commands because input_tensor, out1 is Kerastensor (object of Keras) not tensorflow tensor.


#%% Creating second Keras model
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
        # xy:(None,2)
        # model_passed: input:(None,2), output:(None,2)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([xy]);
            z = model_passed(xy);
            z1 = z[:,0:1]; # 0:1 -> to preserve dimension (end value is excluded in Python)
            z2 = z[:,1:2];
            dz1_dxy = tape.gradient(z1,xy); 
            dz2_dxy = tape.gradient(z2,xy);
            dz1_dx  = dz1_dxy[:,0:1];
            dz2_dy  = dz2_dxy[:,1:2];
             
            # tape.gradient(y1,x[:,0:1]) is wrong.
            # Gradient is to be computed of a part of an output with 
            # respect to the ENTIRE variable.
        
        result=tf.concat([z1, z2, dz1_dx, dz2_dy], axis=1);
        return result;

# Defining second Keras model
out2=cust_layer_grad(1)(input_tensor, model1);   
    
model=Model(inputs=[input_tensor],outputs=[out2]);   
model.summary();


#%% Compiling Keras model
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0000001));
# Not required here


#%% Prediction
a=numpy.array([[1,2],[3,4]],dtype=numpy.float64)
z_out=model.predict(a)
print(z_out)
print(sess.run(model1.weights))




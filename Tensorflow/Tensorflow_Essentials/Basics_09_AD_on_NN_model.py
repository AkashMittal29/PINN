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
from keras import layers, models, Input
from keras.models import Model
import tensorflow as tf
from keras import backend
import pickle


#%% Executing in graph mode
tf.compat.v1.disable_eager_execution();
sess=tf.compat.v1.Session();
backend.set_session(sess); # Required to tell Keras to use this session.


#%% Creating a Keras model
input_tensor=Input(shape=(1,)); # input: x, (None, 1)
out1=layers.Dense(1,activation='sigmoid')(input_tensor); # (None,1)
model1 = Model(inputs=[input_tensor], outputs=[out1]); 
# We can not directly tape these commands because input_tensor, out1 are Kerastensor (objects of Keras) not tensorflow tensors.

# Defining custom Keras layer
class cust_layer_grad(keras.layers.Layer):
    def __init__(self, units=1, **kwargs):
        super().__init__(**kwargs);
        self.units=units;
        
    def get_config(self):
        config=super().get_config();
        config ["units"]=self.units;
        return config;
    
    def call(self, x, model_passed): 
        with tf.GradientTape() as tape:
            tape.watch([x]);
            y=model_passed(x);
        
        dy_dx=tape.gradient(y,x);
        result=tf.concat([y,dy_dx],axis=1);
        return result;

# Creating second Keras model
out2=cust_layer_grad(1)(input_tensor, model1);   
    
model=Model(inputs=[input_tensor],outputs=[out2]);   
model.summary();


#%% Compiling Keras model
model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.0000001));
# Not required here


#%% Prediction
y=model.predict([1,2,3])




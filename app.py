import streamlit as st 
st.title("Deep Dream Generator")

st.write("Packages Loading....")
import urllib.request
from PIL import Image
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import IPython.display as display
import PIL.Image
from tensorflow.keras.preprocessing import image
import cv2
import time
st.write("Packages Loaded Successfully XD ")
b=0
text = st.text_input("Enter url here....")
if text:
	try:
		urllib.request.urlretrieve(text, "sample.jpg")
		dimg=cv2.imread("sample.jpg")
		st.image(dimg, caption='Uploaded image', use_column_width=True)
		b=1
	except:
		st.write("Inavlid url")
else:
	uploaded_file = st.file_uploader("Choose an image...", type="jpg")
	if uploaded_file is not None:
			image = Image.open(uploaded_file)
			pil_image = Image.open(uploaded_file).convert('RGB') 
			dimg = np.array(pil_image)
			st.image(image,caption="Uploaded image",use_column_width=True)
			b=1

def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)

def show(img):
    display.display(PIL.Image.fromarray(np.array(img)))

base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

st.write(base_model.summary())

names=['mixed3','mixed5']
layers=[base_model.get_layer(name).output for name in names]
dream_model=tf.keras.Model(inputs=base_model.input,outputs=layers)

def calculate_loss(img,model):
    img_batch=tf.expand_dims(img,axis=0)
    layer_activations=model(img_batch)
    if len(layer_activations)==1:
        layer_activations=[layer_activations]
        
    losses=[]
    for activation in layer_activations:
        loss=tf.math.reduce_mean(activation)
        losses.append(loss)
        
    return tf.reduce_sum(losses)

class Deepdream(tf.Module):
    
    def __init__(self,model):
        self.model=model
        
    @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),)
       )
    
    def __call__(self,img,steps,step_size):
        
        print("Tracing the loss")
        
        loss=tf.constant(0.0)
        
        for n in tf.range(steps):
            
            with tf.GradientTape() as tape:
                
                tape.watch(img)
                loss=calculate_loss(img,self.model)
                
            gradients=tape.gradient(loss,img)
            
            gradients /= tf.math.reduce_std(gradients) + 1e-8
            
            img=img+gradients*step_size
            
            img=tf.clip_by_value(img,-1,1)
            
        return loss,img

deepdream=Deepdream(dream_model)

def run_deepdream(img,steps=100,step_size=0.01):
    
    img=tf.keras.applications.inception_v3.preprocess_input(img)
    img=tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining>100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        steps_remaining -= run_steps
        step += run_steps
        
        loss,img=deepdream(img,run_steps,tf.constant(step_size))
             
        his=f"Step {step}, loss {loss}"
        st.write(his)
    result = deprocess(img)

    return result
if b:
	my_bar = st.progress(0)
	for percent_complete in range(80):
		time.sleep(0.1)
		my_bar.progress(percent_complete + 1)
	dreamimage=run_deepdream(img=dimg,steps=100,step_size=0.01)
	image=np.array(dreamimage)
	for percent_complete in range(80,100):
		time.sleep(0.1)
		my_bar.progress(percent_complete + 1)
	st.image(image,caption="Deepdream image",use_column_width=True)

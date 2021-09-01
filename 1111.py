import tensorflow as tf
import numpy as np
input_image = tf.constant([[1., 2., 3., 4.],
                           [4., 5., 6., 4.],
                           [3., 4., 5., 4.],
                           [3., 4., 5., 4.]])
input_image = tf.reshape(input_image, [1, 4, 4, 1]) 
output_valid=tf.keras.layers.Conv2D(inputs=input_image,filters=1,kernel_size=(3,3),strides=(1, 1), padding='valid')
output_same=tf.layers.conv2d(inputs=input_image,filters=1,kernel_size=(3,3),strides=(1, 1), padding='same')
output_valid.get_shape() == [1, 2, 2, 1]  
output_same.get_shape() == [1, 4, 4, 1]  
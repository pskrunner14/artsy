import os
import sys

import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf

from vgg import load_vgg_model, CONFIG
from utils import *

IMAGE_PAIRS = [
    ('taj_mahal.jpg',  'sunrise_monet.jpg'),
    ('stonehenge.jpg', 'impression_1.jpg'),
    ('new_york.jpg', 'impression_2.jpg'),
    ('stone_henge.jpg', 'sunset_at_ivry.jpg'),
    ('arc_de_triomphe.jpg', 'impression_1.jpg'),
    ('arc_de_triomphe.jpg', 'impression_2.jpg'),
    ('arc_de_triomphe.jpg', 'lois_griffel.jpg'),
    ('new_york.jpg', 'sunrise_monet.jpg'),
]

def model_nn(content_image, style_image, num_iterations = 500):

    content_image = scipy.misc.imread(CONFIG.CONTENT_IMAGES_DIR + content_image)
    content_image = reshape_and_normalize_image(content_image)

    style_image = scipy.misc.imread(CONFIG.STYLE_IMAGES_DIR + style_image)
    style_image = reshape_and_normalize_image(style_image)

    def compute_content_cost(a_C, a_G):

        # Retrieve dimensions from a_G
        _, n_H, n_W, n_C = a_G.get_shape().as_list()
        
        # Reshape a_C and a_G
        a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
        a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))
        
        # compute the cost
        J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
        J_content *= (1 / (4 * n_H * n_W * n_C))
        
        return J_content

    def compute_gram_matrix(A):
        
        return tf.matmul(A, tf.transpose(A))

    def compute_layer_style_cost(a_S, a_G):

        # Retrieve dimensions from a_G
        _, n_H, n_W, n_C = a_G.get_shape().as_list()
        
        # Reshape the images to have them of shape (n_C, n_H*n_W)
        a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
        a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

        # Computing gram_matrices for both images S and G
        GS = compute_gram_matrix(a_S)
        GG = compute_gram_matrix(a_G)

        # Computing the loss
        J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) 
        J_style_layer *= (1 / (4 * np.square(n_H * n_W) * np.square(n_C)))
        
        return J_style_layer

    # values for the lambda hyperparam for each layer
    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)
    ]

    def compute_style_cost(model, STYLE_LAYERS):

        # initialize the overall style cost
        J_style = 0

        for layer_name, coeff in STYLE_LAYERS:

            # Select the output tensor of the currently selected layer
            out = model[layer_name]

            # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
            a_S = sess.run(out)

            # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
            # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
            # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
            a_G = out
            
            # Compute style_cost for the current layer
            J_style_layer = compute_layer_style_cost(a_S, a_G)

            # Add coeff * J_style_layer of this layer to overall style cost
            J_style += coeff * J_style_layer

        return J_style

    def total_cost(J_content, J_style, alpha = 10, beta = 40):

        J = (alpha * J_content) + (beta * J_style)
        return J

    # Reset the graph
    tf.reset_default_graph()

    # Start interactive session
    sess = tf.InteractiveSession()

    # Now, we initialize the "generated" image as a noisy image created from the content_image. 
    # By initializing the pixels of the generated image to be mostly noise but still slightly 
    # correlated with the content image, this will help the content of the "generated" image 
    # more rapidly match the content of the "content" image. 
    generated_image = generate_noise_image(content_image)

    model = load_vgg_model(CONFIG.VGG_MODEL)

    # Compute the content cost
    # Assign the content image to be the input of the VGG model.  
    sess.run(model['input'].assign(content_image))
    # Select the output tensor of layer conv4_2
    out = model['conv4_2']
    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)
    a_G = out
    J_content = compute_content_cost(a_C, a_G)

    # Compute the style cost
    # Assign the input of the model to be the "style" image 
    sess.run(model['input'].assign(style_image))
    J_style = compute_style_cost(model, STYLE_LAYERS)

    J = total_cost(J_content, J_style)

    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)
    
    # Initialize global variables
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model..
    sess.run(model['input'].assign(generated_image))
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Save image every 50 iteration.
        if i % 50 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print()
            print("Total cost = " + str(Jt))
            print("Content cost = " + str(Jc))
            print("Style cost = " + str(Js))
            print()
            # save current generated image in the "/output" directory
            # save_image(CONFIG.OUTPUT_DIR + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image(CONFIG.OUTPUT_DIR + content_image + '-' + style_image  +  'gen.jpg', generated_image)
    # return generated_image

for content_image, style_image in IMAGE_PAIRS:
    print('For content: {} and style: {}'.format(content_image, style_image))
    print()
    model_nn(content_image, style_image)


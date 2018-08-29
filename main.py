import os
import sys

import argparse
import logging
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from matplotlib.pyplot import imshow
from PIL import Image

from vgg import load_vgg_model, CONFIG
from utils import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='NST using tensorflow [configuration]')

    parser.add_argument('--num-iters', type=int, default=500, dest='iterations',
                        help='Number of iterations for training')

    parser.add_argument('--save-every', type=int, default=50, dest='save_every',
                        help='Iteration interval after which to save the generated image')
    
    parser.add_argument('--log-level', type=str, default='info', dest='log_level',
                        help='Logging level')

    return parser.parse_args()

IMAGE_PAIRS = [
    ('taj_mahal.jpg',  'sunrise_monet.jpg'),
    ('stonehenge.jpg', 'impression_1.jpg'),
    ('new_york.jpg', 'impression_2.jpg'),
    ('stone_henge.jpg', 'sunset_at_ivry.jpg'),
    ('arc_de_triomphe.jpg', 'impression_1.jpg'),
    ('arc_de_triomphe.jpg', 'impression_2.jpg'),
    ('arc_de_triomphe.jpg', 'lois_griffel.jpg'),
    ('new_york.jpg', 'sunrise_monet.jpg')
]

def read_image(DIR, image_name):
    return reshape_and_normalize_image(scipy.misc.imread(DIR + image_name))

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

def compute_style_cost(sess, model, STYLE_LAYERS):
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
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
    return alpha * J_content + beta * J_style

def model_nn(content_image_name, style_image_name, iterations=500, save_every=50):

    content_image = read_image(CONFIG.CONTENT_IMAGES_DIR, content_image_name)
    style_image = read_image(CONFIG.STYLE_IMAGES_DIR, style_image_name)

    # Reset the graph
    tf.reset_default_graph()

    # Start tensorflow session
    with tf.Session() as sess:

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
        content_cost = compute_content_cost(a_C, a_G)

        # Compute the style cost
        # Assign the input of the model to be the "style" image 
        sess.run(model['input'].assign(style_image))
        style_cost = compute_style_cost(sess, model, STYLE_LAYERS)

        # Define the cost and optimizer
        cost = total_cost(content_cost, style_cost)
        optimizer = tf.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(cost)
        
        sess.run(tf.global_variables_initializer())

        # Run the noisy input image (initial generated image) through the model..
        sess.run(model['input'].assign(generated_image))
        
        for i in range(1, iterations + 1):
        
            # Run the session on the train_step to minimize the total cost
            sess.run(train_step)
            # Compute the generated image by running the session on the current model['input']
            generated_image = sess.run(model['input'])

            # Save image every 50 iteration.
            if i % save_every == 0:
                Jt, Jc, Js = sess.run([cost, content_cost, style_cost])
                logging.info("Iteration {}\n".format(i))
                logging.info("Total cost: {:.4f}".format(Jt))
                logging.info("Content cost: {:.4f}".format(Jc))
                logging.info("Style cost: {:.4f}\n".format(Js))

                image_path = '{}{}_{}_{}.jpg'.format(CONFIG.OUTPUT_AUX_DIR, 
                            content_image_name[: -4], style_image_name[: -4], i)
                save_image(image_path, generated_image)
    
    image_path = '{}{}_{}_gen.jpg'.format(CONFIG.OUTPUT_DIR, 
                content_image_name[: -4], style_image_name[: -4])
    save_image(image_path, generated_image)
    return generated_image

def main():
    args = parse_arguments()

    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, args.log_level.upper()))

    for content_image, style_image in IMAGE_PAIRS:
        print('For content: {} and style: {}\n'.format(content_image, style_image))
        model_nn(content_image, style_image, iterations=args.iterations, save_every=args.save_every)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('EXIT')
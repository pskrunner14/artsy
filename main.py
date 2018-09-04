import os
import argparse
import logging
import scipy.io
import scipy.misc
import numpy as np
import tensorflow as tf

from vgg import load_vgg_model, CONFIG, STYLE_LAYERS
from utils import generate_noise_image, reshape_and_normalize_image, save_image

IMAGE_PAIRS = [
    ('village.jpg', 'lois_griffel.jpg'),
    ('taj_mahal.jpg',  'sunrise_monet.jpg'),
    ('arc_de_triomphe.jpg', 'starry_night.jpg'),
    ('new_york.jpg', 'sunrise_monet.jpg'),
    ('stone_henge.jpg', 'impression_1.jpg'),
    ('new_york.jpg', 'impression_2.jpg'),
    ('stone_henge.jpg', 'sunset_at_ivry.jpg'),
    ('arc_de_triomphe.jpg', 'impression_1.jpg'),
    ('arc_de_triomphe.jpg', 'impression_2.jpg'),
    ('arc_de_triomphe.jpg', 'lois_griffel.jpg')
]

def parse_arguments():
    parser = argparse.ArgumentParser(description='NST using tensorflow [configuration]')

    parser.add_argument('--num-iters', type=int, default=500, dest='iterations',
                        help='Number of iterations for training')

    parser.add_argument('--save-every', type=int, default=50, dest='save_every',
                        help='Iteration interval after which to save the generated image')
    
    parser.add_argument('--log-level', type=str, default='info', dest='log_level',
                        help='Logging level')

    return parser.parse_args()

def compute_content_cost(a_C, a_G):
        _, n_H, n_W, n_C = a_G.get_shape().as_list()
        
        a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
        a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

        J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
        J_content *= (1 / (4 * n_H * n_W * n_C))

        return J_content

def compute_gram_matrix(A):
    return tf.matmul(A, A, transpose_b=True)

def compute_layer_style_cost(a_S, a_G):
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))
    
    GS = compute_gram_matrix(a_S)
    GG = compute_gram_matrix(a_G)

    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) 
    J_style_layer *= (1 / (4 * np.square(n_H * n_W) * np.square(n_C)))
    
    return J_style_layer

def compute_style_cost(sess, model, STYLE_LAYERS):
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)
        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    return alpha * J_content + beta * J_style

def read_image(DIR, image_name, image_shape=None):
    return reshape_and_normalize_image(scipy.misc.imread(DIR + image_name), image_shape=image_shape)

class NeuralStyleTransfer():

    def __init__(self, content_image_name=None, style_image_name=None, iterations=1500, save_every=50):
        if content_image_name is None or style_image_name is None:
            raise UserWarning('Image names should not be empty!')
        self.content_image_name = content_image_name[: -4]
        self.style_image_name = style_image_name[: -4]
        self.iterations = iterations
        self.save_every = save_every

        self.model = load_vgg_model()
        # self.content_image, self.style_image = None, None
        self._preprocess_images(content_image_name, style_image_name)
        
    def _preprocess_images(self, content_image_name, style_image_name):
        # Reshape and Normalize images
        self.content_image = read_image(CONFIG.CONTENT_IMAGES_DIR, content_image_name)
        _, height, width, _ = self.content_image.shape
        self.style_image = read_image(CONFIG.STYLE_IMAGES_DIR, style_image_name, (height, width))

    def _train(self, sess, train_step):
        sess.run(train_step)
        # Compute the generated image by running the session on the current model['input']
        return sess.run(self.model['input'])

    def _train_iters(self, sess, costs, noise_image):

        content_cost, style_cost = costs

        # Define the cost and optimizer
        cost = total_cost(content_cost, style_cost)
        optimizer = tf.train.AdamOptimizer(1.0)
        train_step = optimizer.minimize(cost)
        
        sess.run(tf.global_variables_initializer())

        # Run the noisy input image (initial generated image) through the model..
        sess.run(self.model['input'].assign(noise_image))
        
        for i in range(1, self.iterations + 1):
            generated_image = self._train(sess, train_step)

            if i % self.save_every == 0:
                Jt, Jc, Js = sess.run([cost, content_cost, style_cost])
                self._log_results(i, [Jt, Jc, Js])
                self._save_image(generated_image, iteration=i)

        return generated_image

    def generate_image(self):

        with tf.Session() as sess:
            # Now, we initialize the "generated" image as a noisy image created from the content_image. 
            # By initializing the pixels of the generated image to be mostly noise but still slightly 
            # correlated with the content image, this will help the content of the "generated" image 
            # more rapidly match the content of the "content" image. 
            noise_image = generate_noise_image(self.content_image)

            # Assign the content image to be the input of the VGG model.  
            sess.run(self.model['input'].assign(self.content_image))
            # Select the output tensor of layer conv4_2
            out = self.model['conv4_2']
            # Set a_C to be the hidden layer activation from the layer we have selected
            a_C = sess.run(out)
            a_G = out
            # Compute the content cost
            content_cost = compute_content_cost(a_C, a_G)

            # Assign the input of the model to be the "style" image 
            sess.run(self.model['input'].assign(self.style_image))
            # Compute the style cost
            style_cost = compute_style_cost(sess, self.model, STYLE_LAYERS)

            generated_image = self._train_iters(sess, (content_cost, style_cost), noise_image)

        tf.reset_default_graph()
        
        self._save_image(generated_image)
        return generated_image

    def _log_results(self, iteration, costs):
        logging.info("Iteration {}\n".format(iteration))
        logging.info("Total cost: {:.4f}".format(costs[0]))
        logging.info("Content cost: {:.4f}".format(costs[1]))
        logging.info("Style cost: {:.4f}\n".format(costs[2]))

    def _save_image(self, generate_image, iteration=None):
        if iteration is None:
            image_path = '{}{}_{}_gen.jpg'.format(CONFIG.OUTPUT_DIR, 
                        self.content_image_name, self.style_image_name)
        else:
            image_path = '{}{}_{}_{}.jpg'.format(CONFIG.OUTPUT_AUX_DIR, 
                        self.content_image_name, self.style_image_name, iteration)
        save_image(image_path, generate_image)

def main():
    
    # Create auxillary output directory if it does'nt exist
    if not os.path.isdir(CONFIG.OUTPUT_AUX_DIR):
        os.mkdir(CONFIG.OUTPUT_AUX_DIR)

    args = parse_arguments()

    LOG_FORMAT = '%(levelname)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, args.log_level.upper()))

    for content_image, style_image in IMAGE_PAIRS:
        print('For content: {} and style: {}\n'.format(content_image, style_image))
        nst_model = NeuralStyleTransfer(content_image, style_image, 
                    iterations=args.iterations, save_every=args.save_every)
        _ = nst_model.generate_image()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('EXIT')
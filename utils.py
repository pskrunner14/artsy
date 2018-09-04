import scipy
import numpy as np

from vgg import CONFIG

def generate_noise_image(content_image, noise_ratio = CONFIG.NOISE_RATIO):
    """
    Generates a noisy image by adding random noise to the content_image
    """
    _, height, width, channels = content_image.shape
    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, (1, height, width, channels)).astype('float32')
    
    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    
    return input_image


def reshape_and_normalize_image(image, image_shape=None):
    """
    Reshape and normalize the input image (content or style)
    """
    if image_shape == None:
        image_shape = (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH)
        
    try:
        resized_image = scipy.misc.imresize(image, image_shape)
        # Reshape image to mach expected input of VGG16
        image = np.reshape(resized_image, ((1, ) + resized_image.shape))
    except ValueError as e:
        print(str(e))
        exit(0)
    # Substract the mean to match the expected input of VGG16
    image = image - CONFIG.MEANS
    return image


def save_image(path, image):
    # Un-normalize the image so that it looks good
    image = image + CONFIG.MEANS
    
    # Clip and Save the image
    image = np.clip(image[0], 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)
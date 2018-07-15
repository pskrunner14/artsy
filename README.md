# Overview ![Under Development](./data/under-development-orange.svg)

Neural Style Transfer is a fairly exciting and fun application of Deep Learning, as it can generate quite genuine looking artistic transmutations of an ordinary photograph by augmenting onto it the artistic style of any painting of your liking. So essentially, we can train a network to transform any image into a stylized version of it, based on the style of our handpicked image. That is incredible! In case you were wondering, this technology is exactly how apps like [Prisma](https://prisma-ai.com/) work.

Here are a few output examples:

![“Neckarfront”, Tubingen, Germany in various styles of different paintings](data/ns1.png)

!["Mona Lisa" by Leonardo da Vinci](data/ns2.png)

## Data

*Neural Style Transfer* (NST) uses a previously trained Convolutional Neural Network, and builds on top of that. This concept of using a network trained on a different task (such as *Object Detection and Recognition*) and applying it to a new task is called *Transfer Learning*.

As per the original paper - [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), we will use the *VGG network*. Specifically, we'll use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large *ImageNet* database, and thus has learned to recognize a variety of low level features (at the earlier layers) and high level features (at the deeper layers).

The VGG-19 (imagenet-vgg-verydeep-19.mat) model can be downloaded from [here](http://www.vlfeat.org/matconvnet/pretrained/).

## Getting Started

In order to be able to generate neural style artistic images, you will need to install the required python packages:

`pip install -r requirements.txt`

Now you can open up a terminal and run the network on your content image:

`python main.py`

Now you are all set up!

## Results



## Built With

* Python
* TensorFlow
* NumPy
* Matplotlib
* SciPy

## Authors

Prabhsimran Singh

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
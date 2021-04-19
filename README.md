# dog-breed-identification
Identify the breed of the dog by image.

## Dataset : https://www.kaggle.com/c/dog-breed-identification/data


# Introduction
This is the code for a [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification/overview), a [Kaggle](https://www.kaggle.com/) Competition. This was an open competiton from September 29, 2017 to February 27, 2018.

In this project we're going to be using machine learning to help us identify different breeds of dogs.

To do this, we'll be using data from the [Kaggle dog breed identification competition](https://www.kaggle.com/c/dog-breed-identification/data). It consists of a collection of 10,000+ labelled images of 120 different dog breeds. We will process the images with the labels i.e the breeds of the dogs in the images and feed it into a machine learning model which then processes the images and finds patterns and similarities within the images. So when we give it an image with the one that we want to identify, it makes use of the patterns and the similarities from the images given to it and identifies the breed of the dog.

This kind of problem is called multi-class image classification. It's multi-class because we're trying to classify mutliple different breeds of dog. If we were only trying to classify dogs versus cats, it would be called binary classification (one thing versus another).

We'll use an existing model from TensorFlow Hub.
TensorFlow Hub is a resource where you can find pretrained machine learning models for the problem you're working on.
Using a pretrained machine learning model is often referred to as transfer learning.

### What is Multi-class Image Classification?
Multi-class classification refers to those classification tasks that have more than two class labels. Consider an example, for any movie, Central Board of Film Certification, issue a certificate depending on the contents of the movie. A movie is rated as ‘U/A’ (meaning ‘Parental Guidance for children below the age of 12 years’) certificate. There are other types of certificates classes like ‘A’ (Restricted to adults) or ‘U’ (Unrestricted Public Exhibition), but it is sure that each movie can only be categorised with only one out of those three type of certificates.

In short, there are multiple categories but each instance is assigned only one, therefore such problems are known as multi-class classification problem.

Other similar examples include:

* Face classification.
* Fruits and vegetables recognition.
* Optical character recognition.
* Hand written digit recognition.

Read more at : https://medium.com/@srijaneogi31/exploring-multi-class-classification-using-deep-learning-cd3134290887

### What is Transfer Learning?
In transfer learning, the knowledge of an already trained machine learning model is applied to a different but related problem. For example, if you trained a simple classifier to predict whether an image contains a backpack, you could use the knowledge that the model gained during its training to recognize other objects like sunglasses.

With transfer learning, we basically try to exploit what has been learned in one task to improve generalization in another. We transfer the weights that a network has learned at "task A" to a new "task B."

The general idea is to use the knowledge a model has learned from a task with a lot of available labeled training data in a new task that doesn't have much data. Instead of starting the learning process from scratch, we start with patterns learned from solving a related task.

Transfer learning has several benefits, but the main advantages are  saving training time, better performance of neural networks (in most cases), and not needing a lot of data. 

Usually, a lot of data is needed to train a neural network from scratch but access to that data isn't always available — this is where transfer learning comes in handy. With transfer learning a solid machine learning model can be built with comparatively little training data because the model is already pre-trained. This is especially valuable in natural language processing because mostly expert knowledge is required to create large labeled datasets. Additionally, training time is reduced because it can sometimes take days or even weeks to train a deep neural network from scratch on a complex task.

Read more at : https://builtin.com/data-science/transfer-learning


### Why use a pretrained model?
Building a machine learning model and training it on lots from scratch can be expensive and time consuming.
Transfer learning helps eliviate some of these by taking what another model has learned and using that information with your own problem.

# Usage

## Clone Repository
Clone this Repository using:

	git clone https://github.com/mayursrt/dog-breed-identification.git

## Execute Code
You can use [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb) for the easy execution of the code and skip the following steps.

Install `jupyter` from [here](http://jupyter.readthedocs.io/en/latest/install.html) or use

	pip install jupyter

After installing jupyter notebook Just run `jupyter notebook` in terminal and you can visit the notebook in your web browser.



## Dependencies

* [Pandas](https://pandas.pydata.org/docs/)
* [NumPy](https://numpy.org/devdocs/user/index.html)
* [Sklearn](https://scikit-learn.org/stable/)
* [Matplotlib](https://matplotlib.org/3.3.3/contents.html)
* [TensorFlow](https://www.tensorflow.org/guide)

Install missing dependencies using,

	pip install pandas numpy sklearn matplotlib tensorflow



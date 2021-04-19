# dog-breed-identification
Identify the breed of the dog by image.

## Dataset : https://www.kaggle.com/c/dog-breed-identification/data


# Introduction
This is the code for a [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification/overview), a [Kaggle](https://www.kaggle.com/) Competition. This was an open competiton from September 29, 2017 to February 27, 2018.

In this project we're going to be using machine learning to help us identify different breeds of dogs.

To do this, we'll be using data from the [Kaggle dog breed identification competition](https://www.kaggle.com/c/dog-breed-identification/data). It consists of a collection of 10,000+ labelled images of 120 different dog breeds. We will process the images and Feed it into a machine learning model

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

# Usage

## Clone Repository
Clone this Repository using,

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



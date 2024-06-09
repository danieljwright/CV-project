# CNN Similarity and Warm-Starting Ability
Authors: Daniel Wright (5932033) & Kunal & Alex

In this blog post we present the results of our study into whether CNN's learn similar filters and if these learned weights can be used to warm start a model. This work is part of the course CS4245 Computer Vision by Deep Learning in 2023/2024 at TU Delft. 

## 1. Introduction

As the applications of computer vision have grown and deep learning models have become more ubiqitous, the environmental impact of training these models becomes apparent. One solution to the water usage and carbon emissions of training models is to use warm-starting. This is the process of initalising model weights using weights from previously trained models, instead of random initialisation. Prior work has shown that warm starting neural networks can harm generalization, with a "shrink and perturb" trick being applied to warm started weights to try and help warm started networks perform better. We explore warm starting in a different context to the methods utilized before. Our idea is to train a simple Convolutional Neural Network (CNN) on 6 different noisy subsets of the MNIST Dataset, which is a dataset of greyscale images of size 28*28 of the numbers 0-9. The final convolutional filters that we get corresponding to training the model on these noisy datasets are compared for similarities, and then we use filters that are similar across different models to warm-start the same Convolutional Neural Network on the entire MNIST dataset, to see if we can get better accuracy on the MNIST test dataset compared to simply training our Convolutional Neural Network with random initializtion on the entire MNIST dataset.

## 2. Methodology

We first shuffle the MNIST training dataset, which consists of 60000 images. We then take 6 different subsets of this shuffled dataset, and apply a different noise to each of these subsets.

1) The noise that we apply to the first subset is a perspective transform.




## 5. Results

With fixed noise levels: 20x runs with the three different similarity measures, record train_loss and test accuracy. calculate average and std dev. Also record similarity scores somehow??

Calculate similarity between random and final, and then intial warm start and final warm start and compare the result.


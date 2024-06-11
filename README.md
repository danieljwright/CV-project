# CNN Similarity and Warm-Starting Ability
Authors: Daniel Wright (5932033) & Kunal Kaushik (6050549) & Alex

In this blog post we present the results of our study into whether CNN's learn similar filters and if these learned weights can be used to warm start a model. This work is part of the course CS4245 Computer Vision by Deep Learning in 2023/2024 at TU Delft. 

## 1. Introduction

As the applications of computer vision have grown and deep learning models have become more ubiqitous, the environmental impact of training these models becomes apparent. One solution to the water usage and carbon emissions of training models is to use warm-starting. This is the process of initalising model weights using weights from previously trained models, instead of random initialisation. Prior work has shown that warm starting neural networks can harm generalization, with a "shrink and perturb" trick being applied to warm started weights to try and help warm started networks perform better. We explore warm starting in a different context to the methods utilized before. Our idea is to train a simple Convolutional Neural Network (CNN) on 6 different noisy subsets of the MNIST Dataset, which is a dataset of greyscale images of size 28*28 of the numbers 0-9. The final convolutional filters that we get corresponding to training the model on these noisy datasets are compared for similarities, and then we use filters that are similar across different models to warm-start the same Convolutional Neural Network on the entire MNIST dataset, to see if we can get better accuracy on the MNIST test dataset compared to simply training our Convolutional Neural Network with random initializtion on the entire MNIST dataset.

## 2. Methodology

We first shuffle the MNIST training dataset, which consists of 60000 images. We then take 6 different subsets of this shuffled dataset, with 10000 images in each subset, and apply a different noise to each of these subsets.

1) The noise that we apply to the first subset is a random perspective transform on each input image, using torchvision.transforms.v2.RandomPerspective(). We set the probability of an image being transformed as 1, as we want all images to be transformed. The degree of distortion is set to 0.6, and the fill value is set to -1, so areas outside the transformed image are filled using the edge pixel value. An example of such a transformation is shown in the image below.

2) For the second subset, we apply random gaussian blur to each image, using torchvision.transforms.v2.GaussianBlur(), using a gaussian kernel os size 5, and a standard deviation uniformly chosen from the range (1, 5) for creating the kernel that will do the blurring.

3) For the third subset, for each image, we create a 28*28 array where all values are sampled from a Gaussian distribution with mean 0 and standard deviation 1, and we add this gaussian noise to the image.

4) For the fourth subset, we randomly erase a rectangular region from the images with probability 0.5, using torchvision.transforms.v2.RandomErasing(), with an erasing value of -1.

5) For the fifth subset, we perform an elastic transform on the images, using torchvision.transforms.v2.ElasticTransform(), taking magnitude of displacements as 125.

6) Finally, for the sixth subset, we randomly rotate the images using torchvision.transforms.v2.RandomRotation() upto 30 degrees.

We then define a simple Convolutional Neural Network, which will train on all of these subsets, and the entire training dataset (without any noise) as well. Our CNN has two convolutional layers, the first one having 1 input channel and 2 output channels, and the second one having 2 input and 4 output channels, both having kernel sizes of 3, and stride and padding of 1. Both convolutional layers are followed by max pooling layers using a kernel size of 2, stride of 2, and a padding of 0. These are followed by two fully connected linear layers, with the last layer having 10 outputs corresponding to the 10 classes. We train this network across all datasets using the Adam optimizer using a learning rate of 0.001, a batch size of 64, for 1 epoch, and the loss function that is optimized is the Cross Entropy Loss. We have a deliberately chosen a very simple CNN, and a simple greyscale dataset, as we can scale up the network and dataset complexity in case of positive results with a smimpler dataset and network, and as it is wise in the interest of energy efficiency to initially do a scaled down version of this research. Also, it is not so hard to achieve more than 99% test accuracy for MNIST with more complex networks, so we choose to use a simple network so we can observe if there are any gains using our similarity-based warm started network.

The three similarity metrics that we use to compare filters across our six different models are as follows : 

1) Cosine Similarity : We flatten the two filters to be compared so that they become one dimensional vectors and then compute the cosine similarity between them using torch.nn.functional.cosine_similarity() which emplys the following formula, similarity = (x1 ⋅ x2) / (max(∥x1∥2, ϵ) ⋅ max(∥x2∥2, ϵ))
where x_1 and x_2 are the flattened vectors.
2) 




## 3. Results

We obtained the average similarity scores for 10 iterations for each of the 10 filters in the model. These are shown below in Table 1. 

The model warm-started using the Pearson similarity metric performed the best, with an average test accuracy similar to that of the model that was randomly initalised. The average test accuracy of the warm-started model trained for each similarity metric and the average test accuracy of the randomly intialised model, are shown in Table 2.

## 4. Discussion

Do these six models trained on the noisy datasets converge to similar learned weights?

If we use these learned weights to warm-start a network on a dataset which includes all the different noises/artifacts does it perform better in terms of classification accuracy than random initialization?




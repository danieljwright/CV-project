# CNN Similarity and Warm-Starting Ability
Authors: Daniel Wright (5932033) & Kunal Kaushik (6050549) & Alex

In this blog post we present the results of our study into whether CNN's learn similar filters and if these learned weights can be used to warm start a model. This work is part of the course CS4245 Computer Vision by Deep Learning in 2023/2024 at TU Delft. 

## 1. Introduction

As the applications of computer vision have grown and deep learning models have become more ubiqitous, the environmental impact of training these models becomes apparent. One solution to the water usage and carbon emissions of training models is to use warm-starting. This is the process of initalising model weights using weights from previously trained models, instead of random initialisation. Prior work has shown that warm starting neural networks can harm generalization, with a "shrink and perturb" trick being applied to warm started weights to try and help warm started networks perform better. We explore warm starting in a different context to the methods utilized before. Our idea is to train a simple Convolutional Neural Network (CNN) on 6 different noisy subsets of the MNIST Dataset, which is a dataset of greyscale images of size 28*28 of the numbers 0-9. The final convolutional filters that we get corresponding to training the model on these noisy datasets are compared for similarities, and then we use filters that are similar across different models to warm-start the same Convolutional Neural Network on the entire MNIST dataset, to see if we can get better accuracy on the MNIST test dataset compared to simply training our Convolutional Neural Network with random initializtion on the entire MNIST dataset.

## 2. Methodology

We first shuffle the MNIST training dataset, which consists of 60000 images. We then take 6 different subsets of this shuffled dataset, with 10000 images in each subset, and apply a different noise to each of these subsets.

1) The noise that we apply to the first subset is a random perspective transform on each input image, using torchvision.transforms.v2.RandomPerspective(). We set the probability of an image being transformed as 1, as we want all images to be transformed. The degree of distortion is set to 0.3, and the fill value is set to -1, so areas outside the transformed image are filled using the edge pixel value. An example of such a transformation is shown in the image below.

2) For the second subset, we apply random gaussian blur to each image, using torchvision.transforms.v2.GaussianBlur(), using a gaussian kernel os size 5, and a standard deviation uniformly chosen from the range (1, 2) for creating the kernel that will do the blurring.

3) For the third subset, for each image, we create a 28*28 array where all values are sampled from a Gaussian distribution with mean 0 and standard deviation 0.5, and we add this gaussian noise to the image.

4) For the fourth subset, we randomly erase a rectangular region from the images with probability 0.5, using torchvision.transforms.v2.RandomErasing(), with an erasing value of -1.

5) For the fifth subset, we perform an elastic transform on the images, using torchvision.transforms.v2.ElasticTransform(), taking magnitude of displacements as 50.

6) Finally, for the sixth subset, we randomly rotate the images using torchvision.transforms.v2.RandomRotation() upto 30 degrees.

We then define a simple Convolutional Neural Network, which will train on all of these subsets separately, and the entire training dataset (without any noise) as well. Our CNN has two convolutional layers, the first one having 1 input channel and 2 output channels, and the second one having 2 input and 4 output channels, both having kernel sizes of 3, and stride and padding of 1. Both convolutional layers are followed by max pooling layers using a kernel size of 2, stride of 2, and a padding of 0. These are followed by two fully connected linear layers, with the last layer having 10 outputs corresponding to the 10 classes. We train this network across all datasets using the Adam optimizer using a learning rate of 0.001, a batch size of 64, for 6 epochs, and the loss function that is optimized is the Cross Entropy Loss. We have a deliberately chosen a very simple CNN, and a simple greyscale dataset, as we can scale up the network and dataset complexity in case of positive results with a smimpler dataset and network, and as it is wise in the interest of energy efficiency to initially do a scaled down version of this research. Also, it is not so hard to achieve more than 99% test accuracy for MNIST with more complex networks, so we choose to use a simple network so we can observe if there are any gains using our similarity-based warm started network.

The three similarity metrics that we use to compare filters across our six different models are as follows : 

1) Cosine Similarity : We flatten the two filters to be compared so that they become one dimensional vectors and then compute the cosine similarity between them using torch.nn.functional.cosine_similarity() as follows,

![similarity](https://latex.codecogs.com/png.latex?\dpi{150}\color{white}%20\text{similarity}%20=%20\frac{x_1%20\cdot%20x_2}{\max(%7C%7Cx_1%7C%7C_2,%20\epsilon)%20\cdot%20\max(%7C%7Cx_2%7C%7C_2,%20\epsilon)})

where x<sub>1</sub> and x<sub>2</sub> are the flattened vectors.
A score of -1 represents oppositely pointing vectors, while a score of 1 means that the vectors have the same orientation and direction. A score of 0 represents orthogonal vectors. We return the absolute value of cosine similarity, as we believe that oppositely pointing vectors represent a case where the filters are trying to learn the same concept in opposite manners to each other in space, and hence essentially, they are similar.

2) Frobenius Norm : We take two matrices corresponding to the two filters and subtract one from the other, and take the frobenius norm of the resulting matrix, using numpy.linalg.norm(). setting order = 'fro'. It is computed as follows,

![Frobenius Norm](https://latex.codecogs.com/png.latex?\dpi{150}\color{white}%5C%7CA%20-%20B%5C%7C_F%20%3D%20%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5Em%20%5Csum_%7Bj%3D1%7D%5En%20%7Ca_%7Bij%7D%20-%20b_%7Bij%7D%7C%5E2%7D)

Where:
- A and B are mxn (28x28) matrices.
- a<sub>ij</sub> and b<sub>ij</sub> are the elements in the i-th row and j-th column of A and B, respectively.
The frobenius norm can range from 0 to &infin;, with a score closer to 0 implying more similarity between the filters.

3) Pearson Correlation Coefficient : We flatten the two filters into 1-D vectors and then find the Pearson Correlation Coefficient between them, using numpy.corrcoef(), using the given formula,

![Pearson correlation coefficient formula](https://latex.codecogs.com/png.latex?\dpi{150}&space;\color{white}&space;r&space;=&space;\frac{\sum_{i=1}^{n}&space;(X_i&space;-&space;\bar{X})&space;(Y_i&space;-&space;\bar{Y})}{\sqrt{\sum_{i=1}^{n}&space;(X_i&space;-&space;\bar{X})^2}&space;\sqrt{\sum_{i=1}^{n}&space;(Y_i&space;-&space;\bar{Y})^2}})

- X : The first vector.
- Y : The second vector.
- X<sub>i</sub> : The i-th element in the vector X.
- Y<sub>i</sub> : The i-th element in the vector Y.
- ![X bar](https://latex.codecogs.com/png.latex?\dpi{100}&space;\color{white}&space;\bar{X}): The mean of vector X.
- ![Y bar](https://latex.codecogs.com/png.latex?\dpi{100}&space;\color{white}&space;\bar{Y}): The mean of vector Y.
- n : The number of elements (length) of the vectors X and Y.
A score of -1 implies perfect positive linear relationship, while a score of -1 implies a score of perfect negative linear relationship. A score of 0 implies that there is no linear relationship between the two filters. SImilarly to the cosine similarity case, we return the absolute value of the pearson correlation coefficent, for the same reasons.

We have 8 convolutional filters in our model architecture in total, and after training our model separately on the 6 noisy datasets, we obtain, for each filter, its pairwise similarity across all the 6 models. For example, for filter 1, we obtain its pairwise similarity for the following pairs of models : { (1, 2), (1, 3), (1,4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6) }. We take the pair with the maximum similarity, and randomly select one filter from the pair. For our final warm-started model, which will be trained on the entire noiseless dataset with 60000 images, we initialize each convolutional filter with the corresponding selected filter. Note that we are not initializing biases, and we are only initializing convoulutional filters, not the parameters of the fully connected layers.

## 3. Results

We obtained the average similarity scores for 10 iterations for each of the 10 filters in the model. These are shown below in Table 1. 

### Table 3: Average Similarity Scores

| Metric                         | Filter 1   | Filter 2   | Filter 3   | Filter 4   | Filter 5   | Filter 6   | Filter 7   | Filter 8   | Filter 9   | Filter 10  |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| **Cosine similarity**          |            |            |            |            |            |            |            |            |            |            |
| Avg                            | 0.31619102 | 0.39206314 | 0.32767773 | 0.33037522 | 0.32841057 | 0.3435595  | 0.3361762  | 0.37068677 | 0.32395244 | 0.31451932 |
| Std                            | 0.25814608 | 0.27347025 | 0.23110147 | 0.22741994 | 0.21829844 | 0.22780581 | 0.21624844 | 0.24406238 | 0.22018431 | 0.22252244 |
| **Pearson coefficient**        |            |            |            |            |            |            |            |            |            |            |
| Avg                            | 0.34174002 | 0.37455269 | 0.3160879  | 0.29002075 | 0.31419466 | 0.31730315 | 0.37767744 | 0.34377837 | 0.3730682  | 0.35122528 |
| Std                            | 0.234311   | 0.25126004 | 0.21325038 | 0.20954852 | 0.20930642 | 0.22417264 | 0.2118101  | 0.21775986 | 0.25721901 | 0.24133763 |
| **Frobenius norm**             |            |            |            |            |            |            |            |            |            |            |
| Avg                            | 1.7596636  | 1.6536038  | 1.1763726  | 1.2730144  | 1.2649969  | 1.3172685  | 1.2530814  | 1.1828452  | 1.1642699  | 1.2038964  |
| Std                            | 0.47468022 | 0.41801614 | 0.3258538  | 0.37116513 | 0.33728537 | 0.35514113 | 0.32381335 | 0.3204852  | 0.33116174 | 0.35387298 |




The model warm-started using the Pearson similarity metric performed the best, with an average test accuracy similar to that of the model that was randomly initalised. The average test accuracy of the warm-started model trained for each similarity metric and the average test accuracy of the randomly intialised model, are shown in Table 2.

### Table 2: Average Test Accuracy

| Method                  | Average Accuracy (%) |
|-------------------------|----------------------|
| Cosine similarity       | 91.143               |
| Frobenius norm          |   92.843             |
| Pearson coefficient     | 94.09                |
| Random initialisation   | 94.62                |

## 4. Discussion

Do these six models trained on the noisy datasets converge to similar learned weights?

If we use these learned weights to warm-start a network on a dataset which includes all the different noises/artifacts does it perform better in terms of classification accuracy than random initialization?




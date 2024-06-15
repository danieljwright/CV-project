# Warm Starting Convolutional Neural Networks and Filter Similarity
Authors: Daniel Wright (5932033) & Kunal Kaushik (6050549) & Alexandros Theocharous 5930901

In this blog post we present the results of our study into whether a CNN can be warm started using convolutional filters that are found to be similar after training the same CNN across different noisy subsets of the original dataset. This can potentially lead to better generalization accuracy than simply randomly initializing the filters for training, and decrease training time in the long run as weights can be reused. This work is part of the course CS4245 Computer Vision by Deep Learning in 2023/2024 at TU Delft. 

## 1. Introduction

As the applications of computer vision have grown and deep learning models have become more ubiqitous, the environmental impact of training these models becomes apparent. One solution to the water usage and carbon emissions of training models is to use warm-starting. This is the process of initalising model weights using weights from previously trained models, instead of random initialisation. Prior work has shown that warm starting neural networks can harm generalization, for example, in the case when weights learned from training on 50% of the data are used to initialize training on the entire dataset [1]. We explore warm starting in a different context to the methods utilized before. Our idea is to train a simple Convolutional Neural Network (CNN) on 6 different noisy subsets of the MNIST Dataset, which together comprise the entire dataset. The MNIST Dataset is a set of greyscale images of size 28*28 of handwritten numbers 0-9. The final convolutional filters that we get from training the model on these noisy datasets are compared for similarities, and then we use filters that are similar across different models to warm-start the same Convolutional Neural Network on the entire MNIST dataset, to see if we can get better accuracy on the MNIST test dataset compared to simply training our Convolutional Neural Network with random initializtion on the entire MNIST dataset.

## 2. Methodology

We first shuffle the MNIST training dataset, which consists of 60000 images. We then take 6 different subsets of this shuffled dataset, with 10000 images in each subset, and apply a different noise to each of these subsets. These noises were chosen such that they are meaningful in a grey-scale image digit classification context. As such we have not used any color transformations. Examples of the transformations are shown in the image below.

1) The noise that we apply to the first subset is a random perspective transform on each input image, using torchvision.transforms.v2.RandomPerspective(). We set the probability of an image being transformed as 1, as we want all images to be transformed. The degree of distortion is set to 0.3, and the fill value is set to -1, so areas outside the transformed image are filled using the background value. This mimics a digit being angled away or towards the camera in an image.

2) For the second subset, we apply random gaussian blur to each image, using torchvision.transforms.v2.GaussianBlur(), using a gaussian kernel of size 5, and a standard deviation uniformly chosen from the range (1, 2) for creating the kernel that will do the blurring. This approximates an image being taken with a lower quality camera, or the image being taken at a distance.

3) For the third subset, for each image, we create a 28*28 array where all values are sampled from a Gaussian distribution with mean 0 and standard deviation 0.5, and we add this gaussian noise to the image. This imitiates a brighter image, with overall higher pixel values.

4) For the fourth subset, we randomly erase a rectangular region from the images with probability 0.5, using torchvision.transforms.v2.RandomErasing(), with an erasing value of -1. This approximates the digit being partially occluded.

5) For the fifth subset, we perform an elastic transform on the images, using torchvision.transforms.v2.ElasticTransform(), taking magnitude of displacements as 50. This approximates different handwriting styles.

6) Finally, for the sixth subset, we randomly rotate the images using torchvision.transforms.v2.RandomRotation() up to 30 degrees. This transform mimics the digit being rotated in an image, and also different handwriting styles to an extent. However, due to the rotational symmetry of certain digits, only rotation of up to 30 degrees was used.


![Transformation Pipeline](https://github.com/danieljwright/CV-project/raw/main/transformation_pipeline.jpg)


We then define a simple Convolutional Neural Network, which we will train on all of these subsets separately, and the entire training dataset (without any noise) as well, to create 7 different networks. Our CNN has two convolutional layers, the first one having 1 input channel and 2 output channels, and the second one having 2 input and 4 output channels, both having kernel sizes of 3, and stride and padding of 1. Both convolutional layers are followed by max pooling layers using a kernel size of 2, stride of 2, and a padding of 0. These are followed by two fully connected linear layers, with the last layer having 10 outputs corresponding to the 10 classes. We train this network across all datasets using the Adam optimizer using a learning rate of 0.001, a batch size of 64, for 6 epochs, and the loss function that is optimized is the Cross Entropy Loss. We then check for similarities between the trained filters to warm start training on the entire noiseless MNIST dataset using the most similar filters. We also train the same network on the entire dataset using randomly initialized weights, and compare the classification accuracy on the unseen test dataset for both methods. Both of these training procedures are run for 1 epoch. We believe that the trained filters that are found to be similar across different noisy datasets are noise-invariant. These filters can be used to effectively guide further training in an optimal direction, helping the network generalize better.

We have a deliberately chosen a very simple CNN, and a simple greyscale dataset, to allow for quick and easy testing. We can scale up the network and dataset complexity in case of positive results, and as it is wise in the interest of energy efficiency to initially do a scaled down version of this research. Also, it is possible to achieve more than 99% test accuracy for MNIST with more complex networks, so a simpler network architecture will illuminate clearly any gains found from our method.

To find the most similar filters we need a measure for similarity between two filters. We found that three similarity metrics, namely Frobenius norm, pearson correlation coefficient, and cosine similarity have been used in the past to compare weights of fully connected networks or CNNs [2][3][4]. We use all of these methods as required in our context to compare if one method performs better than the others.

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

We have 10 convolutional filters in our model architecture in total, and after training our model separately on the 6 noisy datasets, we obtain, for each filter, its pairwise similarity across all the 6 models. For example, for filter 1, we obtain its pairwise similarity for the following pairs of models : { (1, 2), (1, 3), (1,4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6) }. We take the pair with the maximum similarity, and randomly select one filter from the pair. For our final warm-started model, which will be trained on the entire noiseless dataset with 60000 images, we initialize each convolutional filter with the corresponding selected filter. We believe that finding the pair with maximum similarity represents a scenario where almost the same filter has been learned for two datasets with different noises applied, which would imply that the filter is noise invariant. We also hypothesize that randomly selecting a filter from the pair with maximum similarity preserves the structural integrity of a filter, while averaging the two filters might result in a less robust filter. Note that we are not warm-starting biases, nor the parameters of the fully connected layers. The model summary is displayed below.

Model Summary

|        Layer (type)      |         Output Shape      |   Param #|
|--------------------------|---------------------------|--------- |
|            Conv2d-1      |     [-1, 2, 28, 28]       |      20  |
|              ReLU-2      |     [-1, 2, 28, 28]       |       0  |
|         MaxPool2d-3      |     [-1, 2, 14, 14]       |       0  |
|            Conv2d-4      |     [-1, 4, 14, 14]       |      76  |
|              ReLU-5      |     [-1, 4, 14, 14]       |       0  |
|         MaxPool2d-6      |     [-1, 4, 7, 7]         |       0  |
|            Linear-7      |     [-1, 16]              |   3,152  |
|              ReLU-8      |     [-1, 16]              |       0  |
|            Linear-9      |     [-1, 10]              |      170 |


## 3. Results and Discussion

We obtained the average similarity scores over 20 iterations for each of the 10 filters in the model across the six noisy datasets. These are the averages of the pairwise similarities for each filter across all the six datasets over 20 iterations. These are shown below in Table 1. 

### Table 1: Average Similarity Scores

| Metric                         | Filter 1   | Filter 2   | Filter 3   | Filter 4   | Filter 5   | Filter 6   | Filter 7   | Filter 8   | Filter 9   | Filter 10  |
|--------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| **Cosine similarity**          |            |            |            |            |            |            |            |            |            |            |
| Avg                            | 0.34358683 | 0.35845244 | 0.34136826 | 0.3577973  | 0.32331955 | 0.35565147 | 0.33031228 | 0.34108618 | 0.34629586 | 0.36081052 |
| Std                            | 0.25206524 | 0.2547853  | 0.2398826  | 0.23000914 | 0.23767488 | 0.245126   | 0.21687616 | 0.23442131 | 0.24160744 | 0.23463367 |
| **Pearson coefficient**        |            |            |            |            |            |            |            |            |            |            |
| Avg                            | 0.362284   | 0.36033085 | 0.32567469 | 0.33541251 | 0.33940847 | 0.32689811 | 0.34990946 | 0.34706977 | 0.32262877 | 0.33764659 |
| Std                            | 0.24923254 | 0.23640222 | 0.21519991 | 0.2344259  | 0.22949002 | 0.20990646 | 0.23824699 | 0.22018443 | 0.22329599 | 0.22861876 |
| **Frobenius norm**             |            |            |            |            |            |            |            |            |            |            |
| Avg                            | 1.8106298  | 1.8253623  | 1.3027258  | 1.2402626  | 1.2421654  | 1.2263647  | 1.2633992  | 1.3154866  | 1.2510338  | 1.3036119  |
| Std                            | 0.52752024 | 0.49355268 | 0.37172166 | 0.40557528 | 0.3969821  | 0.3711481  | 0.37935027 | 0.43620571 | 0.43121612 | 0.40887198 |


On average, each filter across the six noisy datasets is only somewhat similar, as can be seen in the results, meaning that it is hard to find a filter that is similar for all the noisy datasets. However, for a given filter, for certain pairs of noisy datasets, the similarity is more. For example, over all filters, the maximum similarity we found for any filter for any two datasets using the different similarity metrics were as follows : 

1) Cosine Similarity : 0.96
2) Pearson Correlation Coefficient : 0.97
3) Frobenius Norm : 0.39 (The lower the better)

Hence, there do exist pairs for a given filter across certain noisy datasets that are quite similar, which we believe can help us warm start our network. We believe that certain pairs of noises make it more conducive to learn similar noise-invariant filters than the others. For the 20 trials, we looked at the most commonly selected pair of models for maximum similarity for a filter using each similarity metric, across all the filters. This is shown in Table 2.

### Table 2: Most Commonly Selected pair

| Metric             | Pair of models | Mode Count |
|--------------------|----------------|------------|
| Cosine Similarity  | 1,5            | 33         |
| Pearson Coefficient| 2,5            | 25         |
| Frobenius Norm     | 3,4            | 30         |


Hence, using cosine similarity, across all filters, a filter was found to be most similar most often when comparing between models trained on the datasets with random perspective and elastic transformations applied to them, respectively. For Pearson correlation coefficient, the datasets were those corresponding to gaussian blur and elastic transform, and for frobenius norm, the datasets corresponded to gaussian noise and random erasing. We speculate that for datasets with perspective and elastic transformations applied to them, there is a set of noise-invariant filters that have high similarity, as looking at an image from a different perspective might result in the image appearing elastically strecthed or compressed. Hence, it might be possible to represent a perspective transformation as an elastic transform. Blurring an image in some instances might also give an appearance to the image of being stretched in certain directions. Gaussian noise and random erasing, were the more "extreme" forms of noises, so to say, which might have lead to similar filters being learned in their case. Why different similarity measures select different pairs of models more frequently, however, is an open question.

We have also included a comparison of the most similar filter and least similar filter found through each similarity measure. These are displayed in the heatmaps below. The first image shows two filters compared using the Cosine metric; one from a model trained on a dataset with gaussian blur and the other from a model trained on a dataset with additive gaussian noise. Qualitatively it appears that this filter is a horizontal edge detector. Edges would remain largely unchanged by these transformations and so are invariant, giving good performance on the test set. 
![WhatsApp Image 2024-06-14 at 14 15 43_ac876346](https://github.com/danieljwright/CV-project/assets/52325405/19b2b7ca-7122-41fe-96ed-c95524421624)

The second image shows the most dissimilar filters learned, measured with the Cosine metric. This compares two filters; one from a model trained on a dataset with the perspective transform and the other from a model trained with the random erase. These transforms are themselves dissimilar. 
![WhatsApp Image 2024-06-14 at 14 17 20_23409c29](https://github.com/danieljwright/CV-project/assets/52325405/f8d1c2fa-ff05-44cc-b5b0-af8c9b13ec78)

The third image shows the most similar filters learned, measured with the Pearson Correlation. Qualitatively, this appears to be a corner detector. The two filters were obtained from a model trained on the dataset with additive gaussian noise and a model trained on the dataset changed with the elastic transform. 
![WhatsApp Image 2024-06-14 at 14 17 58_5c0735df](https://github.com/danieljwright/CV-project/assets/52325405/b582490d-e8c1-42b9-8929-1abea8ca5c57)

The fourth image shows the most dissimilar filters learned, measured with the Person Correlation. This was obtained from a model trained on a dataset modified with the random erase, and a model trained on a dataset with the random rotate. 
![WhatsApp Image 2024-06-14 at 14 18 25_d17145f0](https://github.com/danieljwright/CV-project/assets/52325405/95ec259c-f9b2-4f85-9d27-9a8b673d608f)

The fifth image shows the most similar filters learned, measured with the Frobenius Norm. This was obtained from a model trained on a dataset with additive gaussian noise and a model trained on a dataset with random rotate applied. Qualitatively, this also appears to be an edge detector. 
![WhatsApp Image 2024-06-14 at 14 18 36_95c67a36](https://github.com/danieljwright/CV-project/assets/52325405/392618f0-6afc-4025-aa34-6bfaf44fb8b8)

The sixth image below shows the most dissimilar filters learned, measured iwth the Frobenius Norm. This was obtained from a model trained on a dataset with additive gaussian noise and a model trained on a dataset with random erase applied. 

![WhatsApp Image 2024-06-14 at 14 18 49_fadade7e](https://github.com/danieljwright/CV-project/assets/52325405/2a1378b1-8ce6-4931-9483-fed391914f76)

Qualitatively, it appears that the random erase leads to the most dissimilar filters being learned. We conjecture that this is because the CNN learns edge and corner detectors, which are features largely invariant to the other transformations applied, however they are not invariant to the erasure of the pixel values. Intuitively this corresponds with the inherent difficulty of performing object classification on an occluded object due to the relative lack of information. 


The average test accuracy of the warm-started model trained for each similarity metric and the average test accuracy of the randomly intialised model, are shown in Table 3.

### Table 3: Average Test Classification Accuracy

| Metric                          | Average Accuracy | Standard Deviation | Max Accuracy |
|---------------------------------|------------------|--------------------|--------------|
| Cosine Similarity               | 89.019           | 9.9472             | 95.68        |
| Pearson Coefficient             | 90.7775          | 7.0131             | 95.34        |
| Frobenius Norm                  | 86.713           | 20.3386            | 95.69        |
| Random Init                     | 94.316           | 1.0788             | 95.68        |



Hence, our method unfortunately does not manage to outperform random initialisation, no matter what similarity metric we choose to select similar filters. Random initialization, over many trials gives much less standard deviation over the test accuracies, and also a higher average test accuracy than our method. However, it must be noted that the maximum accuracy achieved by our method using cosine similarity and frobenius norm is the same or more than random initialization, respectively. However these gains are nor significant neither consistent. Random initialization consistently generalizes better than our method, and also is more robust in terms of range of test classification accuracies. We suspect that our method requires comparison between even more types of noisy datasets to find filters that are even more noise-invariant, and it is very hard to make multiple datasets while exhausting all types of possible noises. The fact that we ony warm-start the convolutional filters might have made our method unstable as well, and warm starting the biases and the parameters of the fully connected layers might be useful. We also do not compare each filter with every filter across all other models, comparing a filter only to its corresponding filter across the other models. Doing a more thorough comparison might result in a selection of more noise-invariant filters. It might also be possible that filters that learn the same thing might appear at different locations for the different models that we train over the six noisy datasets. It might be that our selection procedure for each of the ten filters for our CNN results in roughly the same filter being selected at different locations for warm starting, which can harm generalization. Also, even though our selected filters are noise-invariant, they might still be biased to perform slightly better on the noisy datasets they were trained on. Hence, in the current state in which our method is used, a random initialization of weights still results in better test classification accuracy. Although our method is different to what was done in [1] as mentioned before, we are still unable to achieve positive results.

For our method, using pearson correlation coefficient as the similarity metric resulted in highest average test classification accuracy, while frobenius norm resulted in the least consistent or robust classification test accuracies, having the maximum standard deviation of the test classification accuracies, and the least test classification accuracy. We can infer from this that although taking the frobenius norm of the difference of two filters is quite intuitive, simply computing elementwise distances between filters might not be a suitable metric for similarity between filters, as important information such as functional relationships betwwen filters is left out. Using cosine similarity was worse than using pearson correlation coefficient, although not as bad as using frobenius norm. This leads us to believe that simply flattening filters and comparing the resulting vetors' direction and orientation is not a sufficiently good metric of similarity between filters, and if cosine similarity is to be used, it needs to be done in a more suitable way. However, we do not at the moment know if there is a more suitable way. It might be possible to take the similarity row-wise or column wise, and take an average of all the similarities. However, we suspect that even doing that will not particularly increase performance. Surprisingly, pearson correlation coefficient, which also computes the similarity after flattening the filters, seems to be a more apt similarity measure for the filters. It leads to an interesting insight that linear relationships between filters might be a suitable measure of similarity between them, meaning that more than the magnitudes, the trend of increase or decrease in a filter's values, when viewed as a vector, can give a better understanding of what a filter is doing.



We also compared the chosen weights for the warm-started model (for each similarity metric used to decide similar weights for warm starting) and randomly initialized models with their respective final weights after training. The average similarity, using the pearson correlation coefficient, is shown in Table 4.

### Table 4: Change in warm-started weights

| Metric               | Average Similarity |
|----------------------|--------------------|
| Random Initialisation| 0.733              |
| Cosine Similarity    | 0.907              | 
| Pearson Coefficient  | 0.956              | 
| Frobenius Norm       | 0.931              | 


In [1], the authors found that there are instances where warm starting performs as well as random initialization, but in those cases, the weights by the end of training were not similar to the ones as the warm started weights at the beginning of training. Hence, these warm started models were essentially forgetting their warm started weights, making the warm starting procedure pointless. We agree with this observation through our results, as we observe that the reference average similarity between randomly initialised and converged convolutional filters is less than the average similarity between the warm started and converged convolutional filters for our method, using any similarity metric. Hence, our warm started filters are not "forgotten", which hurts classification performance on the test dataset. Using a different set of hyperparameters might have helped our method "forget" the warm started weights, but that makes the enitre excercise pointless.

## 4. Conclusion

We aimed to investigate if the generalization performance of a CNN could be improved over the performance random initialisation, using a novel warm starting method for training a CNN. Warm starting has been known to negatively impact the generalization ability of a network in the past. However, we believed that warm starting a CNN for a classification task based on noise-invariant trained convolutional filters, that are similar across training on different noisy datasets for the same classification task, could prove an effective method. We were able to effectively determine similar noise-invariant filters, and see that in some cases CNNs will converge to similar weights. Unfortunately, warm starting based on this method also proved less effective than random starting, with a worse average test accuracy and higher standard deviation. Hence, random initialisation remains an appropriate method to train networks.

## 5. References
[1] Ash, Jordan, and Ryan P. Adams. "On warm-starting neural network training." Advances in neural information processing systems 33 (2020): 3884-3894.

[2] Srinivas, Suraj, and R. Venkatesh Babu. “Data-free parameter pruning for deep neural networks.” arXiv preprint arXiv:1507.06149 (2015).

[3] Shang, Wenling, et al. “Understanding and improving convolutional neural networks via concatenated rectified linear units.” international conference on machine learning. 2016.

[4] Roy-Chowdhury, Aruni, et al. “Reducing duplicate filters in deep neural networks.” NIPS workshop on Deep Learning: Bridging Theory and Practice. Vol. 1. 2017.

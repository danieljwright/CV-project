# Warm Starting Convolutional Neural Networks and Filter Similarity
Authors: Daniel Wright (5932033) & Kunal Kaushik (6050549) & Alexandros Theocharous 5930901

In this blog post we present the results of our study into whether training a CNN on a dataset can be warm started using convolutional filters that are found to be similar after training the same CNN across different noisy subsets of the original dataset. This can potentially lead to better generalization accuracy than simply randomly initializing the filters for training. This work is part of the course CS4245 Computer Vision by Deep Learning in 2023/2024 at TU Delft. 

## 1. Introduction

As the applications of computer vision have grown and deep learning models have become more ubiqitous, the environmental impact of training these models becomes apparent. One solution to the water usage and carbon emissions of training models is to use warm-starting. This is the process of initalising model weights using weights from previously trained models, instead of random initialisation. Prior work has shown that warm starting neural networks can harm generalization, for example, in the case when weights learned from training on 50% of the data are used to initialize training on the entire dataset [1]. We explore warm starting in a different context to the methods utilized before. Our idea is to train a simple Convolutional Neural Network (CNN) on 6 different noisy subsets of the MNIST Dataset, which together comprise the entire dataset, which is a dataset of greyscale images of size 28*28 of the numbers 0-9. The final convolutional filters that we get from training the model on these noisy datasets are compared for similarities, and then we use filters that are similar across different models to warm-start the same Convolutional Neural Network on the entire MNIST dataset, to see if we can get better accuracy on the MNIST test dataset compared to simply training our Convolutional Neural Network with random initializtion on the entire MNIST dataset.

## 2. Methodology

We first shuffle the MNIST training dataset, which consists of 60000 images. We then take 6 different subsets of this shuffled dataset, with 10000 images in each subset, and apply a different noise to each of these subsets. These noises were chosen such that they are meaningful in a grey-scale image digit classification context. As such we have not used any color transformations.

1) The noise that we apply to the first subset is a random perspective transform on each input image, using torchvision.transforms.v2.RandomPerspective(). We set the probability of an image being transformed as 1, as we want all images to be transformed. The degree of distortion is set to 0.3, and the fill value is set to -1, so areas outside the transformed image are filled using the edge pixel value. An example of such a transformation is shown in the image below. This mimics a digit being angled away or towards the camera in an image.

2) For the second subset, we apply random gaussian blur to each image, using torchvision.transforms.v2.GaussianBlur(), using a gaussian kernel os size 5, and a standard deviation uniformly chosen from the range (1, 2) for creating the kernel that will do the blurring. This approximates an image being taken with a lower quality camera, or the image being taken at a distance.

3) For the third subset, for each image, we create a 28*28 array where all values are sampled from a Gaussian distribution with mean 0 and standard deviation 0.5, and we add this gaussian noise to the image. This is a catch-all transformation to mimic general noise that can occur in image. 

4) For the fourth subset, we randomly erase a rectangular region from the images with probability 0.5, using torchvision.transforms.v2.RandomErasing(), with an erasing value of -1. This approximates the digit being partially occluded.

5) For the fifth subset, we perform an elastic transform on the images, using torchvision.transforms.v2.ElasticTransform(), taking magnitude of displacements as 50. This approximates different handwriting styles.

6) Finally, for the sixth subset, we randomly rotate the images using torchvision.transforms.v2.RandomRotation() upto 30 degrees. This transform mimics the digit being rotated in an image, and also different handwriting styles to an extent. However, due to the rotational symmetry of certain digits, only upto 30 degrees was used.


![input](https://github.com/danieljwright/CV-project/assets/60939523/fa3ef1c3-cd13-4782-9b1f-44e2d41dcd6f)


We then define a simple Convolutional Neural Network, which will train on all of these subsets separately, and the entire training dataset (without any noise) as well. Our CNN has two convolutional layers, the first one having 1 input channel and 2 output channels, and the second one having 2 input and 4 output channels, both having kernel sizes of 3, and stride and padding of 1. Both convolutional layers are followed by max pooling layers using a kernel size of 2, stride of 2, and a padding of 0. These are followed by two fully connected linear layers, with the last layer having 10 outputs corresponding to the 10 classes. We train this network across all datasets using the Adam optimizer using a learning rate of 0.001, a batch size of 64, for 6 epochs, and the loss function that is optimized is the Cross Entropy Loss. We then check for similarities between the trained filters to warm start training on the entire noiseless MNIST dataset using the most similar filters. We also train the same network on the entire dataset using randomly initialized weights, and compare the classiciation accuracy on the unseen test dataset for both methods. Both of these training procedures are run for 1 epoch. We believe that using trained filters that are found to be similar across different noisy datasets are noise-invariant and if training on a new and larger dataset is warm started using these noise invariant filters, the training procedure can be guided in optimal directions, helping the network generalize better.

We have a deliberately chosen a very simple CNN, and a simple greyscale dataset, as we can scale up the network and dataset complexity in case of positive results with a simpler dataset and network, and as it is wise in the interest of energy efficiency to initially do a scaled down version of this research. Also, it is not so hard to achieve more than 99% test accuracy for MNIST with more complex networks, so we choose to use a simple network so we can observe if there are any gains using our similarity-based warm started network. 

To find the most similar filters we need a measure for similarity between two filters. We found that three similarity metrics, namely matrix norm, pearson correlation coefficient, and cosine similarity have been used in the past to compare weights of fully connected networks or CNNs. We use all of these methods as required in our context to compare if one method performs better than the others.

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

We have 8 convolutional filters in our model architecture in total, and after training our model separately on the 6 noisy datasets, we obtain, for each filter, its pairwise similarity across all the 6 models. For example, for filter 1, we obtain its pairwise similarity for the following pairs of models : { (1, 2), (1, 3), (1,4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6) }. We take the pair with the maximum similarity, and randomly select one filter from the pair. For our final warm-started model, which will be trained on the entire noiseless dataset with 60000 images, we initialize each convolutional filter with the corresponding selected filter. We believe that finding the pair with maximum similarity represents a scenario where almost the same filter has been learned for two datasets with different noises applied, which would imply that the filter is noise invariant. We also hypothesize that randomly selecting a filter from the pair with maximum similarity preserves the structural integrity of a filter, while averaging the two filters might result in a less robust filter. Note that we are not initializing biases, and we are only initializing convoulutional filters, not the parameters of the fully connected layers. 

## 3. Results

We obtained the average similarity scores over 20 iterations for each of the 10 filters in the model across the six noisy datasets. These are the averages of the pairwise similarities for each filter across all the six datasets over 20 iterations. These are shown below in Table 1. 

### Table 1: Average Similarity Scores

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

On average, each filter across the six noisy datasets is only somewhat similar, as can be seen in the results, meaning that it is hard to find a filter that is similar for all the noisy datasets. However, for a given filter, for certain pairs of noisy datasets, the similarity is more. For example, over all filters, the maximum similarity we found for any filter for any two datasets using the different similarity metrics was as follows : 

1) Cosine Similarity : 0.94
2) Pearson Correlation Coefficient : 0.95
3) Frobenius Norm : 0.34 (The lesser the better)

Hence, there do exist pairs for a given filter across certain noisy datasets that are quite similar, which we believe can help us warm start our network. We believe that certain pairs of noises make it more conducive to learn similar noise-invariant filters than the others. For the 20 trials, we looked at the most commonly selected pair of models for each similarity metric, across all the filters. This is shown in Table 2.

### Table 2: Most Commonly Selected pair

| Metric             | Pair of models | Mode Count |
|--------------------|----------------|------------|
| Cosine Similarity  | 1,5            | 33         |
| Pearson Coefficient| 2,5            | 25         |
| Frobenius Norm     | 3,4            | 30         |

Hence, using cosine similarity, across all filters, a filter was found to be most similar most often when comparing between models trained on the datasets with random perspective and elastic transformations applied to them, respectively. For Pearson correlation coefficient, the datasets were those corresponding to gaussian blur and elastic transform, and for frobenius norm, the datasets corresponded to gaussian noise and random erasing. We speculate that for datasets with perspective and elastic transformations applied to them, there is a set of noise-invariant filters that have high cosine similarity (which essentially means that the filters when flattened, represent similar vectors in terms of orientation and direction),

The model warm-started using the Pearson similarity metric performed the best, with an average test accuracy similar to that of the model that was randomly initalised. The average test accuracy of the warm-started model trained for each similarity metric and the average test accuracy of the randomly intialised model, are shown in Table 3.

### Table 3: Average Test Classification Accuracy

| Metric                          | Average Accuracy | Standard Deviation | Max Accuracy |
|---------------------------------|------------------|--------------------|--------------|
| Cosine Similarity               | 90.2295          | 8.3578             | 95.11        |
| Random Init                     | 94.057           | 1.6521             | 95.68        |
| Pearson Coefficient             | 91.8830          | 4.9802             | 95.29        |
| Frobenius Norm                  | 90.8165          | 3.4740             | 94.76        |

Hence, our method unfortunately does not manage to outperform random initialisation, no matter what similarity metric we choose to select similar filters. Random initialization, over many trials gives much less standard deviation over the test accuracies, a higher maximum test accuracy, and also a higher average test accuracy than our method. Hence, it consistently generalizes better than our method, and also is more robust in terms of range of test classification accuracies. We suspect that our method requires comparison between even more types of noisy datasets to find filters that are even more noise-invariant, and it is very hard to make multiple datasets while exhausting all types of possible noises. The fact that we ony warm-start the convolutional filters might have made our method unstable as well, and warm starting the biases and the parameters of the fully connected layer might be useful. We also do not compare each filter with every filter across all other models, comparing a filter only to its corresponding filter across the other models. Doing a more thorough comparison might result in a selection of more noise-invariant filters. It might also be possible that filters that learn the same thing might appear at different locations for the different models that we train over the six noisy datasets. It might be that our selection procedure for each of the ten filters for our CNN results in roughly the same filter being selected at different locations for warm starting, which can harm generalization. Also, even though our selected filters are noise-invariant, they might still be biased to perform slightly better on the noisy datasets they were trained on. Hence, in the current state in which our method is used, a random initialization of weights still results in better test classification accuracy. Although our method is different to what was done in [1] as mentioned before, we are still unable to achieve positive results.

For our method, using pearson correlation coefficient as the similarity metric resulted in highest average test classification accuracy, while frobenius norm resulted in the most consistent and robust classification test accuracies, having the least standard deviation of the test classification accuracies. Using cosine similarity was particularly bad, resulting in the most inconsistent and least accurate classifications. This leads us to believe that simply flattening filters and comparing the resulting vetors' direction and orientation is not a sufficiently good metric of similarity between filters, and if cosine similarity is to be used, it needs to be done in a more suitable way. However, we do not at the moment know if there is a more suitable way. It might be possible to take the similarity row-wise or column wise, and take an average of all the similarities. However, we suspect that even doing that will not particularly increase performance. Surprisingly, pearson correlation coefficient, which also computes the similarity after flattening the filters, seems to be a more apt similarity measure for the filters. It leads to an interesting insight that linear relationships between filters might be a suitable measure of similarity between them, meaning that more than the magnitudes, the trend of increase or decrease in a filter's values, when viewed as a vector, can give a better understanding of what a filter is doing. Also, simply taking the frobenius norm of the difference of two filters is quite intuitive, as it represents the average distance between the corresponding elements, and it results in the most consistent performance, but still, unfortunately, it is not close to random initialisation.


We also compared the chosen weights for the warm-started model with the final weights after the fine-tuning with the three similarity measures. The average similarity is shown in Table 4.

### Table 4: Change in warm-started weights

| Metric               | Average Similarity |
|----------------------|--------------------|
| Random Initialisation| 0.709              |
| Cosine Similarity    | 0.954              | 
| Pearson Coefficient  | 0.930              | 
| Frobenius Norm       | 0.929              | 


![WhatsApp Image 2024-06-14 at 14 15 43_ac876346](https://github.com/danieljwright/CV-project/assets/52325405/19b2b7ca-7122-41fe-96ed-c95524421624)
![WhatsApp Image 2024-06-14 at 14 17 20_23409c29](https://github.com/danieljwright/CV-project/assets/52325405/f8d1c2fa-ff05-44cc-b5b0-af8c9b13ec78)
![WhatsApp Image 2024-06-14 at 14 17 58_5c0735df](https://github.com/danieljwright/CV-project/assets/52325405/b582490d-e8c1-42b9-8929-1abea8ca5c57)
![WhatsApp Image 2024-06-14 at 14 18 25_d17145f0](https://github.com/danieljwright/CV-project/assets/52325405/95ec259c-f9b2-4f85-9d27-9a8b673d608f)
![WhatsApp Image 2024-06-14 at 14 18 36_95c67a36](https://github.com/danieljwright/CV-project/assets/52325405/392618f0-6afc-4025-aa34-6bfaf44fb8b8)
![WhatsApp Image 2024-06-14 at 14 18 49_fadade7e](https://github.com/danieljwright/CV-project/assets/52325405/2a1378b1-8ce6-4931-9483-fed391914f76)

## 4. Discussion

Do these six models trained on the noisy datasets converge to similar learned weights?

If we use these learned weights to warm-start a network on a dataset which includes all the different noises/artifacts does it perform better in terms of classification accuracy than random initialization?




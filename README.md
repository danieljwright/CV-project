# CNN Similarity and Warm-Starting Ability
Authors: Daniel Wright (5932033) & Kunal & Alex

In this blog post we present the results of our study into whether CNN's learn similar filters and if these learned weights can be used to warm start a model. This work is part of the course CS4245 Computer Vision by Deep Learning in 2023/2024 at TU Delft. 

## 1. Introduction

As the applications of computer vision have grown and deep learning models have become more ubiqutious, the environmental impact of training these models becomes apparent. One solution to the water usage and carbon emissions of training models is to use warm-starting. This is the process of initalising model weights using weights from previously trained models, instead of random initialisation. 

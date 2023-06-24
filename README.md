# DataFest-2023

DataFest@Integration2023 was a competition in ML conducted by ISI, Kolkata at an international level. It had two problem statements, second one being the bonus round. My fellow teammates and I competed amongst 17 teams and bagged the first position in the trio category. 

## Problem 1: Classification of Whale Sounds (Supervised Deep Learning Project)
The deep blue sea is home to a plethora of fascinating and mysterious creatures. Sound is the fundamental way these giants communicate with each other, but much of their communication remains a mystery to scientists. In this competition, we unlocked the secrets of blue whale communication by building model that can accurately identify various calls made by these creatures.

### Input
#### What are whale vocalisations?
The different types of vocalizations and their purposes are still not well understood, and ongoing research continues to uncover new insights into the vocal behaviour of these fascinating animals. Here we focus on a specific vocalization,

🐋 A Calls: They are characterized by a low-frequency, repetitive pattern of pulses that are typically around 70-90 Hz in frequency. These calls are known to be used for long-distance communication between individual whales and can be heard over large distances. They are typically produced by adult males and can last for several minutes.

Here is how the graph can be visualised:





![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/273696d0-e39f-4183-9f70-e58ae097abdb)

The dataset is annotated by the **DataFest Team at Indian Statistical Institute Kolkata**. The occurrences of A calls were identified and labelled from underwater audio recordings captured by deep-sea hydrophone devices. The recording spanned almost a month.

The number of samples per class:

No A calls (Labeled as 0): 12952

A calls (Labeled as 1): 12996

Unlabelled (test): 2000

Visualization of an audio file from the training dataset:
![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/da17d45e-3373-4a04-aa7c-5a1d838fcfd2)

### Dependency/ Library Used
os 

pathlib 

matplotlib

numpy

seaborn

tensorflow

keras

IPython

### Setting up and Importing the Data 
We used tensorflow's DatasetLoader to lazily load the dataset. The key characteristic of lazy loading is that the data is loaded incrementally as we iterate over the dataset. This approach is beneficial for large datasets that may not fit entirely into memory. By loading the data lazily, we can efficiently handle large datasets without running into memory limitations.

The number of samples included in each batch of the dataset is 64.
10% of the data has been reserved for validation, while the remaining 90% has been used for training. seed=0 ensures that the dataset splitting remains consistent across different runs. Each audio sample will be resized to have a length of 64,000 samples.

Therefore, there will be 365 batches. Each batch has shape ((64,64000),(64)) where the 1st element is a tensor of 64 audio tensors of length 64,000 and the 2nd ones are its labels.

### Visualisation and Inspection
The sequential audio data is transformed to image by Short Term Fourier Transform using tensorflow's stft().

#### What is a spectrogram?
A spectrogram is a visual representation of the spectrum of frequencies of a signal over time. It provides a way to analyze and visualize the frequency content of a time-varying signal. Spectrograms are widely used in various fields, including audio signal processing, speech recognition, music analysis, and acoustic research.

#### What is waveform?
A waveform represents the amplitude of a signal as a function of time. It is a one-dimensional representation, where time is plotted on the horizontal axis, and the signal's amplitude is plotted on the vertical axis. They are commonly used for visualizing and analyzing the temporal characteristics of signals.

The spectrograms of A and Non A calls:

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/5b0dc027-848f-404c-8abe-a60b28aea4f0)

The waveform of a Non A call:

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/c81aec21-18e6-47a2-a924-e53fae2b1192)

Improvised colored spectrogram for better visualisation:

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/2c5edcdd-570e-40d5-b556-2cb7fc859564)


### Model Construction and training
 A CNN model for audio classification is implemented. It preprocesses the spectrogram inputs by normalizing them, applies several convolutional layers for feature extraction, and ends with fully connected layers for classification.
The Adam optimizer is chosen, which is a popular optimization algorithm for deep learning. The SparseCategoricalCrossentropy loss function is used, indicating that the model's output is not one-hot encoded and comes from logits. The evaluation metric is set to 'accuracy', which calculates the accuracy of the model during training and evaluation. The model has been set to be trained for 15 epochs. Due to early stopping, the training has been stopped at the 8th epoch to prevent over-fitting. At this stage, the val_accuracy has reached 0.9977.

### Inspecting Model Accuracy and loss
The model has trained pretty well.

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/82880c74-8672-4ede-8601-8a64fb7c253c)

The confusion matrix is drawn to see the performance of the model and it does seem of some negative error prediction rate.

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/e8515a06-3634-441b-9e18-c21dde632c04)

False negatives and false positives were inspected manually. The F1 score was calculated which came out to be 0.9969.

## Problem 2: I'm Hard to Spot (Unsupervised Learning Project)
The Sundarbans mangrove forest is a place of dense forests, winding rivers, and an incredible diversity of plant and animal life. But there is one creature that calls the Sundarbans home that is more majestic and fearsome than any other: the Royal Bengal Tiger. These powerful predators are known for their distinctive orange coats, black stripes, and piercing eyes. But despite their fearsome reputation, the number of Royal Bengal Tigers in the Sundarbans has been declining rapidly in recent years. There are fewer than a few hundred tigers left in the Sundarbans. 

In this project we have implemented unsupervised learning algorithms to analyze the distinctive features of tiger pugmarks, and use that data to estimate the number of tigers living in the Sundarbans. 

### Input
The dataset is collected by the Forest Rangers at Sundarban and some part of it was collected by **Indian Statistical Institute Kolkata Team**. The pugmarks images were captured and later using the computerized technique explained in Data-Description.pdf they were converted to features.

The number of pugmarks:

Train.csv: 64

Census.csv (Unlabelled): 1059

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/278e35a8-1178-4c3d-8a0b-113d71588e81)

For detailed explanation of the features and techniques used to differentiate between pugmarks, you can go through these sources of information:)

[Gender_discrimination_of_tigers_by_using_their_pug.pdf](https://github.com/AkGu2002/DataFest-2023/files/11856002/Gender_discrimination_of_tigers_by_using_their_pug.pdf)

[Analytics_for_Crowd_Sourced_Data.pdf](https://github.com/AkGu2002/DataFest-2023/files/11856012/Analytics_for_Crowd_Sourced_Data.pdf)


### Dependency/ Library Used
cmath

kmeans

pandas

seaborn

scipy 

matplotlib 

TSNE 

PCA 

numpy 

### Algorithms Used
* Spectral Clustering:

Spectral clustering is a clustering algorithm that aims to partition data points into distinct clusters based on their similarity. Unlike traditional clustering algorithms like k-means, which operate on the original data space, spectral clustering leverages the concept of spectral graph theory to perform clustering in a transformed space. 

We have tried to split x_train into 1000 different clusters.

* PCA (Principal Component Analysis):

It is a popular dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional representation while retaining the most important information. It achieves this by identifying a set of new orthogonal variables, called principal components, that capture the maximum variance in the data.

We have applied PCA for dimensionality reduction on the x_train data, stored the transformed data in z, and then performed spectral clustering on z with 25 clusters. The code further retrieved the cluster labels assigned to the first 25 data points and calculated the number of unique clusters present in those assignments.

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/3cd5ad2e-f71a-4c44-98f4-9b7be63ece16)

* t-SNE (t-Distributed Stochastic Neighbor Embedding): 

It is a nonlinear algorithm that computes pairwise similarities between data points, constructs probability distributions, and optimizes an embedding in a lower-dimensional space to preserve local similarities. It employs stochastic optimization techniques.

We applied t-SNE for dimensionality reduction and visualization of the x_train data. We then performed spectral clustering on the transformed data and visualized the t-SNE projection with the assigned cluster labels. The final output describes the number of unique clusters obtained from spectral clustering.

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/50d85b37-8093-4e6b-86ec-f7b5d83f118e)


### Final Conclusion

We applied PCA for further reduced components i.e 15 and plotted the graph between
'cummulative explained variance' (y-axis) and 'number of components'(x-axis). It is observed that from 14 onwards 100% variance was being preserved.

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/d0b8749e-7c5c-4577-b848-64345f86a849)

A correlation plot of the features:

![App Screenshot](https://github.com/AkGu2002/DataFest-2023/assets/74046369/c530777f-25de-4d6c-b874-c1cad5e7c29b)









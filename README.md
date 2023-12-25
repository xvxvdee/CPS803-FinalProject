# CPS803-FinalProject

# Solving a Practical Clustering Problem: Exploring the Daily Kos Dataset

This project is a machine learning assignment from Toronto Metropolitan University. The goal is to apply the KMeans algorithm to cluster a bag of words dataset from the Daily Kos political blog.

## Data

The data consists of two files: the bag of words file in sparse format and the vocabulary. The repository's sample dataset contains 3420 documents, a vocabulary of 6906 terms, and 467,714 words. The creation of the vocabulary was based on the tokenization and elimination of stop words from each document. If the token occurred more than ten times, it was added to the vocab.

## Methods

The pipeline for preprocessing the data includes:

- Building each post by using the bag of words file
- Cleaning the content by replacing underscores, eliminating words with numbers, and stemming the vocabulary
- Vectorizing the text using the TF-IDF vectorizer
- Reducing the dimensionality using PCA

To cluster the bag of words, the KMeans algorithm was applied. To choose the optimal number of clusters, the Elbow method was used, which calculated the Sum of Squared Errors (SSE) for different values of k.

## Results

The optimal number of clusters was found to be four, based on the analysis of the SSE plot and the words in each cluster. The clusters were labeled as follows:

- Cluster 0: General politics and news
- Cluster 1: Iraq war and foreign policy
- Cluster 2: US elections and candidates
- Cluster 3: Bush administration and criticism

## Conclusions

The project demonstrated the use of KMeans to cluster a text-based dataset and the importance of considering other factors besides the Elbow method when choosing the number of clusters. The clusters showed some meaningful patterns and topics that reflected the nature of the Daily Kos blog.

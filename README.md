# PCA for Banana Quality Data Reduction

This project implements Principal Component Analysis (PCA) to reduce the dimensionality of the Banana Quality dataset from 8 features to 2. The goal is to capture the most important variance in the data while simplifying its complexity.

## Overview

Principal Component Analysis (PCA) is a statistical technique used for dimensionality reduction. It transforms a large set of variables into a smaller one while maintaining as much variance as possible. In this project, we use PCA to reduce the 8-feature Banana Quality dataset into 2 principal components.

## Features of the Dataset

The Banana Quality dataset includes the following 8 features:

1. **Size** (RGB values)
2. **Weight** (diameter, length)
3. **Sweetness**
4. **Softness**
5. **Harvest Time**
6. **Ripeness**
7. **Acidity**
8. **Quality** (Will Be Removed due to "string" data type)

## Program Details

This program uses Python with the following libraries:

- `numpy` for numerical computations
- `pandas` for data manipulation
- `matplotlib` for visualization
- `PCA` implementation (without `scikit-learn`)

The program follows these steps:

1. **Load the Dataset**: Reads the Banana Quality dataset from a CSV file.
2. **Remove 'Quality' Column**: Drops the 'Quality' column since it contains non-numeric data.
3. **Handle Missing Values**: Fills any missing data with the column's median.
4. **Convert to Numpy Array**: Converts the dataset to a NumPy array for processing.
5. **Visualize Original Data**: Displays a scatter plot using the first two features.
6. **Calculate Mean**: Computes the mean of each feature in the dataset.
7. **Calculate Zero Mean**: Subtracts the mean from the data to achieve zero mean.
8. **Calculate Covariance Matrix**: Computes the covariance matrix of the centered data.
9. **Find Eigenvalues and Eigenvectors**: Performs eigenvalue decomposition on the covariance matrix.
10. **Sort Eigenvalues and Eigenvectors**: Sorts them in descending order to select the most important components.
11. **Select Top 2 Components**: Chooses the top 2 eigenvectors for dimensionality reduction.
12. **Transform Data**: Projects the original data onto the new 2-dimensional space.
13. **Visualize Reduced Data**: Plots the reduced data in 2 dimensions.
14. **Convert Reduced Data to DataFrame**: Converts the reduced data back into a DataFrame for further use.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/JadiDapa/python-pca.git
   cd python-pca
   ```
2. Install the required Python packages:
   ```bash
   pip install numpy pandas matplotlib
   ```

## Usage

1. Make sure your dataset is in CSV format, with 8 features.
2. Run the PCA reduction main file
3. The program will output:
   - The transformed 2D data.
   - A scatter plot of the data points along the 2 principal components.

# Sources

1. Banana Quality Dataset : Kaggle (https://www.kaggle.com/datasets/l3llff/banana)

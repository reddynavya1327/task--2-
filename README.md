# Task 2: Exploratory Data Analysis (EDA) - Titanic Dataset

This project focuses on performing Exploratory Data Analysis (EDA) on the Titanic dataset to understand the data using statistics and visualizations.

In this task, the dataset is loaded using Python and Pandas from an online source. The structure of the dataset is explored using functions like `head()`, `info()`, and `isnull()` to understand the data types and identify missing values.

Summary statistics such as mean, median, and standard deviation are calculated using the `describe()` function. This helps in understanding the distribution and central tendency of numerical features.

Histograms are created for numerical columns to visualize the distribution of data. Boxplots are used to identify outliers and understand the spread of values in each feature.

Count plots are used to analyze categorical variables such as survival count, passenger class, and gender. These visualizations help in identifying patterns in the dataset.

A pairplot is generated to understand relationships between important features like Age, Fare, and Passenger Class with respect to survival.

A correlation matrix is plotted using a heatmap to identify relationships between numerical features. This helps in understanding how features are related to each other.

Skewness of numerical features is calculated to detect asymmetry in data distribution.

From the analysis, it is observed that missing values exist in some columns, the Fare feature contains outliers, and survival is influenced by factors like gender and passenger class.

This task helps in understanding the importance of data visualization, statistical analysis, and pattern recognition in machine learning.

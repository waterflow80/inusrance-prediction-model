# Inusrance Prediction Model - ML Classification Problem
In in this project we tried to create different machine learning supervised models for pedicting whether an owner of a given house should consider applying for insurance. The data preprocessing and visualization will be included in this project.

The machine learning techniques/models used are: **Decision Tree Classifier**, **Logistic Regression**, **Random Forest Classifier**, **SVM Classifier**, and **MLP Classifier**. 

## The Code
- The `main.ipynb` is the main entry point for the project.
- Other files contain useful code for formatting, normalizing, and preprocessing of the data.

## Data Visualization
Before jumping in and applying any classification algorithm, we should first understand and visualize the dataset. Here's some of the visualizations that we made for our dataset:
### Correlation between the features
![corr-matrix](https://github.com/user-attachments/assets/cc479a19-016f-47d2-9e29-9957de482374)

We can see that there are no highly correlated features in our dataset.

### Histograms and Distribution
#### Buidling Dimension
![building-dim-hist](https://github.com/user-attachments/assets/1ac52420-ca0f-4d3c-aa16-8923b1da6933)

#### Buidling Type
![building-type-hist](https://github.com/user-attachments/assets/6054669b-f76a-4778-8b56-0fe5dcbb18a3)


#### Number of Windows
![number-windows0hist](https://github.com/user-attachments/assets/07552631-39d3-437b-8b62-58307710384e)

#### Label (Class)
![claim-hist](https://github.com/user-attachments/assets/14f8a958-f400-471c-a3e8-32294f9b1872)

We can see that we are facing a problem of unbalanced data, so we should apply some oversampling techniques to avoid biased models.

## Data Preprocessing
After understanding the dataset and the different features, we can now apply some data preprocessing to prepare the data for the classifcation model. In this project we applied the following data preprocessing:
- **NaN values:** after careful study of the data, we removed some entries having NaN values, and replaces others with either **mean**, **previous val**, or **next value**.
- **Outliers:** we used the **Boxplot** method to determine outliers, and again made a study on whether to remove these outliers or replace them with other values.
- **Encoding:** we had to encode non-numeric values in order for the ML algrorithm to function correctly.
- **Normalization:** in order to make it easier for the ML algorithm to learn, we applied scaling techninques like **RobustScaler** to normalize the data.

## Classification Models
We applied different classification models and made some evaluation and comparisons to select the best model.
### Decision Tree Classifier
After training this classfier, we got the following results on the test data:
- **accuracy (in %):** 70.37727061015372
- **Confustion Matrix:**
  
  ![confusion-matrix](https://github.com/user-attachments/assets/6f206abf-15b0-4ea3-94c1-1e30747f5f1f)

### Logistic Regression Classifier
- **accuracy (in %):** 77.17745691662785
- **Confustion Matrix:**
  
![confusion-matrix-logis](https://github.com/user-attachments/assets/587f547d-2e4d-4680-966c-3e726dcfa563)

### Random Forest Classifier
- **accuracy (in %):** 71.1690731252911
- **Confustion Matrix:**

  ![confusion-matrix-rand-forest](https://github.com/user-attachments/assets/cb0d0ada-a6d0-43b5-bc87-cff409cdface)

### SVM Classifier
- **accuracy (in %):** 76.61853749417791
- **Confustion Matrix:**

![svm-confusion](https://github.com/user-attachments/assets/3fff44e9-2840-4647-83d5-01875a7ea9fe)

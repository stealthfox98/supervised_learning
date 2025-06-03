# Credit Card Fraud Detection using Machine Learning

A capstone project by Daniel Pérez Hernández., current Analyst at Deloitte and a Machine Learning enthusiast.

Most current update: April 03, 2023.

## Why doing a project of this nature?

Foe 6 months, Deloitte's AI Academy has provided me basic knowledge regarding the fascinanting world of data analysis and artifical intelligence. From basic Python functions, through SQL queries and Pandas data manipulation, to advanced mathematical models that employ any kind of data (structured, unestructured) to classify or predict behaviours previously unknown to us and that now are essential at any workplace where decision-making takes place.

This is a summary and application of all these knowledge and techniques with a clear purpose: extract and analyze data, generate information from it and build up  machine learning-powered models that can give solution to personal or business needs, leaning towards scalable automation.

## About the files used

Six files can be found within this repository:

1.  A PPTX file called `fraud_detection.pptx`.
2.  A PDF version of the pptx file, also called `fraud_detection.pdf`.
3.  A Jupyter Notebook called `fraud_detection.ipynb` where all the code has been written, aongside displayed plots.
4.  A PDF called `Capstone_Project_Proposal_Template.pdf` where the Capstone proposal is available.
5.  A `README.md` file where all of this information is displayed and ready for its lecture.
6.  A `.gitignore` file.

## About the dataset

The collected dataset consists of a CVS type file with transactions made by credit cards in September 2013 by European cardholders.

Data was obtained data through a direct download form the Kaggle website, result of an extensive search of machine learning datasets. The web link to this dataset is: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Once downloaded, the data from the downloaded file `creditcard.csv` will be obtained via Python `pandas` library and use the `read_csv()` function.

## Business understanding

Purchases made through websites or in situ stores represent an imminent danger to credit/debit cardholders throughout the world, given their exposure to cybernetic dangers that may generate unrecognized fraud transactions. Thus, it is important to generate a trustworthy AI model that detects potential threats based on transaction datasets collected on a daily-basis routine.

<img src="https://www.wealthandfinance-news.com/wp-content/uploads/2021/05/Banking-Fraud.jpg" width="800" height="400"/>

This applies particularly to the finance industry, with an emphasis on the fraud monitoring techniques that are implemented on a bigger scale at banks worldwide.

The main focus of this project is to demonstrate how a bank can detect frauds based on your daily financial activities.

## Knowing our data

This dataset consists of numerical input variables only. Most features (V1 to V28) are the result of PCA transformation of the original features due to confidentiality issues; the rest (‘Time’, ‘Amount’ and 'Class') haven’t been modified at all.

There are non-null values in this dataset, which means no method to delete or fill N/A values with mean (for numerical variables) or mode (for ctageorical variables) are necessary at all. 29 features consist of float-type data and only the 'Class' category is an integer, given the 1 or 0 values that identify if a transaction is an authentic fraud or not.

But from all characteristics, the most important are:

1.   There is a confirmed unbalanced phenomenon present in our data, and the affected category is non other that "Fraud"
2.   By observing the correlation matrix, it becomes evident than most of the features (V1 to V28) have basically a null correlation between them. However, there is a modest linear correlation between V1 to V28 variables and Time, Ammount and Class, though most of the correlation values are positive (not exceeding the 0.40 treshold) and the rest negative (not exceeding the -0.60 treshold).

## Data preparation

Once we know what kind of data we are dealing with, we will divide our dataset in features and label (X and y, respectively). This is used through the `train_test_split()` function obtained through `from sklearn.model_selection import train_test_split`.

It is important to keep in mind the first step to avoid data leakage after spliting our data: normalizing it using `StandardScaler()` through `from sklearn.preprocessing import StandardScaler`. 

## Treating imbalanced data

Having split and normalize the test and train sets, we need to deal with the imbalanced nature of our dataset. The SMOTE (Synthetic Minority Oversampling Technique) method will be used, since with it new instances are synthesized from preexisting data using k-nearest neighbor to select a random nearest neighbor, and a synthetic instance is created randomly in feature space. It is basicaly an oversampling method, meaning there will be an increase the minority class' row quantity.

<img src="https://miro.medium.com/v2/resize:fit:725/0*FeIp1t4uEcW5LmSM.png" width="1000" height="500"/>

According to the article Research on expansion and classification of imbalanced data based on SMOTE algorithm (Wang, et.al, 2021), the authors affirm "SMOTE algorithm can improve the classification effect of imbalanced data by randomly generating new minority sample points to increase the imbalance rate to a certain extent.". This is the reason why it will be applied to both our training and test sets.

The article can be consulted on the following URL link: https://www.nature.com/articles/s41598-021-03430-5, and the steps to import the `SMOTE()` function is:

*  `%pip install -U imbalanced-learn`
*  `from imblearn.over_sampling import SMOTE`

## The machine learning models

There are several ML problems that require classification solutions to distinguish desired categories, in this case if a transaction is a fraud or not. For this capstone project, two famous models were chosen: `LogisticRegression()` and `RandomForestClassifier()`, both functions called through `from sklearn.linear_model import LogisticRegression` and `from sklearn.ensemble import RandomForestClassifier`

## Metrics to compare performance

The authors recommend measuring accuracy through the Area Under the Precision-Recall Curve, given the unbalanced nature of the dataset. The rest of the traditional metrics will be incorporated (precision, F1-score, recall), as well as a confusion matrix to see how well the models classified data.

The following code snippets are used to call the metric functions from `sklearn`:

-  `from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score`
-  `from sklearn.metrics import roc_curve, roc_auc_score`
-  `from sklearn.metrics import confusion_matrix`

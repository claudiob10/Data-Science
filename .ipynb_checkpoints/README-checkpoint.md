# Data Science Services

The following document gives a detail view of the machine learning and data science processes that they will use in the following project

There is a description of the process, with a technical overview of the algorithms that were used, the primary goal of each one, and a diagram.

All the services has one version in a notebook and a diagram of how to do it in Azure Machine Learning Studio


## Client & Matter lookup 

This  Service allows us to check to misspell of Clients or Matters name. To achieve this, we compare the clients and matter list from the database and using the Levenshtein distance (1), which count the number of changes needed to go from String A to String B. In this process, it does not use any Machine Learning module. But we use a StringDist (2) Python module that allows to calculate that distance between strings. 
 
The levenshtein_norm is simply a normalized version that returns a float between 0 and 1. We use this output to iterate over the clients and matters list looking for the closest one.


In Azure Machine Learning Studio, there are two projects, one for Clients and other for Matters. The exact same algorithm may be used, but with different datasets.
 

## Cost Matter Prediction

This Service allows us to predict how many billable hours a matter will need. For now, the dataset we use to train the model has two categorical variables, “area of law” and “Activity code description”, to predict the outcome of billable hours. 
 
The data in CostPredictionDataset.csv is dummy data. After prepossessing the data, we split the information to train the model, 75% to train 25% to check the model. A Boosted Decision Tree Regression Model is used to make the prediction of billable hours. We obtained an accuracy of almost 96% with a mean absolute error of 0.73 hours. 

## Matter Recommender

The following Service recommend matters while the user works on a matter, he might be interested in looking at other matters that are related or relevant to that. 
 
First, we obtain the information from de database, we preprocess the information to avoid duplicate or null rows. Second, we select the columns of the information: 
•	MatterID
•	MatterCode
•	MatterName
•	MatterPhaseName
•	PracticeAreaName
•	CityName
•	MatterComments

When we have all the data ready, we filter the information first for the CityName and then for the PracticeAreaName, so we have information of similar matter. 
 
Finally, to find the top 5 matter recommendations, we use TF-IDF, short for term frequency–inverse document frequency to analysis the MatterComments , which is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. The TF-IDF value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general. 
 
What we want is to create a TF-IDF matrix, which has a row for each row of comments and a column for each term in the corpus. On each field, the matrix will have a weight that determines how important that term is to that comment. 

This can be achieved via scikit learn using CountVectorizer (3), TfidfTransformer (4) or TfidfVectorizer (5). After that, we defined a function that takes in a matter id, a TF-IDF matrix and a number n and finds the n closest rows (matters) using the cosine similarity distance.
 

## Document Classification
This machine learning service allows us to automatic classify documents in different types.
The data uses in this ML experiment is dummy data of 2004-2005 BBC news dataset. The dataset consists of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005. The news is classified into five classes as Business, Entertainment, Politics, Sports and Tech.
This ML web service demonstrates how to use multiclass classifiers and feature hashing in Azure ML Studio to classify news into appropriate categories.
The names of the categories have been used as the class label, or attribute to predict. The CSV file has uploaded to Azure ML Studio to use for the experiment.

Multiclass Neural Networks module with default parameters has been used for training the model. 

Results All accuracy values were computed using evaluate module. The confusion table is shown below.

## References:
1. https://en.wikipedia.org/wiki/Levenshtein_distance
2. https://pypi.org/project/StringDist/
3. http://scikit-learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage
4. http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
5. http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html










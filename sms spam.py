# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:48:42 2020

@author: Compaq
"""
import pandas as pd
messages=pd.read_csv('C:/Users/Compaq/Downloads/smsspamcollection/SMSSpamCollection',sep='\t',
                     names=['label','messages'])

messages['label'].value_counts()

#cleaning & preparocessing the dataset

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

Wordnet=WordNetLemmatizer()
corpus=[]

for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['messages'][i])
    review=review.lower()
    review=review.split()
    review=[Wordnet.lemmatize(word)for word in review if word not in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    
# Create TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


#Train-Test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=0)

#Model Building with naive Bayes' classifier-

from sklearn.naive_bayes import MultinomialNB
spam_detect_model= MultinomialNB().fit(X_train,y_train)
y_pred=spam_detect_model.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix 
cm=confusion_matrix(y_test,y_pred)

#Accuracy of the model
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)







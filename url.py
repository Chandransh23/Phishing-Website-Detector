#------Importing some useful libraries------

import pandas as pd          # use for data manipulation and analysis
import numpy as np           # use for multi-dimensional array and matrix

import seaborn as sns                # use for high-level interface for drawing attractive and informative statistical graphics 
import matplotlib.pyplot as plt      # It provides an object-oriented API for embedding plots into applications
#matplotlib inline 
# It sets the backend of matplotlib to the 'inline' backend:
import time                 # calculate time 

from sklearn.linear_model import LogisticRegression as lr                  # algo use to predict good or bad
from sklearn.naive_bayes import MultinomialNB                         # nlp algo use to predict good or bad

from sklearn.model_selection import train_test_split                 # spliting the data between feature and target
from sklearn.metrics import classification_report                    # gives whole report about metrics (e.g, recall,precision,f1_score,c_m)
from sklearn.metrics import confusion_matrix                         # gives info about actual and predict
from nltk.tokenize import RegexpTokenizer                            # regexp tokenizers use to split words from text  
from nltk.stem.snowball import SnowballStemmer                       # stemmes words
from sklearn.feature_extraction.text import CountVectorizer          # create sparse matrix of words using regexptokenizes  
from sklearn.pipeline import make_pipeline # use for combining all prerocessors techniuqes and algos

from PIL import Image                            # getting images in notebook
                                                 # from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator# creates words colud

from bs4 import BeautifulSoup                         # use for scraping the data from website
from selenium import webdriver                        # use for automation chrome 
import networkx as nx                     #for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

import pickle                    # use to dump model 

import warnings                  # ignores pink warnings 
warnings.filterwarnings('ignore')


phish_data = pd.read_csv('phishing_site_urls.csv')
phish_data.head()
phish_data.tail()
phish_data.info()
phish_data.isnull().sum()     # there is no missing values

#------Preprocessing ->Now that we have the data, we have to vectorize our URLs. I used CountVectorizer and gather words using tokenizer,since 
#there are words in urls that are more important than other words e.g ‘virus’, ‘.exe’ ,’.dat’ etc. Lets convert the URLs into a vector form
#RegexpTokenizer->A tokenizer that splits a string using a regular expression, which matches either the tokens or the separators between tokens---
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
phish_data.URL[0]

# this will be pull letter which matches to expression
tokenizer.tokenize(phish_data.URL[0])     # using first row


print('Getting words tokenized ...')
t0= time.perf_counter()
phish_data['text_tokenized'] = phish_data.URL.map(lambda t: tokenizer.tokenize(t)) # doing with all rows
t1 = time.perf_counter() - t0
print('Time taken',t1 ,'sec')


#phish_data.sample(5)

#------SnowballStemmer-> Snowball is a small string processing language, gives root words------
stemmer = SnowballStemmer("english")                          # choose a language
print('Getting words stemmed ...')
t0= time.perf_counter()
phish_data['text_stemmed'] = phish_data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')

#phish_data.sample(5)

print('Getting joiningwords ...')
t0= time.perf_counter()
phish_data['text_sent'] = phish_data['text_stemmed'].map(lambda l: ' '.join(l))
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')

#phish_data.sample(5)


#------Visualization-> 1. Visualize some important keys using word cloud------
#sliceing classes
bad_sites = phish_data[phish_data.Label == 'bad']
good_sites = phish_data[phish_data.Label == 'good']
#bad_sites.head()
#good_sites.head()

#-----Creating Model-------
#CountVectorizer->is used to transform a corpora of text to a vector of term / token counts.
#create cv object
cv = CountVectorizer()
#help(CountVectorizer())
feature = cv.fit_transform(phish_data.text_sent)                #transform all text which we tokenize and stemed
feature[:5].toarray()                                           #convert sparse matrix into array to print transformed features

#Spliting the data
trainX, testX, trainY, testY = train_test_split(feature, phish_data.Label)

lr.score(testX, testY)

#Logistic Regression is giving 96% accuracy, Now we will store scores in dict to see which model perform best
Scores_ml = {}
Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)


print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


#------MultinomialNB------
#Applying Multinomial Naive Bayes to NLP Problems. Naive Bayes Classifier Algorithm is a family of probabilistic algorithms based on applying
#Bayes' theorem with the “naive” assumption of conditional independence between every pair of a feature.
# create mnb object
mnb = MultinomialNB()
mnb.fit(trainX,trainY)


mnb.score(testX,testY)

Scores_ml['MultinomialNB'] = np.round(mnb.score(testX,testY),2)


print('Training Accuracy :',mnb.score(trainX,trainY))
print('Testing Accuracy :',mnb.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(mnb.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(mnb.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


acc = pd.DataFrame.from_dict(Scores_ml,orient = 'index',columns=['Accuracy'])
sns.set_style('darkgrid')
sns.barplot(acc.index,acc.Accuracy)


#------So, Logistic Regression is the best fit model, Now we make sklearn pipeline using Logistic Regression------

pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), lr())
##(r'\b(?:http|ftp)s?://\S*\w|\w+|[^\w\s]+') ([a-zA-Z]+)([0-9]+) -- these tolenizers giving me low accuray 


trainX, testX, trainY, testY = train_test_split(phish_data.URL, phish_data.Label)

pipeline_ls.fit(trainX,trainY)

pipeline_ls.score(testX,testY) 

print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")

pickle.dump(pipeline_ls,open('phishing.pkl','wb'))

loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.score(testX,testY)
print(result)

predict_bad = ['yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php','fazan-pacir.rs/temp/libraries/ipad','tubemoviez.exe','svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt']
predict_good = ['youtube.com/','youtube.com/watch?v=qI0TQJI3vdU','retailhellunderground.com/','restorevisioncenters.com/html/technology.html']
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
#predict_bad = vectorizers.transform(predict_bad)
# predict_good = vectorizer.transform(predict_good)
result = loaded_model.predict(predict_bad)
result2 = loaded_model.predict(predict_good)
print(result)
print("*"*30)
print(result2)








import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#%%
df = pd.read_csv("Spam.csv", encoding="unicode_escape")

#%% checking the first 5 rows and the shape of the df
print(df.head())
print(df.shape)

#%% check the number of unique values in the df to identify if there are duplicates
print(df.nunique())

# there are duplicate entries since shape = 5572 but unique values = 5168

#%% drop the duplicates 
df.drop_duplicates(inplace = True)

#%% reset the index
df = df.reset_index(drop = True)
print(df.shape)
print(df.head())

#%% check number of legit and spam messages
classes = df['Class']
print(classes.value_counts())

# there are more legit messages than there are spam messages

#%% word cloud for spam messages
spam_text = (df[df['Class'] == 'spam'] ['Text'])
print(spam_text)

# creating the word cloud
from wordcloud import WordCloud
spam_wc = WordCloud(width=800, height=400, colormap="RdYlBu").generate(" ".join(spam_text))
plt.figure(figsize=(10,10), facecolor="black")
plt.imshow(spam_wc)
plt.axis("off")
plt.tight_layout(pad=0.4)
plt.show()
  
#%% word cloud for legit messages
legit_text = (df[df['Class'] == 'legit'] ['Text'])
print(legit_text)

# creating the word cloud
from wordcloud import WordCloud
spam_wc = WordCloud(width=800, height=400, colormap="coolwarm").generate(" ".join(legit_text))
plt.figure(figsize=(10,10), facecolor="black")
plt.imshow(spam_wc)
plt.axis("off")
plt.tight_layout(pad=0.4)
plt.show()

#%% binary conversion: convert 'legit' and 'spam' to 0 and 1, respectively. 
def binary(row):
    if row['Class'] == 'legit':
        return 0
    else:
        return 1

df['spam'] = df.apply(binary, axis=1)
df = df.drop(['Class'], axis=1)

#%% TEXT PREPROCESSING
# remove https, phone numbers, email addresses, currency signs, punctuations
    # remove email and phone number to remove the possibility of a correlation between spam and a specific email/phone numbers
# numbers, white spaces, and leading and tailing white spaces
# fix the dataset such that Word = word, word = words

#%% assign the text column to a variable
texts = df['Text']
print(texts)

#%%
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

#%% using http://regexlib.com/Search.aspx to replace email addresses
# and numbers with regular expressions

import re

# create a class
class CleanText:
    def __init__(self, language):
        self.Lemmatizer = WordNetLemmatizer()
        self.stop_words = stopwords.words(language)
    
    def change_emails(self, texts):
        return re.sub('^\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,3}$', 'emailaddr', texts)

# this function uses text as an argument
    
    def change_urls(self, texts):
        return re.sub('^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddr', texts)
    
    def change_money(self, texts):
        return re.sub('(\$\d)|(\€\d)|(\£\d)', 'money', texts)
    
    def change_phone(self, texts):
        return re.sub('^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phone', texts)

    def change_number(self, texts):
        return re.sub('^[-+]?\d+(\.\d+)?$', 'number', texts)
    
    def delete_punctuation(self, texts):
        return re.sub('[^\w\d\s]', ' ', texts.lower())

    def clean_text(self, texts, keep_emails = False, keep_urls = False, keep_money = False, keep_phone = False, keep_number = False):
        texts = texts
        if keep_emails == False:
            texts = self.change_emails(texts)
        
        if keep_urls == False:
            texts = self.change_urls(texts)
            
        if keep_money == False:
            texts = self.change_money(texts)
            
        if keep_phone == False:
            texts = self.change_phone(texts)
        
        if keep_number == False:
            texts = self.change_number(texts)
        
        texts = self.delete_punctuation(texts)
        texts = ' '.join(
            self.Lemmatizer.lemmatize(term)
            for term in texts.split()
            if term not in self.stop_words
            )
        return texts
    
    def clean_data(self, data, keep_emails = False, keep_urls = False, keep_money = False, keep_phone = False, keep_number = False):
        assert(type(data) == list)
        processed_data =  []
        for i in data:
            processed_data.append(self.clean_text(i, keep_emails = keep_emails, keep_urls = keep_emails, keep_money = keep_emails, keep_phone = keep_emails, keep_number = keep_emails))
        return processed_data

#%%
print(texts.head())    

#%% clean the text
cltext = CleanText(language = 'english')
texts = pd.Series(cltext.clean_data(list(texts)))
print(texts.head())

#%% data tokenization - tokenization is needed for the machine to process text data
# https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,2))
vectors = tfidf_vectorizer.fit_transform(texts)

# check the shape of the final data
print(vectors.shape)

#%% modelling
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#%% split the dataset into train and test (80/20) and set the x and y variables
X = vectors 
y = df['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2022)

#%% finding the  best parameters for SVM
from sklearn.svm import SVC
def svm_model(params):
    classifier_ = SVC(**params)
    return cross_val_score(classifier_, X_train, y_train).mean()

model_space = {
    'C': hp.uniform('C', 1,3),
    'kernel': hp.choice('kernel', ["linear", "poly", "rbf", "sigmoid"]),
    'gamma': hp.choice('gamma', ["scale", "auto"])}

best_svm = 0
best_params = []
def f(params):
    global best_svm
    acc = svm_model(params)
    if acc > best_svm:
        best_svm = acc
        best_params.append(params)
    print('The accuracy is {} and the best params are {}'.format(round(acc,3), params))
    return -acc

trials = Trials()
best_svm = fmin(f, 
                model_space, 
                algo=tpe.suggest, 
                max_evals=50, 
                trials=trials)
#%% fit the SVM model on the train dataset
from sklearn.svm import SVC
best_params = best_params[-1]
SVM = SVC(**best_params)
SVM1 = SVM.fit(X_train, y_train)

# confusion matrix and AUC score for the SVM model
SVM_pred = SVM1.predict(X_test)
print(confusion_matrix(y_test, SVM_pred))
print("Test AUC score for SVM is", roc_auc_score(y_test, SVM_pred))
print(classification_report(y_test, SVM_pred))

#%% finding the best parameters for random forest
from sklearn.ensemble import RandomForestClassifier
def acc_model(params):
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, X_train, y_train).mean()

param_space = {
    'max_depth': hp.choice('max_depth', range(100,200)),
    'max_features': hp.choice('max_features', range(100,200)),
    'n_estimators': hp.choice('n_estimators', range(250,350)),
    'criterion': hp.choice('criterion', ["gini", "entropy"])
}

best_rf = 0
best_params_rf = []
def f(params):
    global best_rf
    acc = acc_model(params)
    if acc > best_rf:
      best_rf = acc
      best_params_rf.append(params)
    print('The accuracy is {} and the best params are {}'.format(round(acc,3), params))
    return -acc

trials = Trials()
best_rf = fmin(f, param_space, algo=tpe.suggest, max_evals=50, trials=trials)

#%% fitting the random forest model to the train data
from sklearn.ensemble import RandomForestClassifier
best_params_rf = best_params_rf[-1]
RandomForestClassifier = RandomForestClassifier(**best_params_rf)
RandomForestClassifier1 = RandomForestClassifier.fit(X_train, y_train)
# Make the confustion matrix and calculate AUC score
RF_pred = RandomForestClassifier1.predict(X_test)
print(confusion_matrix(y_test, RF_pred))
print("Test AUC score for RF is", roc_auc_score(y_test, RF_pred))
print(classification_report(y_test, RF_pred))
#%% finding the best parameters for a logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=2022)

#%% fitting the lr model on the train dataset
lrfit = lr.fit(X_train, y_train)

# confusion matrix and AUC curve for the lr model
lr_pred = lr.predict(X_test)
print(confusion_matrix(y_test, lr_pred))
print("Test AUC score for Logistic Regression is", roc_auc_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

#%% finding the best parameters for a naive-bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)

# confusion matrix and AUC curve for the lr model
nb_pred = nb.predict(X_test)
print(confusion_matrix(y_test, nb_pred))
print("Test AUC score for Naive-Bayes is", roc_auc_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

#%% graph the AOC curve for all the models
svm_fpr, svm_tpr, _ = roc_curve(y_test, SVM_pred)
rf_fpr, rf_tpr, _ = roc_curve(y_test, RF_pred)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_pred)
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_pred)

# make the plot
plt.figure(figsize=(8,8))
plt.plot([0,1],[0,1], 'k--')
plt.plot(svm_fpr, svm_tpr, label= "SVM")
plt.plot(rf_fpr, rf_tpr, label= "RF")
plt.plot(lr_fpr, lr_tpr, label= "LR")
plt.plot(nb_fpr, nb_tpr, label= "NB")
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title('Receiver Operating Characteristic')
plt.show()

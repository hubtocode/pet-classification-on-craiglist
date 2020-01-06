#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import spacy 
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas_profiling
nlp = spacy.load('en_core_web_sm')
nlp = en_core_web_sm.load()
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# In[5]:


## Load excel file into dataset
ca = pd.read_excel('classified_mixed.xlsx',encoding='latin-1')
cg = pd.read_csv('Craiglist_labelled_20.csv',encoding='latin-1')


# In[6]:


### Splitting ClassifiedAds.com datasets into test and train with ratio of 80-20 
X_train, X_test, y_train, y_test = train_test_split(ca['review'], ca['label'],stratify=ca['label'],test_size=0.20,random_state=42 )
training_x,testing_x,training_c,testing_c = X_train, X_test, y_train, y_test

### Splitting craigslist.com datasets into test and train with ratio of 80-20 
X_test_cg = cg['review']
y_test_cg = cg['label']


# In[ ]:





# In[7]:


# Tokenization, removal of stopwords, punctuations and transformation for preprocessing
def tk(doc):
    return doc

vec= TfidfVectorizer(analyzer='word',tokenizer=tk, preprocessor=tk,token_pattern=None, min_df=2, ngram_range=(1,4), stop_words='english') 
vec.fit(training_x)
training_x= vec.transform(training_x)
testing_x= vec.transform(testing_x)

#Fitting CG dataset
testing_x_cg= vec.transform(X_test_cg)


# In[9]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# training
lr.fit(training_x,training_c)
y_pred_lr= lr.predict(testing_x)

# evaluation
y_pred_lr = lr.predict(testing_x_cg)
acc_cg_lr = accuracy_score(y_test_cg, y_pred_lr) 
print("LR model Accuracy: {:.2f}%".format(acc_cg_lr*100))


from sklearn.linear_model import SGDClassifier
SCD = SGDClassifier(loss='modified_huber',shuffle=True,random_state=101)

# training
SCD.fit(training_x,training_c)
y_pred_SCD= SCD.predict(testing_x)

# evaluation
y_pred_SCD = SCD.predict(testing_x_cg)
acc_cg_SCD = accuracy_score(y_test_cg, y_pred_SCD) 
print("SCD model Accuracy: {:.2f}%".format(acc_cg_SCD*100))


knn = KNeighborsClassifier(n_neighbors=3)

# training

knn.fit(training_x,training_c)
y_pred_knn= knn.predict(testing_x)

# evaluation
y_pred_knn = knn.predict(testing_x_cg)
acc_cg_knn = accuracy_score(y_test_cg, y_pred_knn) 
print("KNN model Accuracy: {:.2f}%".format(acc_cg_knn*100))


# LinearSVC
SVMmodel = LinearSVC(C=.8)

# training 
SVMmodel.fit(training_x, training_c)
y_pred_SVM = SVMmodel.predict(testing_x)

# evaluation 
acc_SVM = accuracy_score(testing_c, y_pred_SVM) 
print("SVM model Accuracy: {:.2f}%".format(acc_SVM*100))


GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01,max_depth=2, random_state=0)

# training

GBC.fit(training_x,training_c)
y_pred_GBC= GBC.predict(testing_x)

# evaluation
y_pred_GBC = GBC.predict(testing_x_cg)
acc_cg_GBC = accuracy_score(y_test_cg, y_pred_GBC) 
print("GBC model Accuracy: {:.2f}%".format(acc_cg_GBC*100))


# training
DTmodel = DecisionTreeClassifier() 
DTmodel.fit(training_x, training_c) 
y_pred_DT = DTmodel.predict(testing_x)
# evaluation 
acc_DT = accuracy_score(testing_c, y_pred_DT) 
print("Decision Tree Model Accuracy: {:.2f}%".format(acc_DT*100))

RFmodel = RandomForestClassifier(n_estimators=250, max_depth=3, bootstrap=True, random_state=42)
RFmodel.fit(training_x, training_c) 
y_pred_RF = RFmodel.predict(testing_x)

acc_RF = accuracy_score(testing_c, y_pred_RF) 
print("Random Forest Model Accuracy: {:.2f}%".format(acc_RF*100))


# In[10]:


df = pd.read_csv('Craiglist_unlabelled_80.csv',encoding='latin-1')
ads = df["review"]
ads_t = df["title"]

lemmatizer = WordNetLemmatizer()
fin = []
# ads = df["lemma"]
entities = []
for ad in ads:
    if ad:
        tk_ = nltk.word_tokenize(ad.lower()) #tokenize
        lm_ = [lemmatizer.lemmatize(token) for token in tk_] 
        sw_ = [token for token in lm_ if not token in stopwords.words('english')] # stopword removal
        sn_ = " ".join(sw_) #joinback sentence
        fin.append(sn_)


# In[11]:


row_num,money_val = [],[]
for i,val in enumerate(fin):
    doc = nlp(val)
    for ent in doc.ents:
        if ent.label_=='MONEY' and ent.text.isnumeric() and float(ent.text)>0:
            row_num.append(i)
            money_val.append(ent.text)


# In[12]:


fin[:2]


# In[13]:


cost_entity = pd.DataFrame(list(zip(row_num,money_val)),columns=["Row","Price"])
cost_entity_transform = cost_entity.drop_duplicates(subset="Row").reset_index().drop(columns="index")


# In[14]:


# cg2 = pd.read_csv('Craiglist_unlabelled_80.csv',encoding='latin-1')
cg2 = df.reset_index()


# In[15]:


cg2.head()


# In[16]:


cost_entity_transform.head()


# In[17]:


cost_entity_cg2_merged = cg2.merge(cost_entity_transform, left_on='index', right_on='Row')


# In[18]:


cost_entity_cg2_merged.drop(columns=["index","Row"])


# In[19]:


###final model prediction
final_prediction_transform = vec.transform(df['review'])
final_prediction = SVMmodel.predict(final_prediction_transform)


# In[21]:


import matplotlib.pyplot as plt
plt.hist(final_prediction)
plt.show()


# In[22]:


final_prediction


# In[ ]:


cost_entity_cg2_merged.profile_report()


# In[23]:


## Deep Neural Network 
from sklearn.neural_network import MLPClassifier
DLmodel = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(8,5,8), random_state=42)

# training
DLmodel.fit(training_x, training_c)
y_pred_DL= DLmodel.predict(testing_x)

# evaluation
acc_DL = accuracy_score(testing_c, y_pred_DL)
print("DL model Accuracy: {:.2f}%".format(acc_DL*100))


# In[29]:


plt.plot(cost_entity_cg2_merged.Price)


# In[ ]:





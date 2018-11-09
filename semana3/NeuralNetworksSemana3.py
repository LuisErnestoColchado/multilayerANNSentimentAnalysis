#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression for Sentiment Analysis

# Adapted from http://nbviewer.jupyter.org/github/rasbt/pattern_classification/blob/master/machine_learning/scikit-learn/outofcore_modelpersistence.ipynb

# ## The IMDb Movie Review Dataset

# In this section, we will train a simple logistic regression model to classify movie reviews from the 50k IMDb review dataset that has been collected by Maas et. al.
# 
# > AL Maas, RE Daly, PT Pham, D Huang, AY Ng, and C Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Lin- guistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics
# 
# [Source: http://ai.stanford.edu/~amaas/data/sentiment/]
# 
# The dataset consists of 50,000 movie reviews from the original "train" and "test" subdirectories. The class labels are binary (1=positive and 0=negative) and contain 25,000 positive and 25,000 negative movie reviews, respectively.
# For simplicity, I assembled the reviews in a single CSV file.
# 

# In[20]:


import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import warnings
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')


df = pd.read_csv('/home/luisernesto/Documentos/MasterComputerScienceII/IntelligentSystem/semana3/shuffled_movie_data.csv')
df.tail()


# Let us shuffle the class labels.

# ## Preprocessing Text Data

# Now, let us define a simple `tokenizer` that splits the text into individual word tokens. Furthermore, we will use some simple regular expression to remove HTML markup and all non-letter characters but "emoticons," convert the text to lower case, remove stopwords, and apply the Porter stemming algorithm to convert the words into their root form.

# In[21]:


import numpy as np
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

stop = stopwords.words('english')
porter = PorterStemmer()

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    text = [w for w in text.split() if w not in stop]
    tokenized = [porter.stem(w) for w in text]
    return text


# Let's give it at try:

# In[22]:


tokenizer('This :) is a <a> test! :-)</br>')


# ## Learning (SciKit)

# First, we define a generator that returns the document body and the corresponding class label:

# In[23]:


def stream_docs(path):
    with open(path, 'r') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


# To conform that the `stream_docs` function fetches the documents as intended, let us execute the following code snippet before we implement the `get_minibatch` function:

# In[24]:


next(stream_docs(path='/home/luisernesto/Documentos/MasterComputerScienceII/IntelligentSystem/semana3/shuffled_movie_data.csv'))


# After we confirmed that our `stream_docs` functions works, we will now implement a `get_minibatch` function to fetch a specified number (`size`) of documents:

# In[25]:


def get_minibatch(doc_stream, size):
    docs, y = [], []
    for _ in range(size):
        text, label = next(doc_stream)
        docs.append(text)
        y.append(label)
    return docs, y


# Next, we will make use of the "hashing trick" through scikit-learns [HashingVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html) to create a bag-of-words model of our documents. Details of the bag-of-words model for document classification can be found at  [Naive Bayes and Text Classification I - Introduction and Theory](http://arxiv.org/abs/1410.5329).

# In[26]:


from sklearn.feature_extraction.text import HashingVectorizer
vect = HashingVectorizer(decode_error='ignore', 
                         n_features=2**21,
                         preprocessor=None, 
                         tokenizer=tokenizer)

# Exercise 1: define features based on word embeddings (pre-trained word2vec / Glove/Fastext emebddings can be used)
# Define suitable d dimension, and sequence length

reviewList = []
wordList = []
countRows = 0
model = np.empty((50000,101))
model[:,:] = 0
mean = np.empty((1,100))

for text in df.loc[:,'review']:
    countColumns=0
    wordList = tokenizer(text)
    reviewList.append(wordList)
    W2V = Word2Vec(wordList,size=100,min_count=1,workers=10)
    W2V.train(wordList,total_examples=len(wordList),epochs=10)
    X = W2V[W2V.wv.vocab]
    #for x in range(0,X.shape[0]):
    #pca = PCA(n_components=1)
    #Xpca = pca.fit_transform(X)
    mean[0,:]  = X.mean(0)
    #Xpca.reshape((Xpca.shape[1],Xpca.shape[0]))
    #    mean[0,:] += X[x,:]
    #pca = PCA(n_components=100)
    #Xpca = pca.fit_transform(X)
    #countColumns = 0
    model[countRows,0:100] = mean[0,:]
    #for i in range (0,Xpca.shape[0]):
    #    model[countRows,i] = Xpca[i,0] 
    #for x in Xpca:
    #    model[countRows,countColumns] = x
    #    countColumns+=1
    model[countRows,100:101] = df.loc[countRows,'sentiment']
    if (countRows % 5000 == 0) or (countRows == 0):
        print("charging: ",countRows/500,"%") 
    elif (countRows == 49999):
        print("charging:  100%");
    countRows+=1


# In[30]:

model.shape


# In[31]:
xTrain = model[0:40000,0:100] * 1000
yTrain = model[0:40000,100:101]
xTest = model[40000:50000,0:100] * 1000
yTest = model[40000:50000,100:101]
print(xTrain.shape)
print(yTrain.shape)
xTrain[0:20,:]

# In[ ]:
# Using the [SGDClassifier]() from scikit-learn, we will can instanciate a logistic regression classifier that learns from the documents incrementally using stochastic gradient descent. 
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='shuffled_movie_data.csv')
# Exercise 2: Define at least a Three layer neural network. Define its structure (number of hidden neurons, etc)
# Define a nonlinear function for hidden layers.
# Define a suitable loss function for binary classification
# Implement the backpropagation algorithm for this structure
# Do not use Keras / Tensorflow /PyTorch etc. libraries
# Train the model using SGD
numberCycle = 5000
numberHiddenLayer = 3
numberNeuron = 3
numberLayer = numberHiddenLayer + 1


#alfa[0,0] = np.power((1+np.power(5,0.5))/2,-1)
b = 0.01 * np.random.rand(numberNeuron,numberHiddenLayer)
#b[:,:] = 1
bOut = 0.01 * np.random.rand(1,1)
#bOut[:,:] = 1

lamba = np.random.rand(1,1)
lamba[0,0] = np.power(10.0,-3.0)
wInput = 0.01 * np.random.rand(numberNeuron,100) 
wHidden1 = 0.01 * np.random.rand(numberNeuron,numberNeuron) 
wHidden2 = 0.01 * np.random.rand(numberNeuron,numberNeuron)
wOutput =  0.01 * np.random.rand(numberNeuron,1) 

wInput[:,:] = np.random.rand()
#wHidden[:,:] = 0.1*np.random.rand()
wOutput[:,:] = np.random.rand()

derivateInput = np.empty((numberNeuron,100))
derivate1 = np.empty((numberNeuron,numberNeuron))
derivate2 = np.empty((numberNeuron,numberNeuron))
derivate = 0

wCurrent = np.zeros((numberNeuron,numberNeuron))
yHidden1 = np.zeros((numberNeuron,1))
yHidden2 = np.zeros((numberNeuron,1))
yHidden3 = np.zeros((numberNeuron,1))
yOutput = 0
E = np.zeros((numberNeuron,numberHiddenLayer))

hHidden1 = np.zeros((numberNeuron,1))
hHidden2 = np.zeros((numberNeuron,1))
hHidden3 = np.zeros((numberNeuron,1))
hOutput = np.zeros((1,1))
e = np.zeros((1,1))

Y = np.zeros((40000,1))
Ypredict = np.zeros((10000,1))
def sigmoid(MU):
    return 1 / (1 + np.exp(-MU))
    
def th(MU):
    return (1.0 - np.exp(-MU)) / (1.0 + np.exp(-MU))

def optimization():
    for cycle in range (0, numberCycle):
        for i in range(0,40000):
            y = forward(numberLayer,xTrain[i,:])
            Y[i,0] = y
            #print("out ",y)
            e[0,0] = getError(y,yTrain[i,0])
            #print("e ", e)
            backpropagation()
            updateWeight(i)
            #print(y)
        #loss = -1*np.mean(np.multiply(yTrain,np.log(Y)) + np.multiply((1-yTrain),np.log(1-Y)))
        #print(cycle)
        #alfa[0,0] += np.power(10.0,-6)
        if (cycle % 10 == 0):
            for j in range(0,10000):
                predictY = forward(numberLayer,xTest[j,:])
                Ypredict[j,0] = predictY
            getAccuracy(cycle,yTest,Ypredict)
            print(Y)
                #print("loss ",loss)
            
def forward(numberLayer,Data):
    for hl in range (0,numberLayer):
        if(hl == 0):
            hHidden1[:,0] = np.dot(wInput[:,:],Data.T) + b[:,hl]
            yHidden1[:,0] = sigmoid(hHidden1[:,0])
            #print(wInput[:,:]) 
        elif(hl == 1):
            hHidden2[:,0] = np.dot(wHidden1[:,:],yHidden1[:,0]) + b[:,hl]
            yHidden2[:,0] = sigmoid(hHidden2[:,0])
            #print("pesos ", wCurrent[:,:])
            #print(wHidden1[:,:]) 
        elif(hl==2):
            hHidden3[:,0] = np.dot(wHidden2[:,:],yHidden2[:,0]) + b[:,hl]
            yHidden3[:,0] = sigmoid(hHidden3[hl-1,:])
            #print(wHidden2[:,:]) 
        else: 
            hOutput[0,0] = np.dot(wOutput[:,0],yHidden3[:,0].T) + bOut[0,0]
            yOutput = sigmoid(hOutput[0,0])
            #print("pesos", wOutput[:,0])
            
            #print("out",yOutput) 
    return yOutput

def getError(y,yTrain):
    e =  y - yTrain
    #print(e)
    return e
     
def backpropagation():
    for hl in range (numberLayer-1,0,-1):
        #if(hl == 0):
        #    E[hl,:] = np.dot(wInput[:,:].T,E[:,hl-1]) 
        if(hl == 1):
            sumError = E[:,hl].sum()
            E[:,hl-1] = sumError * (yHidden1[:,0]*(1-yHidden1[:,0]))
            #print("whiden ", wHidden1[:,:])
            #print("error ", E[:,hl])
            #print("y ", yHidden[hl-1,:]*(1-yHidden[hl-1,:]))
            #print("hl ", hl, " e ",E[:,hl-1])
        if(hl == 2):
            sumError = E[:,hl].sum()
            E[:,hl-1] = sumError * (yHidden2[:,0]*(1-yHidden2[:,0]))
            #print("whiden ", wHidden2[:,:])
            #print("error ", E[:,hl])
            #print("y ", yHidden[hl-1,:]*(1-yHidden[hl-1,:]))
            #print("hl ", hl, " e ",E[:,hl-1])
        else:
            E[:,hl-1] = np.dot(yHidden3[:,0]*(1-yHidden3[:,0]),e[0,0])
            #E[:,hl-1] = np.dot(wOutput[:,0],e[0,0])
            #print("whiden ", wOutput[:,0])
            #print("error ", e[0,0])
            #print("y  ", yHidden[hl-1,:]*(1-yHidden[hl-1,:]))
            #print("hl ", hl, " e ",E[:,hl-1])
            
def updateWeight(i):
    alfa = np.power(10.0,-1.0) 
    for hl in range (0,numberLayer):
        if(hl == 0):
            #E[:,hl] = wHidden1[:,:].T * E[:,hl] * (yHidden[hl-1,:]*(1-yHidden[hl-1,:]))
            derivateInput[:,:] = np.dot(E[:,hl].reshape((3,1)),xTrain[i,:].reshape((1,100))) 
            wInput[:,:] = ((1-lamba * alfa) * wInput[:,:]) - alfa * derivateInput[:,:]
            #print("derivate ",derivateInput)
            #print("menor ",hl)
        elif(hl==1):
            derivate1[:,:] = np.dot(E[:,hl],yHidden1[:,0].T)  
            #print(wHidden[:,:,hl-1])
            wHidden1[:,:] = ((1-lamba * alfa) * wHidden1[:,:])  - alfa * derivate1[:,:]
         #  b[hl-1,:] -=  alfa[0,0] * E[:,hl].T
        elif(hl == 2):
            derivate2[:,:] = np.dot(E[:,hl],yHidden2[:,0].T)  
            #print(wHidden[:,:,hl-1])
            wHidden2[:,:] = ((1-lamba * alfa) * wHidden2[:,:]) - alfa * derivate2[:,:]
       #     b[hl-1,:] -=  alfa[0,0] * E[:,hl].T
            #print(wHidden[:,:,hl-1])
            #print("menor ",hl)
        else:
            #print(wOutput[:,0])
            derivate = np.dot(e[0,0],yHidden3[:,0])
            wOutput[:,0] = ((1-lamba * alfa) * wOutput[:,0]) - alfa*derivate
            bOut[0,0] -= alfa * e[0,0]
            #print(wOutput[:,0])
            #print("igual ",hl)
def getAccuracy(i,yTest,predictTest):
    print("accuracy in the cycle: ",i, " is: ",accuracy_score(yTest, predictTest.round()))
    
optimization()

# In[16]:
def sigmoid2(M):
     return  1 / (1 + np.exp(-M))


#Define My Linear Regression 
def MyLinearRegression(pTrain,pYTrain,pTest,pYTest,cross):
    alfa = np.power(10.0,-3.0)
    lamba = np.power(10.0,-8.0)
    parameters = np.empty((100,1))
    parameters[:] = 0.1 * np.random.rand() 
    #parameters[:] = np.random.rand() 
    count = 0
    cycle = 100000
    while(count < cycle):
        for i in range(1,5):
            currentTrain = pTrain[10000*(i-1):(i*10000),:]
            currentYTrain = pYTrain[10000*(i-1):(i*10000)]
            h = np.matmul(currentTrain,parameters)
            y = sigmoid2(h)
            error = y - currentYTrain
            derivate = np.matmul(error.T,currentTrain)
            parameters = parameters - (alfa *derivate.T) - (lamba * parameters)
            loss = -1*np.mean(np.multiply(currentYTrain,np.log(y)) + np.multiply((1-currentYTrain),np.log(1-y)))
        count= count + 1
        if (count % 1000 == 0) or (count == 1):  
            hTest = np.matmul(pTest,parameters)
            hTest = sigmoid2(hTest)
            print("----Cross validation ",cross," ----")
            print("----Epoch ",count," ----")
            print("loss: ",loss)
            print("accuracy: ",accuracy_score(pYTest, hTest.round()))
        #alfa = alfa + np.power(10.0,-15.0)
    if cross == 3:
        global mrlWeight
        mrlWeight = parameters

#optimization()
#import pyprind
#pbar = pyprind.ProgBar(45)

#classes = np.array([0, 1])
#for _ in range(45):
#    X_train, y_train = get_minibatch(doc_stream, size=1000)
#    X_train = vect.transform(X_train)
#    clf.partial_fit(X_train, y_train, classes=classes)
    #pbar.update()


# Depending on your machine, it will take about 2-3 minutes to stream the documents and learn the weights for the logistic regression model to classify "new" movie reviews. Executing the preceding code, we used the first 45,000 movie reviews to train the classifier, which means that we have 5,000 reviews left for testing:

# In[17]:
from sklearn.model_selection import KFold
kFolds = KFold(n_splits=4)
countCross = 1

for train, test in kFolds.split(xTrain):
    trainD = xTrain[train,:]
    testD = xTrain[test,:]
    yTrain1 = yTrain[train,:]
    yTest = yTrain[test,:]
    print("------------- My Logistics Regression ---------")
    MyLinearRegression(trainD,yTrain1,testD,yTest,countCross)
    countCross = countCross + 1
#X_test, y_test = get_minibatch(doc_stream, size=5000)
#X_test = vect.transform(X_test)
#print('Accuracy: %.3f' % clf.score(X_test, y_test))
#Exercise 3: compare  with your Neural Network



# I think that the predictive performance, an accuracy of ~87%, is quite "reasonable" given that we "only" used the default parameters and didn't do any hyperparameter optimization. 
# 
# After we estimated the model perfomance, let us use those last 5,000 test samples to update our model.

# In[18]:



# <br>
# <br>

# # Model Persistence

# In the previous section, we successfully trained a model to predict the sentiment of a movie review. Unfortunately, if we'd close this IPython notebook at this point, we'd have to go through the whole learning process again and again if we'd want to make a prediction on "new data."
# 
# So, to reuse this model, we could use the [`pickle`](https://docs.python.org/3.5/library/pickle.html) module to "serialize a Python object structure". Or even better, we could use the [`joblib`](https://pypi.python.org/pypi/joblib) library, which handles large NumPy arrays more efficiently.
# 
# To install:
# conda install -c anaconda joblib

# In[21]:


import joblib
import os
if not os.path.exists('./pkl_objects'):
    os.mkdir('./pkl_objects')
    
joblib.dump(vect, './vectorizer.pkl')
joblib.dump(clf, './clf.pkl')


# Using the code above, we "pickled" the `HashingVectorizer` and the `SGDClassifier` so that we can re-use those objects later. However, `pickle` and `joblib` have a known issue with `pickling` objects or functions from a `__main__` block and we'd get an `AttributeError: Can't get attribute [x] on <module '__main__'>` if we'd unpickle it later. Thus, to pickle the `tokenizer` function, we can write it to a file and import it to get the `namespace` "right".

# In[22]:


get_ipython().run_cell_magic('writefile', 'tokenizer.py', "from nltk.stem.porter import PorterStemmer\nimport re\nfrom nltk.corpus import stopwords\n\nstop = stopwords.words('english')\nporter = PorterStemmer()\n\ndef tokenizer(text):\n    text = re.sub('<[^>]*>', '', text)\n    emoticons = re.findall('(?::|;|=)(?:-)?(?:\\)|\\(|D|P)', text.lower())\n    text = re.sub('[\\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')\n    text = [w for w in text.split() if w not in stop]\n    tokenized = [porter.stem(w) for w in text]\n    return text")


# In[23]:


from tokenizer import tokenizer
joblib.dump(tokenizer, './tokenizer.pkl')


# Now, let us restart this IPython notebook and check if the we can load our serialized objects:

# In[24]:


import joblib
tokenizer = joblib.load('./tokenizer.pkl')
vect = joblib.load('./vectorizer.pkl')
clf = joblib.load('./clf.pkl')


# After loading the `tokenizer`, `HashingVectorizer`, and the tranined logistic regression model, we can use it to make predictions on new data, which can be useful, for example, if we'd want to embed our classifier into a web application -- a topic for another IPython notebook.

# In[25]:


example = ['I did not like this movie']
X = vect.transform(example)
clf.predict(X)


# In[26]:


example = ['I loved this movie']
X = vect.transform(example)
clf.predict(X)


# In[ ]:




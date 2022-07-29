from nltk import corpus
import pandas as pd
import numpy as np 

dataset = pd.read_csv("file.csv",delimiter='\t', quoting=3)

#cleaning the text
import re
import nltk
nltk.download('stopwords')                   #geting the stopwords(non_predictors) like he/she the/a/an etc.
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer   #removing confilcts like loved/love ,hate/hated

corpus = []
for i in range(0,1000):
    review = re.sub("[^a-zA-Z]"," ",dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    s = stopwords.words('english')
    s.remove('not')
    review = [ps.stem(word)for word in review if not word in set(s)]
    review = ' '.join(review)
    corpus.append(review)

#creating the bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.svm import SVC  #achieved 77 in logistic regressor model
gb = SVC(kernel = "rbf",random_state=0)
gb.fit(x_train,y_train)

y_pred = gb.predict(x_test)
##checking the accuracy/confusion matrix 
from sklearn.metrics import accuracy_score,confusion_matrix
print(confusion_matrix(y_pred,y_test))
print(accuracy_score(y_pred,y_test))






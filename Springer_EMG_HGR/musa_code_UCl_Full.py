#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:39:31 2018

@author: simao
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import auc


from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode


import matplotlib.pyplot as plt
from pandas import set_option
import pandas as pd


import numpy as np 
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn 

'''
import keras
from keras.layers import Dense, Dropout, Activation

#from keras.optimizers import SGD
#from keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers import SGD 

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

#from keras.optimizers import SGD
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.layers import Dense, Flatten, Conv1D
from keras.models import Sequential
import keras.utils
from keras.layers import Dense
#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
'''
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import pickle
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np

import os
from sklearn import model_selection
#from tensorflow.keras import layers, models, metrics, losses, optimizers, callbacks
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

output1=[]
output2=[]
outputKNN=[]
outputLogistic=[]
outputRF=[]
outputSVM=[]
class_num: int = 8
repeat_num: int = 110
timestep_num: int = 400
channel_num: int = 20
sub_no=1


import numpy as np
import pandas as pd
path='F:/PhD/EMG_All/Dataset/UC2018 DualMyo Hand Gesture Dataset/dualmyo_dataset.pkl'
a = pd.read_pickle(path)
print(np.array(a[1]).shape)
Data=a[0]
label=a[1]

print(np.array(Data).shape)
print(np.array(label).shape)

sub_no=sub_no+1
sample_num = class_num * repeat_num
data = {'X': np.zeros(shape=(sample_num, timestep_num, channel_num), ),
                  'raw_Y': np.zeros(shape=(sample_num, timestep_num, class_num), dtype=int),
                  'Y': np.zeros(shape=(sample_num, timestep_num, class_num), dtype=int),
                  'start': np.zeros(shape=(sample_num,), dtype=int),
                  'end': np.zeros(shape=(sample_num,), dtype=int),
                  'S1': np.zeros(shape=(sample_num, timestep_num, channel_num), ),
                  'S2': np.zeros(shape=(sample_num, timestep_num))
                     }

data['X']=Data


X=data['X']
#print(X.shape)

data_x=np.moveaxis(X, 1, 2)
data_1=data_x[:,:,30:370]
#data_1=data_x

def zcruce(X, th=0.005):
    th = 0.005
    cruce = 0
    for cont in range(len(X) - 1):
        can = X[cont] * X[cont + 1]
        can2 = abs(X[cont] - X[cont + 1])
        if can < 0 and can2 > th:
            cruce = cruce + 1
    return cruce

zcruc=lambda x:  zcruce(x)

#mean absolute value
def musa_mav(x):
    N=len(x);
    out=sum(abs(x))/N;
    return out

energy= lambda x: np.sum(x*x)
power= lambda x: np.sum(x*x)/3
hpa= lambda x: np.sum(x-np.mean(x))/3 # Hjort parameter activity 
iqr=lambda x: (np.quantile(x,.75)-np.quantile(x,.25))    

#def power(x):
#   return pd.DataFrame(x).apply(ssc, axis=1)
#slop sign changes

def myssc(x,th=0.005): #% th: noise threshold
    N=len(x);
    ssc=0;
    for i in range(2,N-1):
        if ((x[i]>x[i-1] and x[i]>x[i+1]) or (x[i]<x[i-1] and x[i]<x[i+1])) and (abs(x[i]-x[i+1])>th and abs(x[i]-x[i-1])>th):
            ssc=ssc+1;
        else:
            ssc=ssc+0;
            
    return ssc
# end of slope change
#willson threshold
def mywamp(x,th=0.005): 
    wamp=0;
    N=len(x);
    for i in range(N-1):
        if abs(x[i]-x[i+1])>th:
            wamp=wamp+1;
    return wamp
    

def ssc(x):
    return pd.DataFrame(x).apply(myssc, axis=1)

def ener(x):
    return pd.DataFrame(x).apply(energy, axis=1)



def mav(x):
    return pd.DataFrame(x).apply(musa_mav, axis=1)


'''
features=[]
for d in data_1:
    print(d.shape)
    print(ssc(d))
'''

e=ener(data_1[1])


print(e)


hp=ssc(data_1[1])
print(hp.shape)
f=mav(data_1[1])






f=np.mean(data_1[1], axis=-1)
print(f.shape)
a=np.concatenate((f,e,hp),axis=-1)
print(a.shape)
#zcrossing end



#print(f.shape)
#print(f)
from scipy import stats

def mean(x):
    return np.mean(x,axis=-1)
def mav(x):
    return pd.DataFrame(x).apply(musa_mav, axis=1)
def std(x):
    return np.std(x,axis=-1)
def ptp(x):
    return np.ptp(x,axis=-1)
def var(x):
    return np.var(x,axis=-1)
def minim(x):
    return np.min(x,axis=-1)
def maxim(x):
    return np.max(x,axis=-1)

def argminim(x):
    return np.argmin(x,axis=-1)

def argmaxim(x):
    return np.argmax(x,axis=-1)

def rms(x):
    return np.sqrt(np.mean(x**2,axis=-1))
def abs_diff_signal(x):
    return np.sum(np.abs(np.diff(x,axis=-1)),axis=-1)
 

def skewness(x):
    return stats.skew(x,axis=-1)
def kurtosis(x):
    return stats.kurtosis(x,axis=-1)
def zcru(x):                                         #zcrossing
    return pd.DataFrame(x).apply(zcruc, axis=1)
def ener(x):
    return pd.DataFrame(x).apply(energy, axis=1)  #energy

def hjortpa(x):
    return pd.DataFrame(x).apply(hpa, axis=1)  #hjpart 
def interqr(x):
    return pd.DataFrame(x).apply(iqr, axis=1)  #interquartile
def ssc(x):                                         #slope change
    return pd.DataFrame(x).apply(myssc, axis=1)
def wlson(x):
    return pd.DataFrame(x).apply(mywamp, axis=1)


def concatenate_features(x):
    #return np.concatenate((mean(x), std(x),ptp(x),var(x),minim(x),maxim(x),argminim(x),argmaxim(x),rms(x),abs_diff_signal(x),skewness(x),kurtosis(x),zcru(x),ener(x),hjortpa(x),interqr(x),ssc(x),wlson(x)),axis=-1)
    return np.concatenate((mav(x),std(x),ptp(x),var(x),minim(x),maxim(x),argminim(x),argmaxim(x),rms(x),abs_diff_signal(x),skewness(x),kurtosis(x),zcru(x),ener(x),hjortpa(x),interqr(x),ssc(x),wlson(x)),axis=-1)
    #return np.concatenate((mean(x),std(x),ptp(x),var(x),minim(x),maxim(x),argminim(x),argmaxim(x),rms(x),abs_diff_signal(x),skewness(x),kurtosis(x),zcru(x),axis=-1) #maximum 98.8

features=[]
for d in data_1:
    features.append(concatenate_features(d))
    #print(d.shape)
    #print(features.shape)


features_array=np.array(features)
print(features_array.shape)
#print(features_array)
updat_data=features_array




y=np.array(label)
X=np.array(features_array)

''' Feature Selection'''
#Feature_selection

'''
print(X.shape,y.shape)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectFromModel
log=SelectFromModel(LogisticRegression(penalty='l1',C=2, solver='liblinear'))
log.fit(X,y)
#print(log.summary())
selected_feature_uc2018=log.get_support()
print(log.get_support())

#print(log.estimator_.coef_)

updat_data=log.transform(X)
print(updat_data.shape)
updat_data=pd.DataFrame(updat_data)





import statsmodels.api as sm
X1 = sm.add_constant(X)
model = sm.OLS(y, X1)
model = model.fit()
p=model.tvalues
print(p.shape)
'''
update_data=X
# array([ 0.37424023, -2.36373529,  3.57930174])
# compute p-values
#t.sf(np.abs(model.tvalues), n-X1.shape[1])*2 
# array([7.09042437e-01, 2.00854025e-02, 5.40073114e-04])  

#model.summary()
def metrics(Y_validation,predictions):
    print('Accuracy:', accuracy_score(Y_validation, predictions))
    print('F1 score:', f1_score(Y_validation, predictions,average='weighted'))
    print('Recall:', recall_score(Y_validation, predictions,average='weighted'))
    print('Precision:', precision_score(Y_validation, predictions, average='weighted'))
    print('\n clasification report:\n', classification_report(Y_validation, predictions))
    #print('\n confusion matrix:\n',confusion_matrix(Y_validation, predictions))
    #print('\n auc:\n',auc(Y_validation, predictions))
    #print('\n classification report\n',classification_report(Y_validation, predictions))
    #Creating confussion matrix
    cm1 = confusion_matrix(Y_validation, predictions)
    total1=sum(sum(cm1))
    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    print('Sensitivity : ', sensitivity1 )
    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    print('Specificity : ', specificity1)
    
    print('Extra Tree Classifier')
b_score=0
#for i in range(100):
X_train, X_test, y_train, y_test = train_test_split(updat_data,label, test_size=0.33, random_state=42)
Labels_train = y_train
Features_train = X_train
Features_test=X_test
Labels_test=y_test

classifier='Etra_Tree'
from sklearn.ensemble import ExtraTreesClassifier
#print("Extra Tree classifier original")
model=ExtraTreesClassifier()
model.fit(Features_train, Labels_train)
#y_pred=model.predict(Features_test)
#metrics(Labels_test,y_pred)



#k-fold CV

from sklearn.model_selection import cross_val_score
#5 fold cross validation
scores=cross_val_score(model, updat_data,label,scoring='r2', cv=5)
print(scores)
#printing the average score
av=np.mean(scores)
print(np.mean(scores))
if b_score<av:
    b_score=av
    #print('i=',i)

print('best_score',b_score)
#result1 = "{0}:{1}".format(subj_name, b_score)
result1 = "{0}".format(b_score)
output1.append(result1)

predictions = model.predict(Features_test)
metrics(Labels_test,predictions)


print('1: KNN')
#Testing different values of n_neighbors
Features_train, Features_test, Labels_train, Labels_test = train_test_split(updat_data,label, test_size=0.33, random_state=42)

scaler = preprocessing.StandardScaler().fit(Features_train)
Features_train_scaler = scaler.transform(Features_train)
Features_test_scaler = scaler.transform(Features_test)

limit=3
x=[x for x in range(1,limit)]
ytest=[0 for x in range(1,limit)]
ytrain=[0 for x in range(1,limit)]
ytestScaler=[0 for x in range(1,limit)]
ytrainScaler=[0 for x in range(1,limit)]
ytestPCA=[0 for x in range(1,limit)]
ytrainPCA=[0 for x in range(1,limit)]
ytestPCAScaler=[0 for x in range(1,limit)]
ytrainPCAScaler=[0 for x in range(1,limit)]
for i in range(1,limit):
    KNN = KNeighborsClassifier(n_neighbors=i)
    
    KNN.fit(Features_train, Labels_train)
    trainScore=KNN.score(Features_train,Labels_train)
    testScore=KNN.score(Features_test,Labels_test)
  
    
    #print('n-neighbors value:',i)
    ytrain[i-1]=trainScore   
    ytest[i-1]=testScore
    

"""    
plt.figure(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
plt.plot(x,ytrain,label='Train')
plt.plot(x,ytest,label='Validation')
plt.plot(x,ytrainScaler,label='Train-Scaling')
plt.plot(x,ytestScaler,label='Validation-Scaling')
plt.xlabel('n-neighbors')
plt.ylabel('Accuracy')
plt.title('n-neighbors vs Accuracy')
plt.legend()
plt.xticks(range(1,6))
plt.savefig('KNN-Algorithm.pdf', dpi=300)
plt.show()  
"""
print('The best score with data validation: ', max(ytest),'with Neighbors: ',x[ytest.index(max(ytest))])
   

Features_train_test = np.concatenate((Features_train, Features_test), axis=0)
Labels_train_test = np.concatenate((Labels_train, Labels_test), axis=0)
KNN = KNeighborsClassifier(n_neighbors=x[ytest.index(max(ytest))])
KNN.fit(Features_train_test, Labels_train_test)
#predictions = KNN.predict(Features_test)
#metrics(Labels_test,predictions)
scores=cross_val_score(KNN, updat_data,label,scoring='r2', cv=5)

#print(scores)
#printing the average score
Knn_av=np.mean(scores)
#print(np.mean(scores))
#if knn_b_score<av:
#    b_score=av
print('KNN-av', Knn_av)
#resultKNN = "{0}:{1}".format(subj_name, Knn_av)
resultKNN = "{0}".format(Knn_av)
outputKNN.append(resultKNN)

   


"""8. Random Forest"""
print('8: Random Forest')
RF_final=0
#for i in range(100):
X=updat_data
y=label
RF_score=0;

X_train, X_test, y_train, y_test= train_test_split(updat_data,label, test_size=0.33, random_state=42)
limit=100
for i in range(1,limit,10):
    RF = RandomForestClassifier(n_estimators=i)
    
    #RF.fit(X_train, y_train)
    #RF_Score=RF.score(X_test, y_test)
    from sklearn.model_selection import cross_val_score
    #5 fold cross validation
    Score=cross_val_score(RF, updat_data,label,scoring='r2', cv=5)
    #print(scores)
    #printing the average score
    RF_av=np.mean(Score)
    #print(np.mean(scores))
    #print('RF_av',RF_av)
    if RF_score<RF_av:
        RF_score=RF_av
   
    #RF_sum.append(RFo)
    #RF_scaler_sum.append(RFscaler)
 
print('RF BestRF:', RF_score)
#if RF_final<RF_score:
#    RF_final=RF_score
        
#print('Final_RF_output:', RF_final)
#resultRF = "{0}:{1}".format(subj_name,RF_score)
resultRF = "{0}".format(RF_score)
#print('RF Accuracy original:', np.mean(RF_sum), 'RF Scaling:', np.mean(RF_scaler_sum))
#resultRF = "{0}:{1}:{2}".format(subj_name, np.mean(RF_sum),np.mean(RF_scaler_sum) )
outputRF.append(resultRF)
   
'''
   
#SVM
        



'''
print('SVM')
SVM_score=0
from sklearn import svm
from sklearn.model_selection import cross_val_score
for i in range(100):
    X_train, X_test, y_train, y_test= train_test_split(updat_data,label, test_size=0.33, random_state=42)
    clf = svm.SVC(C=1000, kernel="linear", probability=True)
    #clf = svm.SVC(C= 0.1, gamma= 0.0001, kernel= 'rbf')
    #clf.fit(X_train, y_train)
    #score=clf.score(X_test, y_test)
    scores = cross_val_score(clf, updat_data,label, cv=5)
    #print(scores)
    #print(np.mean(scores))
    SVM_av=np.mean(scores)
    if SVM_score<SVM_av:
       SVM_score=SVM_av    
    
print('SVM_score',SVM_score )
#resultSVM = "{0}:{1}".format(subj_name,SVM_score)
resultSVM = "{0}".format(SVM_score)
#print('RF Accuracy original:', np.mean(RF_sum), 'RF Scaling:', np.mean(RF_scaler_sum))
#resultRF = "{0}:{1}:{2}".format(subj_name, np.mean(RF_sum),np.mean(RF_scaler_sum) )
outputSVM.append(resultSVM)
    
print(outputSVM)

#print(grid.best_params_)
#print(grid.best_estimator_)


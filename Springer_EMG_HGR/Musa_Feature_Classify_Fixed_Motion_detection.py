# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 10:19:24 2022

@author: shinlab
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 16:56:46 2022

@author: shinlab
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

from keras.layers import Dense
import matplotlib.pyplot as plt
from pandas import set_option
import pandas as pd
from keras.models import Sequential
import keras.utils
#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
import numpy as np 
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn 
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
#from keras.optimizers import SGD
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.layers import Dense, Flatten, Conv1D
import pickle
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np

import os
from sklearn import model_selection
from tensorflow.keras import layers, models, metrics, losses, optimizers, callbacks
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

class_num: int = 21
repeat_num: int = 30
timestep_num: int = 400
channel_num: int = 8
sub=['1_sub', '2_sub','gxt','gzc','hcs','hcx','lmf','qy','xsm', 'xx','xyy','yk_d','zct']

for subj_name in sub:
    print("subject=",subj_name)
    sample_num = class_num * repeat_num
    data = {'X': np.zeros(shape=(sample_num, timestep_num, channel_num), ),
                      'raw_Y': np.zeros(shape=(sample_num, timestep_num, class_num), dtype=int),
                      'Y': np.zeros(shape=(sample_num, timestep_num, class_num), dtype=int),
                      'start': np.zeros(shape=(sample_num,), dtype=int),
                      'end': np.zeros(shape=(sample_num,), dtype=int),
                      'S1': np.zeros(shape=(sample_num, timestep_num, channel_num), ),
                      'S2': np.zeros(shape=(sample_num, timestep_num))
                         }
    
    class_list = [f'A{i:02}' for i in range(1, 21)] + ['R']
    
    label=[]
    label_y=[]
    #subj_name='gxt'
    w1=20
    w2=10
    start_thresh=8
    save_detection_plots=False
    print(subj_name)
    txt_path = "D:/EMG_All/Project_Data_paper/China_dataset/Datasets/"+subj_name
    print('text path',txt_path)
    i=0
    l=1
    for sample_index, txt_file in enumerate(os.listdir(txt_path)):
        if txt_file != 'desktop.ini':
            sample_index=i
            #print('sample_index',sample_index)
            #print('txt file', txt_file)
            #print('internal loadtxt',f"{txt_path}/{txt_file}")
            p=txt_path+'/'+txt_file
            #print('path=',p)
            #p=os.path.join(txt_path, txt_file)
            #print('path=',p)
            #p='D:/EMG_All/Project_Data_paper/China_dataset/datasets/gzc/A01-1.txt'
            #print(os.path.exists(p))
            a=np.loadtxt(p)
            #print(a)
            
            data['X'][sample_index] = np.loadtxt(p)
            data['raw_Y'][sample_index][:, sample_index // repeat_num] = 1
            data['Y'] = data['raw_Y'].copy()
            l1=i // 30
            label.append(l1)
            #print(txt_file)
            #print('label=',l1)
            
            
            i=i+1
            l=l+1
            
    #print(data['Y'])
    #Y=data['Y']
    #print(Y.shape)
    #print(label)
    #print(np.array(label).shape)
    
    X=data['X']
    data_x=np.moveaxis(X, 1, 2)
    
        
        
    
    #data_1=data_x
    data_1=data_x[:,:,30:370]
    print(data_1.shape)
    #print(data.shape)
    group_list=[[i]*len(j) for i,j in enumerate(data)]
    #print(group_list)
    group_array=np.hstack(group_list)
    #print(group_array)
    #print(group_array.shape)
    
    
    f=np.mean(data_1, axis=-1)
    #print(f.shape)
    #print(f)
    from scipy import stats
    def mean(x):
        return np.mean(x,axis=-1)
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
    
    def concatenate_features(x):
        return np.concatenate((mean(x),std(x),ptp(x),var(x),minim(x),maxim(x),argminim(x),argmaxim(x),rms(x),abs_diff_signal(x),skewness(x),kurtosis(x)),axis=-1)
    
    features=[]
    for d in data_1:
        features.append(concatenate_features(d))
        #print(d.shape)
        #print(features.shape)
    
    features_array=np.array(features)
    #print(features_array.shape)
    #print(features_array)
    
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn. model_selection import GroupKFold, GridSearchCV
    '''
    clf=LogisticRegression()
    gkf=GroupKFold()
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    param_grid={'clf__C':[0.1,0.5,0.7,1,3,5,7]}
    gscv=GridSearchCV(pipe,param_grid,cv=gkf,n_jobs=28)
    gscv.fit(features_array,label,groups=None)
    
    print(gscv.best_score_)
    '''
    X_train, X_test, y_train, y_test = train_test_split(features_array,label, test_size=0.33, random_state=42)
    Labels_train = y_train
    Features_train = X_train
    Features_test=X_test
    Labels_test=y_test
    def metrics(Y_validation,predictions):
        print('Accuracy:', accuracy_score(Y_validation, predictions))
        print('F1 score:', f1_score(Y_validation, predictions,average='weighted'))
        print('Recall:', recall_score(Y_validation, predictions,average='weighted'))
        print('Precision:', precision_score(Y_validation, predictions, average='weighted'))
        #print('\n clasification report:\n', classification_report(Y_validation, predictions))
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
    
    from sklearn.ensemble import ExtraTreesClassifier
    print("Extra Tree classifier original")
    model=ExtraTreesClassifier()
    model.fit(Features_train, Labels_train)
    y_pred=model.predict(Features_test)
    metrics(Labels_test,y_pred)
        
    
    
    
    
    

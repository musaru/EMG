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


'''
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
'''

import numpy as np 
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn 

'''
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
import keras
from keras.layers import Dense, Dropout, Activation

#from keras.optimizers import SGD
#from keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers import SGD 
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
#from keras.optimizers import SGD
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.layers import Dense, Flatten, Conv1D
from tensorflow.keras import layers, models, metrics, losses, optimizers, callbacks
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

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
#import stats

output1=[]
output2=[]
outputKNN=[]
outputLogistic=[]
outputRF=[]
outputSVM=[]
class_num: int = 21
repeat_num: int = 30
timestep_num: int = 400
channel_num: int = 8
sub_no=1
sub=['1_sub', '2_sub','gxt','gzc','hcs','hcx','lmf','qy','xsm', 'xx','xyy','yk_d','zct']

#subj_name='gxt'

for subj_name in sub:
    print("subject=",subj_name,'_number_',sub_no)
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
    
    class_list = [f'A{i:02}' for i in range(1, 21)] + ['R']
    
    label=[]
    label_y=[]
    
    w1=20
    w2=10
    start_thresh=8
    save_detection_plots=False
    
    #txt_path = "D:/EMG_All/Project_Data_paper/China_dataset/Datasets/"+subj_name
    
    txt_path = "F:/PhD/EMG_All/Project_Data_paper/China_dataset/Datasets/"+subj_name
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
    #print(repeat_num)
    
    X=data['X']
    #print(X.shape)
    
    data_x=np.moveaxis(X, 1, 2)
    
    #data_1=data_x[:,:,30:370] # with segmentation
    
    data_1=data_x # with segmentation
    print(data_x.shape)
    #data_1=data_x
    #print(data.shape)
    group_list=[[i]*len(j) for i,j in enumerate(data)]
    #print(group_list)
    group_array=np.hstack(group_list)
    #print(group_array)
    #print(group_array.shape)
    #feature function
    
    #zcrossing start
    
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
        
    '''
    def ssc(x):
        return pd.DataFrame(x).apply(myssc, axis=1)
    
    def ener(x):
        return pd.DataFrame(x).apply(energy, axis=1)
    
    
    
    def mav(x):
        return pd.DataFrame(x).apply(musa_mav, axis=1)
    
    
    
    features=[]
    for d in data_1:
        print(d.shape)
        print(ssc(d))
    
    
    e=ssc(data_1[1])
    
    
    print(e)
    
    
    hp=ssc(data_1[1])
    print(hp.shape)
    
    
    
    
    
    
    f=np.mean(data_1[1], axis=-1)
    print(f.shape)
    a=np.concatenate((f,e,hp),axis=-1)
    print(a.shape)
    #zcrossing end
    '''
    
    
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
    
    
    #'''
    #Feature_selection
    y=np.array(label)
    X=np.array(features_array)
    print('Original FEature shape X', X.shape)
    print('Original label shape X', y.shape)
    
    ''' Feature Selection'''
    
    
    '''
    print(X.shape,y.shape)
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.feature_selection import SelectFromModel
    log=SelectFromModel(LogisticRegression(penalty='l1',C=2, solver='liblinear'))
    log.fit(X,y)
    #print(log.summary())
    selected_feature_true_false=log.get_support()
    print(log.get_support())
    S_coef=log.estimator_.coef_
    print(log.estimator_.coef_)
    
    updat_data=log.transform(X)
    print(updat_data.shape)
    updat_data=pd.DataFrame(updat_data)
    
    
    #calculating p value
    
    import statsmodels.api as sm
    X1 = sm.add_constant(X)
    model = sm.OLS(y, X1)
    model = model.fit()
    print(model.summary())
    p=model.tvalues
    print(p)
    
    
    lm = LinearRegression()
    lm.fit(X,y)
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)
    
    newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))
    
    # Note if you don't want to use a DataFrame replace the two lines above with
    # newX = np.append(np.ones((len(X),1)), X, axis=1)
    # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))
    
    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b
    
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]
    
    sd_b = np.round(sd_b,3)
    ts_b = np.round(ts_b,3)
    p_values = np.round(p_values,3)
    params = np.round(params,4)
    
    myDF3 = pd.DataFrame()
    myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
    print(myDF3)
    coef=params
    
    '''
    # array([ 0.37424023, -2.36373529,  3.57930174])
    # compute p-values
    #t.sf(np.abs(model.tvalues), n-X1.shape[1])*2 
    # array([7.09042437e-01, 2.00854025e-02, 5.40073114e-04])  
    
    
    #print(updat_data.head)
    #'''
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn. model_selection import GroupKFold, GridSearchCV
    
    
    updat_data=X
    label=y
    '''
    clf=LogisticRegression()
    gkf=GroupKFold()
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    param_grid={'clf__C':[0.1,0.5,0.7,1,3,5,7]}
    gscv=GridSearchCV(pipe,param_grid,cv=gkf,n_jobs=28)
    gscv.fit(features_array,label,groups=None)
    
    print(gscv.best_score_)
    '''
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
        
    
    #'''
    print('Extra Tree Classifier')
    b_score=0
    for i in range(100):
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
        #print(scores)
        #printing the average score
        av=np.mean(scores)
        #print(np.mean(scores))
        if b_score<av:
            b_score=av
        #print('i=',i)
    
    print('best_score',b_score)
    #result1 = "{0}:{1}".format(subj_name, b_score)
    result1 = "{0}".format(b_score)
    output1.append(result1)
    
    predictions = model.predict(Features_test)
    metrics(Labels_test,predictions)
    '''
    
    '''
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
    print('Subject: KNN-av', subj_name,Knn_av)
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
    
    #'''
    
    """
    Extra tree tuning
    """
    
    '''
    from numpy import std,mean
    
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.ensemble import ExtraTreesClassifier
    from matplotlib import pyplot
     
    # get the dataset
     
    # get a list of models to evaluate
    def get_models():
    	models = dict()
    	# define number of trees to consider
    	n_trees = [10, 50, 100, 500, 1000, 5000,6000,10000]
    	for n in n_trees:
    		models[str(n)] = ExtraTreesClassifier(n_estimators=n)
    	return models
     
    # evaluate a given model using cross-validation
    def evaluate_model(model, X, y):
    	# define the evaluation procedure
    	cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    	# evaluate the model and collect the results
    	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    	return scores
    b_score=0
    #for i in range(100):
    # define dataset
    X, y =updat_data,label
    # get the models to evaluate
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
    	# evaluate the model
    	scores = evaluate_model(model, X, y)
    	# store the results
    	results.append(scores)
    	names.append(name)
    	# summarize the performance along the way
    	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    	av=mean(scores)
    	#if b_score<av:
        #    b_score=av
        #print('i=',i)
    # plot model performance for comparison
    print('best_score', b_score)
    
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()
    
    '''
    
with open("F:/PhD/EMG_All/Project_Data_paper/China_dataset/Publication/All Result/result_without_feature_selection/Extra_Tree_Classifier.csv", mode="w") as f:
 f.write("\n".join(output1))
with open("F:/PhD/EMG_All/Project_Data_paper/China_dataset/Publication/All Result/result_without_feature_selection/KNN_Classifier.csv", mode="w") as f:
 f.write("\n".join(outputKNN))
#with open("D:/EMG_All/Project_Data_paper/China_dataset/Publication/All Result/Logisti_Regression_Classifier.txt", mode="w") as f:
 #   f.write("\n".join(outputLogistic))
with open("F:/PhD/EMG_All/Project_Data_paper/China_dataset/Publication/All Result/result_without_feature_selection/RandomForest_Classifier.csv", mode="w") as f:
 f.write("\n".join(outputRF))
with open("F:/PhD/EMG_All/Project_Data_paper/China_dataset/Publication/All Result/result_without_feature_selection/OutputSVM_Classifier.csv", mode="w") as f:
 f.write("\n".join(outputSVM))
    

'''
    
    #print(score)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
               ]
scores = ['precision', 'recall']
for score in scores:
    print(" Tuning hyper-parameters for ", score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(updat_data,label)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    
clf.best_params_
'''
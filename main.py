# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 22:03:06 2022

@author: jalowe
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import math

import warnings
warnings.filterwarnings("ignore")

ds = pd.read_csv('Crypto_PhishingDataset_With_3features.csv')
print(ds)
#ds.drop(['site'], axis=1, inplace = True)
#Handle missing values
#ds = ds.dropna()


#Convert String/Bool to integer data
ds.Result[ds.Result == 'yes'] = 1
ds.Result[ds.Result == 'no'] = -1

#Defining dependent columns
Y = ds['Result'].values
Y = Y.astype('int')

#Defining independent columns
X = ds.drop(labels=['Result'], axis = 1)

#Split data into train and test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

########################################################################

#RANDOM FOREST CLASSIFIER | Will be used to identify the feature from most important to least important
k_min = 100
k_max = 301
accuracy_list=[]
scores =[]
n_estimator_list=[]

# This definition will get the best n_estimator for every dataset that 
# is being passed here. 
def get_N_estimate(X_train, X_test, Y_train, Y_test):
   
    
    for k in range(k_min, k_max,10):
        rfc = RandomForestClassifier(n_estimators=k, n_jobs=-1,random_state=0 )
        rfc.fit(X_train, Y_train)
        y_pred = rfc.predict(X_test)
        scores.append(accuracy_score(Y_test, y_pred))
       # print('K = ',k, " accuracy = ",scores[len(scores)-1])

    max_score_ind = round(scores.index(max(scores)),5)
    n_estimator = (max_score_ind*10)+100
    print( 'The best N_estimator values based on accuracy is : ',n_estimator)
    n_estimator_list.append(n_estimator)
    scores.clear()
    run_RandomForest(X_train, X_test, Y_train, Y_test, n_estimator)

#This run_RandomForest Definition will re-run the random forest with the best n_estimator
#and store its accuracy value in a accuracy_list
def run_RandomForest(X_train, X_test, Y_train, Y_test,n_estimator):
    rfc = RandomForestClassifier(n_estimators=n_estimator, n_jobs=-1,random_state=0)
    rfc.fit(X_train, Y_train)
    y_pred = rfc.predict(X_test)
    accuracy = round(accuracy_score(Y_test, y_pred)*100,5)
    print('Accuracy: ', accuracy,'%')
    accuracy_list.append(round(accuracy,5))
    print()



for index in range(1,33):
    select = RFE(RandomForestClassifier(random_state=0), n_features_to_select=index)
    select.fit(X_train, Y_train)
    X_train_rfe = select.transform(X_train)
    X_test_rfe = select.transform(X_test)
    print('Selected Feature: ', index)
    get_N_estimate(X_train_rfe, X_test_rfe, Y_train, Y_test)
   # print(sel.get_support())
   
#print('Accuracy List from k iteration: ', accuracy_list)  
print()

best_index = accuracy_list.index(max(accuracy_list))
linked_n_estimator = n_estimator_list[best_index-1]

print("The best no. of features used is: ", best_index+1)
print("The best n_estimator of the best no. of features used: ", linked_n_estimator)
#print("accuracy rate (RFC) :",round(accuracy_list[best_index],5) )
sel = RFE(RandomForestClassifier(random_state=0), n_features_to_select=best_index)
sel.fit(X_train, Y_train)
X_train_rfe_actual = sel.transform(X_train)
X_test_rfe_actual = sel.transform(X_test)

print()


#########################################################################
#XGBOOST | This will be used on a seperate  after filtration of the features to see the accuracy of the model
print("Initial Prediction XGBOOST")
boost = xgb.XGBClassifier()
boost.fit(X_train_rfe_actual, Y_train)
y_pred = boost.predict(X_test_rfe_actual)
accuracy = accuracy_score(Y_test, y_pred)

print('Initial Prediction Accuracy: ',round(accuracy*100,5),"%")



n_estimators = [100,150,200,300,400]
max_depth = [2,3,6,8]
learning_rate = [0.05,0.1,0.2,0.3]
min_child_weight = [1,2,3,4]
reg_lambda = [0.1, 1.0, 2.0, 5.0, 10.0]

xgb_score=[]
best_min_child_weight=0
best_learning_rate=0.0
best_max_depth=0
best_n_estimator=0


def tuner(X_train, X_test, Y_train, Y_test,reg_lambda,n_estimators,
      max_depth,learning_rate,min_child_weight):
    for h in range (0, len(reg_lambda)):
        for i in range (0,len(n_estimators)): 
            for j in range (0,len(max_depth)):
                for k in range(0,len(learning_rate)):
                    for l in range (0,len(min_child_weight)):
                        boost = xgb.XGBClassifier(reg_lambda=reg_lambda[h],n_estimators=n_estimators[i],max_depth=max_depth[j],
                                                  learning_rate=learning_rate[k], min_child_weight = min_child_weight[l],
                                                  eval_metric='mlogloss')
                        boost.fit(X_train, Y_train)
                        y_pred = boost.predict(X_test)
                        accuracy = accuracy_score(Y_test, y_pred)
                        xgb_score.append(accuracy)
                        sort_score = sorted(xgb_score)
                        current = xgb_score[len(xgb_score)-1]
                        print('reg_lambda: ',reg_lambda[h], ' n_estimator: ', n_estimators[i], ' max_depth: ',
                              max_depth[j], ' learning_rate: ', learning_rate[k],' min_child_weight: ',min_child_weight[l])
                        print('accuracy: ', accuracy)
                        
                        if current == sort_score[len(sort_score)-1]:
                            min_child_weight_index = l
                            learning_rate_index = k
                            max_depth_index = j
                            n_estimators_index = i
                            reg_lambda_index = h
                            
                            
    highest_value = xgb_score.index(max(xgb_score))
    #print(highest_value)
    #print(max(xgb_score))
    #print(xgb_score)
    print()
    
    best_min_child_weight = min_child_weight[min_child_weight_index]
    print('Best minimum child weight: ',best_min_child_weight)


    best_learning_rate = learning_rate[learning_rate_index]
    print('Best Learning Rate: ',best_learning_rate)
 

    best_max_depth = max_depth[max_depth_index]
    print('Best Max Depth:',best_max_depth )


    best_n_estimator = n_estimators[n_estimators_index]
    print('Best N estimator: ',best_n_estimator)
   
    best_reg_lambda = reg_lambda[reg_lambda_index]
    print('Best L2 Regularization Rates: ',best_reg_lambda )
    return (best_n_estimator,best_max_depth,best_learning_rate,best_min_child_weight,best_reg_lambda)
    
stats = tuner(X_train_rfe_actual, X_test_rfe_actual, Y_train, Y_test,reg_lambda, n_estimators,
      max_depth,learning_rate,min_child_weight)
print()
print("Hyparameter Tuned Prediction XGBOOST")


boost = xgb.XGBClassifier(n_estimators=stats[0],max_depth=stats[1],
                          learning_rate=stats[2], min_child_weight = stats[3],
                          eval_metric='mlogloss',reg_lambda = stats[4])
boost.fit(X_train_rfe_actual, Y_train)
y_pred = boost.predict(X_test_rfe_actual)
accuracy = accuracy_score(Y_test, y_pred)

print('Tuned Prediction Accuracy',round(accuracy*100,5),"%")

###########################################################################
#Confusion matrix
'''plt.plot(range(k_min, k_max), scores)
plt.xlabel('Value of n_estimators for Random Forest Classifier')
plt.ylabel('Testing Accuracy')
'''
cm = confusion_matrix(Y_test, y_pred)
sns.heatmap(cm, annot=True)

tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
accu = (tp + tn)/(tn+fn+tp+fp)

print()
print("True Negative = ",tn)
print("False Positive = ",fp)
print("False Negative = ",fn)
print("True Positive = ",tp)
print()
prec = tp/(tp+fp)
print("Precision = ", round(prec,5)*100,"%")

rec = tp/(tp+fn)
print("Recall Score = ", round(rec,5)*100,"%")

f_score = (2*(prec*rec))/(rec+prec)
print("F-Score = ", round(f_score,5)*100,"%")

mcc = ((tp*tn)-(fp*fn))/(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
print("MCC = ", round(mcc,5)*100,"%")



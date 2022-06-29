import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def composition(a,b):
  return a*100/(a+b)

def XNOR (a, b):
  if a != b:
      return 0
  else:
      return 1

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)


def load_features (data_file, subject_no, train_size, test_size):
  csv = pd.read_excel(data_file, subject_no, skiprows = 1)

  nutrients = np.concatenate((csv[csv.columns[2:5]][0:train_size].to_numpy(), csv[csv.columns[2:5]][train_size+2:test_size].to_numpy()))

  insulin = np.concatenate((csv[csv.columns[6]][0:train_size].to_numpy(), csv[csv.columns[6]][train_size+2:test_size].to_numpy()))
  
  DFB = np.concatenate((csv[csv.columns[8]][0:train_size].to_numpy(), csv[csv.columns[8]][train_size+2:test_size].to_numpy()))
  DFB*=100

  sbgl = np.concatenate((csv[csv.columns[10]][0:train_size].to_numpy(), csv[csv.columns[10]][train_size+2:test_size].to_numpy()))

  features = np.zeros((nutrients.shape[0],6))
  print(features.shape)

  for i in range(0, features.shape[0]):
    features[i][0] = composition(nutrients[i][0],nutrients[i][1])
    features[i][1] = DFB[i]
    features[i][2] = sbgl[i]
    features[i][3] = insulin[i]
    features[i][4:7] = nutrients[i][0:2]

  return features

def load_labels (data_file, subject_no, train_size, test_size):
  csv = pd.read_excel(data_file, subject_no, skiprows = 1)
  CGM = np.concatenate((csv[csv.columns[10:130]][0:train_size].to_numpy(), csv[csv.columns[10:130]][train_size+2:test_size].to_numpy()))
  target = np.zeros(CGM.shape[0])

  for i in range(CGM.shape[0]):
    if (np.max(CGM[i])>=8.89):
      target[i] = 1

  return target

xls = pd.ExcelFile('Train_valid_data_AUC_DFB.xlsx')

P01 = load_features (xls, 'P01', 19, 31)
P01_ = load_labels (xls, 'P01', 19, 31)

P02 = load_features (xls, 'P02', 28, 45)
P02_ = load_labels (xls, 'P02', 28, 45)

P03 = load_features (xls, 'P03', 17, 28)
P03_ = load_labels (xls, 'P03', 17, 28)

P04 = load_features (xls, 'P04', 22, 36)
P04_ = load_labels (xls, 'P04', 22, 36)

P05 = load_features (xls, 'P05', 22, 37)
P05_ = load_labels (xls, 'P05', 22, 37)


def LOMO_SVM (features, label):

  loo = LeaveOneOut()
  loo.get_n_splits(features)
  prediction_result = []
  true_p = 0
  false_p = 0
  false_n = 0

  for train_index, test_index in loo.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    clf = make_pipeline(StandardScaler(), SVC(C=0.001, gamma='auto', degree=2, class_weight='balanced'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
  
    prediction_result.append(XNOR(y_test, y_pred))

    TP, FP, TN, FN = perf_measure(y_test, y_pred)

    true_p+=TP
    false_p+=FP
    false_n+=FN

  print('micro average f1-score:', 2*(true_p/(true_p+false_p))*(true_p/(true_p+false_n))/((true_p/(true_p+false_p))+(true_p/(true_p+false_n))))
  print('Accuracy:', np.mean(prediction_result)*100, '%')
LOMO_SVM (P05, P05_)



def LOMO_DT (features, label):

  loo = LeaveOneOut()
  loo.get_n_splits(features)
  prediction_result = []
  true_p = 0
  false_p = 0
  false_n = 0

  for train_index, test_index in loo.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    decision_tree = tree.DecisionTreeClassifier(criterion="gini", max_depth=3, class_weight=None, min_samples_split = 2)
    decision_tree = decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
  
    prediction_result.append(XNOR(y_test, y_pred))

    feature_names = ['carb_compositiion', 'DFB', 'ibgl', 'insulin', 'carb', 'fat']
    class_names = ['PPHG-', 'PPHG+']

    TP, FP, TN, FN = perf_measure(y_test, y_pred)

    true_p+=TP
    false_p+=FP
    false_n+=FN

  print('micro average f1-score:', 2*(true_p/(true_p+false_p))*(true_p/(true_p+false_n))/((true_p/(true_p+false_p))+(true_p/(true_p+false_n))))
  print('Accuracy', np.mean(prediction_result)*100, '%')
LOMO_DT (P04, P04_)



def LOSO_SVM (all_features, all_label):

  loo = LeaveOneOut()
  loo.get_n_splits(all_features)
  prediction_result = []
  true_p = 0
  false_p = 0
  false_n = 0

  for train_index, test_index in loo.split(all_features):
    
    X_train, X_test = all_features[train_index], all_features[test_index]
    y_train, y_test = all_label[train_index], all_label[test_index]

    X_train = np.concatenate((X_train[0],X_train[1],X_train[2],X_train[3]))
    y_train = np.concatenate((y_train[0],y_train[1],y_train[2],y_train[3]))
    X_test = X_test[0]
    y_test = y_test[0]

    clf = make_pipeline(StandardScaler(), SVC(C=0.1, kernel = 'poly', gamma='auto', degree=3, class_weight=None))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    prediction_result.append(sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True))

    TP, FP, TN, FN = perf_measure(y_test, y_pred)

    true_p+=TP
    false_p+=FP
    false_n+=FN

  print('micro average f1-score:', 2*(true_p/(true_p+false_p))*(true_p/(true_p+false_n))/((true_p/(true_p+false_p))+(true_p/(true_p+false_n))))
  print('Accuracy', np.mean(prediction_result)*100, '%')

  return

subjects = np.array((P01, P02, P03, P04, P05),dtype=object)
labels = np.array((P01_, P02_, P03_, P04_, P05_),dtype=object)
LOSO_SVM (subjects, labels)



def LOSO_DT (all_features, all_label):

  loo = LeaveOneOut()
  loo.get_n_splits(all_features)
  prediction_result = []
  true_p = 0
  false_p = 0
  false_n = 0

  for train_index, test_index in loo.split(all_features):
    
    X_train, X_test = all_features[train_index], all_features[test_index]
    y_train, y_test = all_label[train_index], all_label[test_index]

    X_train = np.concatenate((X_train[0],X_train[1],X_train[2],X_train[3]))
    y_train = np.concatenate((y_train[0],y_train[1],y_train[2],y_train[3]))
    X_test = X_test[0]
    y_test = y_test[0]

    decision_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=3, class_weight=None, min_samples_split = 2)
    decision_tree = decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)

    prediction_result.append(sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True))
    TP, FP, TN, FN = perf_measure(y_test, y_pred)

    true_p+=TP
    false_p+=FP
    false_n+=FN

    #print(true_p)
    #print(false_p)

  print('micro average f1-score:', 2*(true_p/(true_p+false_p))*(true_p/(true_p+false_n))/((true_p/(true_p+false_p))+(true_p/(true_p+false_n))))
  print('Accuracy', np.mean(prediction_result)*100, '%')

  return

subjects = np.array((P01, P02, P03, P04, P05),dtype=object)
labels = np.array((P01_, P02_, P03_, P04_, P05_),dtype=object)
LOSO_DT (subjects, labels)




subjects = np.concatenate((P01, P02, P03, P04, P05))
labels = np.concatenate((P01_, P02_, P03_, P04_, P05_))

# report which features were selected by RFE
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

# define RFE
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=1)
# fit RFE
rfe.fit(P01, P01_)
# summarize all features
for i in range(P01.shape[1]):
	print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))






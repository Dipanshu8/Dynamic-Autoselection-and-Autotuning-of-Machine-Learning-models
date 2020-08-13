# Dynamic Autoselection and Autotuning of Machine Learning Models
## Table of contents
* [Introduction](#Introduction)
* [Technical Specification](#technical_specification)
* [Milestones](#milestones)
* [Software Setup](#software_setup)
* [Packages Required](#packages)
* [Code](#code)
## Introduction
Cloud network monitoring data is dynamic and distributed. Signals to monitor the cloud can appear, disappear or change their importance and clarity over time. Machine learning (ML) models tuned to a given data set can therefore quickly become inadequate. A model might be highly accurate at one point in time but may lose its accuracy at a later time due to changes in input data and their features.Distributed learning with dynamic model selection is therefore often required. Under such selection, poorly performing models (although aggressively tuned for the prior data) are retired or put on standby while new or standby models are brought in. The well-known method of Ensemble ML (EML) may potentially be applied to improve the overall accuracy of a family of ML models. Unfortunately, EML has several disadvantages, including the need for continuous training, excessive computational resources, requirement for large training datasets, high risks of overfitting, and a time-consuming model-building process. In this paper, we propose a novel cloud methodology for automatic ML model selection and tuning that automates the model build and selection and is competitive with existing methods.

We use unsupervised learning to better explore the data space before the generation of targeted supervised learning models in an automated fashion. In particular, we create a Cloud DevOps architecture for autotuning and selection based on container orchestration and messaging between containers, and take advantage of a new autoscaling method to dynamically create and evaluate instantiation of ML algorithms. The proposed methodology and tool are demonstrated on cloud network security datasets.
## Technical Specification
Algorithms:
* Naive Bayes
* Decision Tree
* K-Means
* Random Forest
* KNN
* SVM
* MLP
* One vs All
## Milestones
The whole project is divided into many parts. The first part is we needed to find a dataset for the model which could fit in. Then we had to download the appropriate software and libraries in python along with anaconda navigator and PyCharm. Then , out of so many columns , the important columns were extracted and rest of them were dropped. Next , we needed to code for the same and create the model and then the model had to be trained on the data.  Python packages such as Numpy, Scipy, Scikit learn, Pandas, Matplotlib etc. have been used.
Some baseline tests have been performed over the dataset  using existing algorithms such as multi-class classification.  The dataset feature “attack name” is used as the label.
## Software Setup
1.Anaconda Navigator [Click Here](https://www.anaconda.com/products/individual)
2.Jupyter Notebook from Anaconda Navigator
3.Get large Dataset from cloud
## Packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
from keras.models import Sequential,load_model
from sklearn.cluster import KMeans
from keras.layers import Dense,Softmax
from keras.utils import to_categorical
import pickle
import h5py
## Code
### Training Dataset
train_data=pd.read_csv(r'cloud_train.csv')
train_data=train_data.sample(n=1000)
### Testing Dataset
test_data=pd.read_csv(r'cloud_test.csv')
test_data=test_data.sample(n=1000)

train_data.head()
train_data.shape

(1000, 45)


train_data.isnull().sum()
plt.figure(figsize=(20,10))
sns.countplot(train_data['attack_cat'])
attack_type=np.unique(train_data['attack_cat'].values)
attack_type

array(['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic',
       'Normal', 'Reconnaissance', 'Shellcode'], dtype=object)


'''
sample_size=10000
norm_percent=20
train_normal=train_data[train_data['attack_cat']=='Normal'].sample(n=int(np.ceil(norm_percent*sample_size/100)))
train_data1=train_data[train_data['attack_cat']!='Normal'].sample(n=10000)
Train_data=pd.concat([train_normal,train_data1])
'''


train_data.shape

(1000, 45)

plt.figure(figsize=(20,10))
sns.countplot(test_data['attack_cat'])
cols=np.unique(test_data['attack_cat'].values)


Cols

array(['Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic',
       'Normal', 'Reconnaissance', 'Shellcode', 'Worms'], dtype=object)

label_encoder=LabelEncoder()
def label_encoding(data):
    columns=data.columns
    for cols in columns:
       # print(cols)
        if(isinstance(data[cols].values[0],str)):
            data[cols]=label_encoder.fit_transform(data[cols].values)
return data


train=label_encoding(train_data)
test=label_encoding(test_data)
X_train=train.drop(['attack_cat','label'],axis=1)
X_test=test.drop(['attack_cat','label'],axis=1)
Y_train=train['attack_cat'].apply(lambda x:int(x))
Y_test=test['attack_cat'].apply(lambda x:int(x))

scaler=StandardScaler()
x_train=scaler.fit_transform(X_train)
x_test=scaler.fit_transform(X_test)


pd.DataFrame(x_train).head()

print(np.unique(Y_train))
print(x_train.shape,x_test.shape)
### Naive Bayes
NB=GaussianNB()
NB.fit(x_train,Y_train)

GaussianNB(priors=None, var_smoothing=1e-09)


acc_NB=NB.score(x_test,Y_test)

cm=confusion_matrix(NB.predict(x_test),Y_test)
acc_per_class_NB=cm.diagonal()/cm.sum(axis=0)

### Knn Classifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,Y_train)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')


acc_knn=knn.score(x_test,Y_test)

cm=confusion_matrix(knn.predict(x_test),Y_test)
acc_per_class_knn=cm.diagonal()/cm.sum(axis=0)

### Random Forest
rf=RandomForestClassifier(n_estimators=200)
rf.fit(x_train,Y_train)

RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)


acc_rf=rf.score(x_test,Y_test)

cm=confusion_matrix(rf.predict(x_test),Y_test)
acc_per_class_rf=cm.diagonal()/cm.sum(axis=0)

### Decision Tree
dt=DecisionTreeClassifier()
dt.fit(x_train,Y_train)

DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')


acc_dt=dt.score(x_test,Y_test)

cm=confusion_matrix(dt.predict(x_test),Y_test)
acc_per_class_dt=cm.diagonal()/cm.sum(axis=0)

### MLP
mlp=MLPClassifier()
mlp.fit(x_train,Y_train)

MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100,), learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=None, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)


acc_mlp=mlp.score(x_test,Y_test)

### Accuracy Plot
plt.bar(['NB','KNN','RF','DT'],[acc_NB,acc_knn,acc_rf,acc_dt],color=['red','green','blue','cyan'])
cm=confusion_matrix(mlp.predict(x_test),Y_test)
acc_per_class_mlp=cm.diagonal()/cm.sum(axis=0)

acc_per_class=np.vstack([acc_per_class_NB,acc_per_class_knn,acc_per_class_rf,acc_per_class_dt,acc_per_class_mlp])

Acc_per_class=pd.DataFrame(acc_per_class,columns=cols,index=['Naive Bayes','Kth-Nearest Neighbor','Random forest',
                                                                    'DecisionTreeClassifier','MLPClassifier'])

Acc_per_class
one_all=OneVsRestClassifier(SVC(kernel='linear'))
one_all.fit(x_train,Y_train)

OneVsRestClassifier(estimator=SVC(C=1.0, break_ties=False, cache_size=200,
                                  class_weight=None, coef0=0.0,
                                  decision_function_shape='ovr', degree=3,
                                  gamma='scale', kernel='linear', max_iter=-1,
                                  probability=False, random_state=None,
                                  shrinking=True, tol=0.001, verbose=False),
                    n_jobs=None)


one_all.score(x_test,Y_test)
### hyper_parameter tuning for KNN
error_rate = []
####  Will take some time
for i in range(1,40):
 knn = KNeighborsClassifier(n_neighbors=i)# for knn hyperparameter is n_neighbors
 knn.fit(X_train,Y_train)
 pred_i = knn.predict(X_test)

 error_rate.append(np.mean(pred_i != Y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color="blue", linestyle="dashed", marker="o",
 markerfacecolor="red", markersize=10)
plt.title("Error Rate vs. K Value")
plt.xlabel("K")
plt.ylabel("Error Rate")

index_optimum=error_rate.index(min(error_rate))
print(index_optimum)
K_optimum=index_optimum+1
knn= KNeighborsClassifier(n_neighbors=K_optimum)
knn.fit(x_train,Y_train)
acc_knn_optimum=knn.score(x_test,Y_test)
### hyper_parameter tuning for Random Forest
error_rate = []
#### Will take some time
min_i=0
min_error=9999
for i in range(50,300,30):
  rf= RandomForestClassifier(n_estimators=i)# for rf hyperparameter is n_estimators
  rf.fit(x_train,Y_train)
  pred_i =rf.predict(x_test)
  if np.mean(pred_i != Y_test)<=min_error:
    min_i=i
    min_error=np.mean(pred_i != Y_test)
    #print(np.mean(pred_i != Y_test))
  error_rate.append(np.mean(pred_i != Y_test))

plt.figure(figsize=(10,6))
plt.plot(range(50,300,30),error_rate,color="blue", linestyle="dashed", marker="o",
 markerfacecolor="green", markersize=10)
plt.title("Error Rate vs. n Value")
plt.xlabel("n")
plt.ylabel("Error rate")
print(min_i)
n_optimum=min_i
rf=RandomForestClassifier(n_estimators=n_optimum)
rf.fit(x_train,Y_train)
acc_rf_optimum=rf.score(x_test,Y_test)
### hyper_parameter tuning for MLPClassifier
error_rate = []
min_i=0
min_error=999
for i in range(100,600,100):
  mlp=MLPClassifier(max_iter=i)# for mlp hyperparameter is max_iter
  mlp.fit(x_train,Y_train)
  pred_i =mlp.predict(x_test)
  if np.mean(pred_i != Y_test)<=min_error:
    min_i=i
    min_error=np.mean(pred_i != Y_test)
  error_rate.append(np.mean(pred_i != Y_test))

plt.figure(figsize=(10,6))
plt.plot(range(100,600,100),error_rate,color="blue", linestyle="dashed", marker="o",
 markerfacecolor="blue", markersize=10)
plt.title("Error Rate vs. n Value")
plt.xlabel("max_iter")
plt.ylabel("Error Rate")
print(min_i)
iter_optimum=min_i
mlp=MLPClassifier(max_iter=iter_optimum)
mlp.fit(x_train,Y_train)
acc_mlp_optimum=mlp.score(x_test,Y_test)
x_Train=np.vstack((x_train,x_test))
Y_Train=np.vstack((Y_train,Y_test))

x_train1=np.hstack((x_Train,Y_Train.reshape(-1,1)))
pd.DataFrame(x_train1).head()
### Clutering
km=KMeans(n_clusters=6)
km.fit(x_train1)
clusters=km.labels_
cluster=np.unique(clusters)
centroids=km.cluster_centers_


cluster_data={}
for c in cluster:
 cluster_data[c]=x_train1[km.labels_==c]

pd.DataFrame(cluster_data[1]).head()
data_per_cluster=[]
for d in cluster_data:
   data_per_cluster.append(cluster_data[d].shape[0])

plt.bar(range(len(data_per_cluster)),data_per_cluster)
fig, ax = plt.subplots(figsize=(10, 6))
colors=['red','green','blue','yellow','cyan','orange']
for i,C in enumerate(cluster_data):
  plt.scatter(cluster_data[C][:,1],cluster_data[C][:,5],
              c=colors[i])
### Accuracy Comparison per cluster
acc_list_per_cluster={}
confusion_matrix_per_cluster={}
for d in cluster_data:
   x=cluster_data[d][:,:-1]
   y=cluster_data[d][:,-1]
   if x.shape[0]>=300:
     acc_list_per_cluster[d]=[]
     confusion_matrix_per_cluster[d]=[]
     #print(x.shape,y.shape)
     train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=False)
     knn=KNeighborsClassifier(n_neighbors=5)
     knn.fit(train_x,train_y)
     acc_list_per_cluster[d].append(knn.score(test_x,test_y))
     confusion_matrix_per_cluster[d].append(confusion_matrix(knn.predict(test_x),test_y))
     rf=RandomForestClassifier()
     rf.fit(train_x,train_y)
     acc_list_per_cluster[d].append(rf.score(test_x,test_y))
     confusion_matrix_per_cluster[d].append(confusion_matrix(rf.predict(test_x),test_y))

colors=['red','green','blue','cyan','cyan','orange']

j=0
for c in acc_list_per_cluster:
  #print(j)
  plt.subplot(2,2,j+1)
  plt.bar(['KNN','RF'],acc_list_per_cluster[c],color=colors[j])
  j+=1
  plt.title('cluster'+str(c))
### Heat map Visualisation
sns.heatmap(confusion_matrix_per_cluster[0][0],annot=True,fmt='d')
sns.heatmap(confusion_matrix_per_cluster[5][1],annot=True,fmt='d')
sns.heatmap(confusion_matrix_per_cluster[1][1],annot=True,fmt='d')


%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mode
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

np.random.seed(0)

train = pd.read_csv("./landsat_train_small.csv", header=None, sep=",")
test = pd.read_csv("./landsat_test.csv", header=None, sep=",")
validation = pd.read_csv("./landsat_validation.csv", header=None, sep=",")

RFmodel = RandomForestClassifier(n_estimators=10, max_features=None)
# fit x_train, y_train
RFmodel.fit(train.iloc[:,1:], train.iloc[:,0])
valids = RFmodel.predict(validation.iloc[:,1:])
print("Accuracy of model on the validation set: {}".format(accuracy_score(validation.iloc[:,0], valids)))

preds = RFmodel.predict(test)

myclasses = np.sort(train.iloc[:,0].unique()).tolist()

image = np.array(np.split(preds,np.arange(3000,9000000,3000)))

# plt.imshow(image, cmap=plt.cm.gist_earth)
plt.imsave("/Users/ctl564/Documents/LSDA/HW2/Q.4.1_testfile_image.jpeg", image, cmap=plt.cm.gist_earth)


class RFRTrees():
    
    def __init__(self, n):
        
        self.used_features = []
        self.predictions = np.zeros((n,10))

        
    def select_features(self, X, n_features):
        candidates = set(np.random.choice(X.columns, size = n_features, replace=False).tolist())
        
        while candidates in self.used_features:
            candidates = set(np.random.choice(X.columns, size = n_features, replace=False).tolist())
            
        self.used_features.append(candidates)
        
        return candidates
              
            
    def build_decision_tree(self, X, y, n_features):
        use_features = list(self.select_features(X, n_features))
        print("Selected features: {}".format(use_features))
        X = X.loc[:, use_features]
        self.DTmodel = DecisionTreeClassifier()
        self.DTmodel.fit(X,y)
        
        return use_features
          
        
    def decision_tree_classification(self, Xtrain, ytrain, Xtest, n_features):
        features = self.build_decision_tree(Xtrain, ytrain, n_features)
        Xtest = Xtest.loc[:, features]
        tree_preds = self.DTmodel.predict(Xtest)
        
        return tree_preds
        
          
    def RF_classification(self, Xtrain, ytrain, Xtest, n_features):
        
        for i in range(10):
            self.predictions[:,i] = self.decision_tree_classification(Xtrain, ytrain, Xtest, n_features)
        
        label_counts = np.apply_along_axis(mode, axis=1, arr=self.predictions)
        labels = [label_counts[i][0] for i in range(label_counts.shape[0])]
        
        return labels


    
myRF = RFRTrees(n=validation.shape[0])
predsRT = myRF.RF_classification(train.iloc[:,1:], train.iloc[:,0], validation.iloc[:,1:], 2)
print("Accuracy of RF with RTs model on the validation set: {}".format(accuracy_score(validation.iloc[:,0], predsRT)))

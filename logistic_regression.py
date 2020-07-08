import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, precision_score, recall_score
from sklearn.preprocessing import RobustScaler


true_labels_train = pd.read_hdf("./danai/Analysis/analysis2.h5", "true_labels_train")
true_labels_val = pd.read_hdf("./danai/Analysis/analysis2.h5", "true_labels_val")

## Log-reg for the age at last donation and sex

measurements_train = true_labels_train.xs(["SEX", "AGE"], axis=1)
measurements_train.loc[measurements_train.SEX == "F", "SEX"] = 1
measurements_train.loc[measurements_train.SEX == "M", "SEX"] = 0
measurements_train = measurements_train.values

measurements_test = true_labels_val.xs(["SEX", "AGE"], axis=1)
measurements_test.loc[measurements_test.SEX == "F", "SEX"] = 1
measurements_test.loc[measurements_test.SEX == "M", "SEX"] = 0
measurements_test = measurements_test.values

scaler = RobustScaler().fit(measurements_train[:,1].reshape((measurements_train.shape[0],1)))
scaled_measurements_train = scaler.transform(measurements_train[:,1].reshape((measurements_train.shape[0],1)))
scaled_measurements_test = scaler.transform(measurements_test[:,1].reshape((measurements_test.shape[0],1)))

measurements_train = np.hstack((measurements_train[:,0].reshape((-1,1)), scaled_measurements_train))
clf = LogisticRegression(fit_intercept=True, n_jobs=20, solver="lbfgs")
clf.fit(measurements_train, true_labels_train.T2D.values)
pred_labels_train = clf.predict(measurements_train)
pred_probs_train = clf.predict_proba(measurements_train)
measurements_test = np.hstack((measurements_test[:,0].reshape((-1,1)), scaled_measurements_test))
pred_labels_val = clf.predict(measurements_test)
pred_probs_val = clf.predict_proba(measurements_test)


print("\nCoefficients:\nIntercept ~= {}     Sex ~= {}    Age at last donation ~= {}\n".format(round(clf.intercept_[0], 3), round(clf.coef_[0][0],3), round(clf.coef_[0][1],3)))

print("\nOn the training set:\nAccuracy: {}\nMatthew's coeff: {}\nRecall: {}\nPrecision: {}\nConfusion matrix:\n{}".format(accuracy_score(true_labels_train.T2D.values, pred_labels_train), matthews_corrcoef(true_labels_train.T2D.values, pred_labels_train), recall_score(true_labels_train.T2D.values, pred_labels_train), precision_score(true_labels_train.T2D.values, pred_labels_train), confusion_matrix(true_labels_train.T2D.values, pred_labels_train)))

print("\n\nOn the test set:\nAccuracy: {}\nMatthew's coeff: {}\nRecall: {}\nPrecision: {}\nConfusion matrix:\n{}".format(accuracy_score(true_labels_val.T2D.values, pred_labels_val), matthews_corrcoef(true_labels_val.T2D.values, pred_labels_val), recall_score(true_labels_val.T2D.values, pred_labels_val), precision_score(true_labels_val.T2D.values, pred_labels_val), confusion_matrix(true_labels_val.T2D.values, pred_labels_val)))

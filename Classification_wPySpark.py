# Copy the data to the corresponding directories on your distributed file system
ham = sc.wholeTextFiles("hdfs:///user/lsda/spam_ham/ham/*.txt")
spam = sc.wholeTextFiles("hdfs:///user/lsda/spam_ham/spam/*.txt")

# Inspect the data
ham.take(1)
spam.take(1)

from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import MulticlassMetrics


# Split the emails into words
def split2words(email):
    
    words = email.split()
    
    return words


ham = ham.map(lambda e: split2words(e[1]))
spam = spam.map(lambda e: split2words(e[1]))

# Hashing
hashingTF = HashingTF(numFeatures=10000)
ham = hashingTF.transform(ham)
spam = hashingTF.transform(spam)

# Label training/test data
labeled_ham = ham.map(lambda h: LabeledPoint(0, h))
labeled_spam = spam.map(lambda s: LabeledPoint(1, s))
labeled_combo = labeled_ham.union(labeled_spam)
train, test = labeled_combo.randomSplit([0.8, 0.2], seed=0)

# Fit the model on the training data   
LRlbfgs = LogisticRegressionWithLBFGS.train(train)

# Predict labels on the test data
def apply2test(model, x):
    feat = x.features
    pred = model.predict(feat)
    return (float(pred), x.label)


pred_true_labels = test.map(lambda x: apply2test(LRlbfgs, x))

# Classification evaluation
metrics = MulticlassMetrics(pred_true_labels)
print("Classification accuracy: {}".format(metrics.accuracy))

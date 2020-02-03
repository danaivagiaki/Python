from scipy import io, optimize, sparse
import numpy

fname = "data/aut-avn.mat"
content = io.loadmat(fname, struct_as_record=True)
X = content['X']
y = content['Y']
print(type(X))
print(type(y))


class SparseLSSVM():
    
    def __init__(self, lam=1.0):
        """ Instantiates the regression model.
        
        Parameters
        ----------
        lam : float, default 1.0
            The regularization parameter lambda
        """
        
        self.lam = lam

        
    def get_params(self, deep=True):
        """ Returns the parameters of the model
        """
        
        return {"lam": self.lam}

   
    def set_params(self, **parameters):
        """ Sets the parameters of the model
        """        
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        return self

    
    def fit(self, X, y):
        """
        Fits the model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
            Sparse data matrix
        y : Array of shape [n_samples, 1]
            Dense vector
        """   
        
        y = numpy.array(y).reshape((len(y), 1))
        
        self._n = X.shape[0]
        self._X, self._y = X, y
        
        # make use of optimizer provided by the scipy package
        start = numpy.zeros(self._X.shape[0], numpy.float64).reshape((-1,1))
        self.c_opt, self.f_opt, d = optimize.fmin_l_bfgs_b(self._get_function,
                                                 start,
                                                 m=10,
                                                 fprime=self._get_function_grad,
                                                 iprint=1)
        self.c_opt = self.c_opt.reshape((-1,1))

        return self

    
    def _get_function(self, c):
    
        c = c.reshape((-1,1))
        self.Kc = self._X.dot(self._X.transpose().dot(c))
        f = (1/self._n)*numpy.dot((self._y-self.Kc).T, (self._y-self.Kc)) + numpy.dot(self.lam*c.T, self.Kc)
        
        return f

    
    def _get_function_grad(self, c):
    
        c = c.reshape((-1,1))
        
        grad = (-2/self._n)*self._X.dot(self._X.transpose().dot(self._y-self.Kc)) + 2*self.lam*self.Kc
        
        return grad

    
    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
            Sparse data matrix

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """           
        
        # NOTE: You need to convert the real-valued 
        # predictions to -1 or +1 depending on them
        # being negative or positive
        

        # X = X.toarray().reshape((X_test.shape[0], -1))
        c = self.c_opt
        preds = X.dot(self._X.transpose().dot(c))
        preds[numpy.where(preds < 0)] = -1
        preds[numpy.where(preds >= 0)] = +1
        
        return preds           



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

print("Number of training instances: {}".format(X_train.shape[0]))
print("Number of test instances: {}".format(X_test.shape[0]))
print("Number of features: {}".format(X_train.shape[1]))

clf = SparseLSSVM(lam=0.001)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Accuracy of model: {acc}".format(acc=acc))

#!/usr/bin/env python
# coding: utf-8

# # Exercise 4: Logistic Regression, Bayes and EM
# 
# In this assignment you will implement several algorithms as learned in class.
# 
# ## Read the following instructions carefully:
# 
# 1. This jupyter notebook contains all the step by step instructions needed for this exercise.
# 2. Write **efficient vectorized** code whenever possible. Some calculations in this exercise take several minutes when implemented efficiently, and might take much longer otherwise. Unnecessary loops will result in point deduction.
# 3. You are responsible for the correctness of your code and should add as many tests as you see fit. Tests will not be graded nor checked.
# 4. Write your functions in this notebook only. **Do not create Python modules and import them**.
# 5. You are allowed to use functions and methods from the [Python Standard Library](https://docs.python.org/3/library/) and [numpy](https://www.numpy.org/devdocs/reference/) only. **Do not import anything else.**
# 6. Your code must run without errors. During the environment setup, you were given a specific version of `numpy` to install (1.15.4). Changes of the configuration we provided are at your own risk. Any code that cannot run will not be graded.
# 7. Write your own code. Cheating will not be tolerated.
# 8. Submission includes this notebook only with the exercise number and your ID as the filename. For example: `hw4_123456789_987654321.ipynb` if you submitted in pairs and `hw4_123456789.ipynb` if you submitted the exercise alone.
# 9. Answers to qualitative questions should be written in **markdown** cells (with $\LaTeX$ support). Answers that will be written in commented code blocks will not be checked.
# 
# ## In this exercise you will perform the following:
# 1. Implement Logistic Regression algorithm.
# 1. Implement EM algorithm.
# 1. Implement Navie Bayes algorithm that uses EM for calculating the likelihood.
# 1. Visualize your results.

# # I have read and understood the instructions: 203378039_308381847 

# In[1]:


import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# make matplotlib figures appear inline in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make the notebook automatically reload external python modules
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Function for ploting the decision boundaries of a model
# You will use it later
def plot_decision_regions(X, y, classifier, resolution=0.01):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


# ## Reading the data

# In[3]:


training_set = pd.read_csv('training_set.csv')
test_set = pd.read_csv('test_set.csv')
X_training, y_training = training_set[['x1', 'x2']].values, training_set['y'].values
X_test, y_test = test_set[['x1', 'x2']].values, test_set['y'].values


# ## Visualizing the data
# (2 points each - 8 points total)
# 
# Plot the following graphs for the training set:
# 
# For the first feature only:
# 1. For the first 1000 data points plot a histogram for each class on the same graph (use bins=20, alpha=0.5).
# 1. For all the data points plot a histogram for each class on the same graph (use bins=40, alpha=0.5).
# 
# For both features:
# 1. For the first 1000 data points plot a scatter plot where each class has different color
# 1. For all the data points plot a scatter plot where each class has different color

# In[4]:


#### Your code here ####
first1000samp, first1000lbl = X_training[:1000], y_training[:1000]
plt.title("First 1000 samples, x1 only")
first_class = first1000samp[np.where(first1000lbl == 0)]
second_class = first1000samp[np.where(first1000lbl == 1)]
plt.hist(first_class[:,0], bins=20, alpha=0.5)
plt.hist(second_class[:,0], bins=20, alpha=0.5)
plt.title("First 1000 instance, x1 only")
plt.xlabel("x1")
plt.ylabel("Number of samples")
plt.show()


# In[5]:


first_class = X_training[np.where(y_training == 0)]
second_class = X_training[np.where(y_training == 1)]
plt.hist(first_class[:,0], bins=40, alpha=0.5,)
plt.hist(second_class[:,0], bins=40, alpha=0.5)
plt.title("All samples, x1 only")
plt.xlabel("x1")
plt.ylabel("Number of Samples")
plt.show()


# In[6]:


plt.scatter(first1000samp[:,0], first1000samp[:,1], color=['red' if i == 0 else 'blue' for i in first1000lbl])
plt.title("First 1000 samples")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# In[7]:


plt.scatter(X_training[:,0], X_training[:,1], color=['red' if i == 0 else 'blue' for i in y_training])
plt.title("All samples")
plt.xlabel("x1")
plt.ylabel("x2")


# ## Logistic Regression
# 
# (20 Points)
# 
# Implement the Logistic Regression algorithm that uses gradient descent for finding the optimal theta vector. 
# 
# Where:
# $$
# h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}
# $$
# 
# $$
# J(\theta)=\frac{1}{m} \sum_{d=1}^{m} - y^{(d)}ln(h_\theta(x^{(d)}) - (1 - y^{(d)})ln(1 - h_\theta(x^{(d)})
# $$
# 
# Your class should contain the following functions:
# 1. fit - the learning function
# 1. predict - the function for predicting an instance after the fit function was executed
# 
# \* You can add more functions if you think this is necessary
# 
# Your model should also store a list of the costs that you've calculated in each iteration

# In[8]:


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """
    
    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state
        self.thetas = None
        self.cost_hist = [] #Use a python list to save cost computations in order to reduce until convergence

    def fit(self, X, y):
        """ 
        Fit training data (the learning phase).
        Updating the theta vector in each iteration using gradient descent.
        Store the theta vector in an attribute of the LogisticRegressionGD object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        X = np.column_stack((np.ones(X.shape[0]), X))
        np.random.seed(self.random_state)
        self.thetas =np.random.random(size=X.shape[1])
        itr = 0
        difference = 1 + self.eps
        
        while difference > self.eps and itr < self.n_iter:  # Stop if improvement of the loss value is smaller than eps
            hypo = self.hypothesis(X)
            gradient = np.dot(X.T, (hypo - y)) / y.shape[0]
            self.thetas = self.thetas - self.eta * gradient
            cost = (-1 / len(X)) * np.sum(y * np.log(self.hypothesis(X)) + (1 - y) * np.log(1 - self.hypothesis(X)))
            self.cost_hist.append(cost)
            if itr > 0:
                difference = self.cost_hist[itr - 1] - self.cost_hist[itr]
            itr = itr + 1
                     
    def sigmoid(self,z):
       # compute the sigmoid with the combined features & thetas vector
        return 1 / (1 + np.exp(-z))
    
    def hypothesis(self, X):
        # compute the probabilty hypothesis
        return self.sigmoid(np.dot(X, self.thetas))
    
    def predict(self, X):
        """ Return the predicted class label """
        X = np.column_stack((np.ones(len(X)), X))
        prediction = np.around(self.hypothesis(X))
        return prediction


# ## Cross Validation
# 
# (10 points)
# 
# Use 5-fold cross validation in order to find the best eps and eta params from the given lists.
# 
# Shuffle the training set before you split the data to the folds.

# In[9]:


etas = [0.05, 0.005, 0.0005, 0.00005, 0.000005]
epss = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
shuffle_set = training_set.sample(frac=1)
Xshuffle, Yshuffle = shuffle_set[['x1', 'x2']].values, shuffle_set['y'].values
XFolds, YFolds = np.array_split(Xshuffle, 5), np.array_split(Yshuffle, 5)
pred, CV = 0, 0

for eta in etas:
    for eps in epss:
        for i in range(5):
            logisticRegression = LogisticRegressionGD(eta=eta, eps=eps)
            curX = np.concatenate(XFolds[:i] + XFolds[i+1:])
            curY = np.concatenate(YFolds[:i] + YFolds[i+1:])
            logisticRegression.fit(curX, curY)
            predictions = logisticRegression.predict(XFolds[i])
            for j in range(0, len(predictions)):
                if predictions[j] == YFolds[i][j]:
                    pred += 1
            if(pred / len(YFolds[i]) > CV):
                CV = pred / len(YFolds[i])
                epsBest = eps
                etaBest = eta
            pred = 0

print("Best eps: " + str(epsBest))
print("Best eta: " + str(etaBest))


# ## Normal distribution pdf
# 
# (5 Points)
# 
# Implement the normal distribution pdf 
# $$
# f(x;\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\big{(}\frac{x-\mu}{\sigma}\big{)}^2}
# $$
# Write an efficient vectorized code

# In[10]:


# calc normal pdf    
def norm_pdf(data, mu, sigma):
    squr_sigma = sigma ** 2
    return (1 / np.sqrt(2 * np.pi * (squr_sigma))) * np.exp(-(np.square(data - mu)) / (2 * (squr_sigma)))


# ## Expectation Maximization
# 
# (20 Points)
# 
# Implement the Expectation Maximization algorithm for gaussian mixture model.
# 
# The class should hold the distribution params.
# 
# Use -log likelihood as the cost function:
# $$
# cost(x) = \sum_{d=1}^{m}-log(w * pdf(x; \mu, \sigma))
# $$
# 
# \* The above is the cost of one gaussian. Think how to use the cost function for gaussian mixture.
# 
# Your class should contain the following functions:
# 1. init_params - initialize distribution params
# 1. expectation - calculating responsibilities
# 1. maximization - updating distribution params
# 1. fit - the learning function
# 1. get_dist_params - return the distribution params
# 
# \* You can add more functions if you think this is necessary
# 
# Don't change the eps params (eps=0.01)
# 
# When you need to calculate the pdf of a normal distribution use the function `norm_pdf` that you implemented above.

# In[11]:


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """
    
    def __init__(self, k=1, n_iter=1000, eps=0.01):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps

    def init_params(self, data):
       # initialzie a set for each parameter, targeted for the updation step
    
        dat = np.split(data, self.k)
        w_ = np.array([])
        mu_ = np.array([])
        sigma_ = np.array([])
        for i in range(self.k):
            w_ = np.append(w_, [1 / self.k])
            mu_ = np.append(mu_, [np.mean(dat[i])])
            sigma_ = np.append(sigma_, [np.std(dat[i])])

        self.w_ = w_
        self.mu_ = mu_
        self.sigma_ = sigma_
    
    def expectation(self, data):
        """
        E step - calculating responsibilities
        """
         # initialize a set of responsibilities
        responsibilities = np.zeros(shape=(len(data), self.k)) 
        
        # calcualte the responsibilities according to the normal distribution
        for i in range(self.k):
            for j in range(len(data)):
                responsibilities[j][i] = self.w_[i] * norm_pdf(data[j], self.mu_[i], self.sigma_[i])
                
         # responsibilities completion
        for i in range(len(data)):
            totalprob = np.sum(responsibilities[i])
            for j in range(self.k):
                responsibilities[i][j] /= totalprob

        return responsibilities

    def maximization(self, data):
        """
        M step - updating distribution params
        """
        responsibilities = self.expectation(data)
        neww = responsibilities.sum(axis=0) * (1 / len(responsibilities))
        newmu = np.zeros(shape=(self.k))
        newsigma = np.zeros(shape=(self.k))
        
        # re-estimating the parameters using the responsibilties
        for i in range(self.k):
            newmu[i] = np.sum(responsibilities[:, i] * data) * (1 / (len(responsibilities) * neww[i]))
            newsigma[i] = np.sum(responsibilities[:, i] * (np.square(data - newmu[i])))
            newsigma[i] = np.sqrt(newsigma[i] * (1 / (len(responsibilities) * neww[i])))

        return neww, newmu, newsigma
    
    def fit(self, data):
        """ 
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params 
        for the distribution. 
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        self.init_params(data)
        diff = np.infty
        prevCost = np.zeros(self.k)
        cost_hist2 = []
        itr = 0
        while itr < self.n_iter and diff > self.eps:  # until convergence (max likelihood)
            ww, mumu, ssigma = self.maximization(data)
            for i in range(self.k):
                cost = np.sum(-1 * np.log(ww[i] * norm_pdf(data, mumu[i], ssigma[i])))
                cost_hist2.append(cost)

            diff = np.max([abs(np.array(cost) - np.array(prevCost))])
            itr += 1

            prevCost = cost
            self.w_, self.mu_, self.sigma_ = ww, mumu, ssigma

    def get_dist_params(self):
        return self.mu_, self.sigma_


# ## Naive Bayes
# 
# (20 Points)
# 
# Implement the Naive Bayes algorithm.
# 
# For calculating the likelihood use the EM algorithm that you implemented above to find the distribution params. With these params you can calculate the likelihood probability.
# 
# Calculate the prior probability directly from the training set.
# 
# Your class should contain the following functions:
# 1. fit - the learning function
# 1. predict - the function for predicting an instance (or instances) after the fit function was executed
# 
# \* You can add more functions if you think this is necessary
# 

# In[12]:


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """
    
    def __init__(self, k=1):
        self.k = k
        self.prior = [0, 0] # we have two classes and we define two model objects, Respectively
        self.EM1 = [EM(k=self.k) for i in range(2)]
        self.EM2 = [EM(k=self.k) for i in range(2)]

    def fit(self, X, y):
        """ 
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        
        for labels in range(2):
            data = X[np.where(np.column_stack([np.zeros_like(y), y])[:, -1] == labels)]
            self.prior[labels] = len(data) / len(X)
            self.EM1[labels].fit(data[:, 0])
            self.EM2[labels].fit(data[:, 1])
            
    def likelihood(self, X, value):
         # returns the likelihood of the instance
        x1 = 0
        x2 = 0
        for i in range(self.k):
            x1 += self.EM1[value].w_[i] * norm_pdf(X[0], self.EM1[value].mu_[i], self.EM1[value].sigma_[i])
            x2 += self.EM2[value].w_[i] * norm_pdf(X[1], self.EM2[value].mu_[i], self.EM2[value].sigma_[i])
        
        return x1 * x2
    
    def posterior(self, X, value):
        # return the posterior of the instance
        return self.likelihood(X, value) * self.prior[value]

    def predict(self, X):
        """Return the predicted class label"""
        predictions = []
        for i in range(len(X)): # posterior probability check
            first = self.posterior(X[i], 0)
            second = self.posterior(X[i], 1)
            if (first > second):
                predictions.append(0)
            else:
                predictions.append(1)

        return np.asarray(predictions)


# ## Model evaluation
# 
# (10 points)
# 
# In this section you will build 2 models and fit them to 2 datasets
# 
# First 1000 training points and first 500 test points:
# 1. Use the first 1000 points from the training set (take the first original 1000 points - before the shuffle) and the first 500 points from the test set.
# 1. Fit Logistic Regression model with the best params you found earlier.
# 1. Fit Naive Bayes model. Remember that you need to select the number of gaussians in the EM.
# 1. Print the training and test accuracies for each model.
# 1. Use the `plot_decision_regions` function to plot the decision boundaries for each model (for this you need to use the training set as the input)
# 1. Plot the cost Vs the iteration number for the Logistic Regression model
# 
# Use all the training set points:
# 1. Repeat sections 2-6 for all the training set points

# In[13]:


X_Train, y_Train = X_training[:1000], y_training[:1000]
X_Test, y_Test = X_test[:500], y_test[:500]

# creating the relevant objects to prediction 
loR = LogisticRegressionGD(eta=etaBest, eps=epsBest)
loR.fit(X_Train, y_Train)
naiveBayes = NaiveBayesGaussian()
naiveBayes.fit(X_Train, y_Train)

loR_trnAcc, loR_tstAcc = 0, 0
loR_trnPred = loR.predict(X_Train)
loR_tstPred = loR.predict(X_Test)
naive_trnAcc, naive_tstAcc = 0, 0
naive_trnPred = naiveBayes.predict(X_Train)
naive_tstPred = naiveBayes.predict(X_Test)

# counting the true predictions 

for i in range(len(loR_tstPred)):
    if  loR_tstPred[i] == y_Test[i]:
        loR_tstAcc += 1
    if naive_tstPred[i] == y_Test[i]:
        naive_tstAcc += 1
        
for i in range(len(loR_trnPred)):
    if loR_trnPred[i] == y_Train[i]:
        loR_trnAcc += 1
    if naive_trnPred[i] == y_Train[i]:
        naive_trnAcc += 1
        
loR_tstAcc /= len(y_Test)
loR_trnAcc /= len(y_Train)
naive_tstAcc /= len(y_Test)
naive_trnAcc /= len(y_Train)

print("Test set accuracy check: LOR - " + str(loR_tstAcc) + " ,NaiveBayes - " + str(naive_tstAcc))
print("NaiveBayes model test set Accuracy: LOR - " + str(loR_trnAcc) + " NaiveBayes - " +str(naive_trnAcc))

plt.figure()
plt.title("LOR - first 1000 training samples")
plot_decision_regions(X_Train, y_Train, loR)

print()
plt.figure()
plt.title("NaiveBayes - first 1000 training samples");
plot_decision_regions(X_Train, y_Train, naiveBayes)

print()
plt.figure()
plt.title("Cost as a function of iterations")
plt.plot(list(range(len(loR.cost_hist))), loR.cost_hist)
plt.xlabel("Iterations")
plt.ylabel('Loss')
plt.show()


# In[14]:


X_Train = X_training.copy()
y_Train = y_training.copy()
X_Test = X_test.copy()
y_Test = y_test.copy()
# creating the relevant objects to prediction 
loR = LogisticRegressionGD(eta=etaBest, eps=epsBest)
loR.fit(X_Train, y_Train)
naiveBayes = NaiveBayesGaussian(k=4)
naiveBayes.fit(X_Train, y_Train)

loR_trnAcc, loR_tstAcc = 0, 0
loR_trnPred, loR_tstPred = loR.predict(X_Train), loR.predict(X_Test)
 
naive_trnAcc = 0
naive_tstAcc = 0
naive_trnPred = naiveBayes.predict(X_Train)
naive_tstPred  = naiveBayes.predict(X_Test)
# counting the true predictions 
        
for i in range(len(loR_tstPred)):
    if  loR_tstPred[i] == y_Test[i]:
        loR_tstAcc += 1
    if naive_tstPred[i] == y_Test[i]:
        naive_tstAcc += 1

for i in range(len(loR_trnPred)):
    if loR_trnPred[i] == y_Train[i]:
        loR_trnAcc += 1
    if naive_trnPred[i] == y_Train[i]:
        naive_trnAcc += 1
        
loR_tstAcc /= len(y_Test)
loR_trnAcc /= len(y_Train)
naive_tstAcc /= len(y_Test)
naive_trnAcc /= len(y_Train)

print("Test set accuracy check: LOR - " + str(loR_tstAcc) + ", NaiveBayes - " + str(naive_tstAcc))
print("NaiveBayes model test set Accuracy: LOR - " + str(loR_trnAcc) + ", NaiveBayes - " +str(naive_trnAcc))

plt.figure()
plt.title("LOR - first 1000 training samples")
plot_decision_regions(X_Train, y_Train, loR)

print()
plt.figure()
plt.title("NaiveBayes - first 1000 training samples");
plot_decision_regions(X_Train, y_Train, naiveBayes)

print()
plt.figure()
plt.title("Cost as a function of iterations")
plt.plot(list(range(len(loR.cost_hist))), loR.cost_hist)
plt.xlabel("Iterations")
plt.ylabel('Loss')
plt.show()


# ## Open question
# 
# (7 points) 
# 
# Will Full Bayes get better results comparing to Naive Bayes on the following dataset? Explain. 

# In[15]:


mean = [[-2,5], [-2, 12], [4, 12], [4, 5]]
cov = [[[1,0.2],[0.2,2]], [[1,0],[0,2]], [[1,0.2],[0.2,2]], [[1,-0.2],[-0.2,2]]]
x1 = []
x2 = []
y = []
size = [500, 500, 500, 500]
c = ['b', 'r']
for i in range(4):
    xx1,xx2 = np.random.multivariate_normal(mean[i],cov[i],size[i]).T
    x1.extend(xx1)
    x2.extend(xx2)
    y.extend([i%2] * size[i])
    plt.scatter(xx1,xx2, marker='.', c=c[i%2])
plt.axis('equal')
plt.show()


# #### Your answer here ####
# 

# It won't - 
# Naive Bayes gets similiar max posterior of each instance to the Full Bayes. Full Bayes will just be more calculations.
# 
# 

# In[ ]:





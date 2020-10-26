#!/usr/bin/env python
# coding: utf-8

# # Exercise 3: MAP Classifier
# 
# In this assignment you will implement a few of the MAP classifiers learned in class.
# 
# ## Read the following instructions carefully:
# 
# 1. This jupyter notebook contains all the step by step instructions needed for this part of the exercise.
# 2. Write vectorized code whenever possible.
# 3. You are responsible for the correctness of your code and should add as many tests as you see fit. Tests will not be graded nor checked.
# 4. Write your functions in this notebook only.
# 5. You are allowed to use functions and methods from the [Python Standard Library](https://docs.python.org/3/library/) and [numpy](https://www.numpy.org/devdocs/reference/) only. 
# 6. Your code must run without errors. During the environment setup, you were given a specific version of `numpy` to install. Changes of the configuration we provided are at your own risk. Code that cannot run will also earn you the grade of 0.
# 7. Write your own code. Cheating will not be tolerated. 
# 8. Submission includes this notebook and the answers to the theoretical part. Answers to qualitative questions should be written in markdown cells (with $\LaTeX$ support).
# 9. You can add additional functions.
# 10. Submission: zip only the completed jupyter notebook and the PDF with your solution for the theory part. Do not include the data or any directories. Name the file `ID1_ID2.zip` and submit **only one copy of the assignment**.
# 
# ## In this exercise you will perform the following:
# 1. Implement a Naive Bayeas Classifier based on Multi Normal distribution
# 1. Implement a Full Bayes Classifier based on Multi-Normal distribution
# 1. Implement a Distcrete Naive Bayes Classifier.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 
# # Normal Naive Bayes Classifier Vs Normal Full Bayes Classifier
# In the following section we are going to compare 2 models on a given dataset. <br>
# The 2 classifiers we are going to test are:
# 1. Naive Bayes classifer.<br>
# 1. Full Bayes classifier.<br>
# Recall that a Naive Bayes classifier makes the following assumption :<br> 
# ## $$ p(x_1, x_2, ..., x_n|A_j) = \Pi p(x_i | A_j) $$
# But the full Bayes classifier will not make this assumption.<br>

# ### The Data Story

# In a faraway land called **Randomistan** there is a rare animal called the **Randomammal**.<br> 
# We have gathered data about this unique animal to help the **randomian** researchers in observing this beast. <br>
# For a 1000 days straight we have measured the temperature and the humidity in Randomistan and whether the Randomammal was spotted or not. <br>
# The well known randomian **Bob** is a bit of a lazy researcher so he likes to keep things simple, and so he assumes that the temperature and the humidity are independent given the class. <br>
# **Alice** on the other hand is a hard working researcher and does not make any assumptions, she's young and is trying to gain some fame in the randomian community.
# 
# The dataset contains 2 features (**Temperature**, **Humidity**) alongside a binary label (**Spotted**) for each instance.<br>
# 
# We are going to test 2 different classifiers :
# * Naive Bayes Classifier (Bob)
# * Full Bayes Classifier. (Alice)
# 
# Both of our researchers assume that our features are normally distributed. But while Bob with his Naive classifier will assume that the features are independent, Alice and her Full Bayes classifier will not make this assumption.<br><br>
# Let's start off by loading the data (train, test) into a pandas dataframe and then converting them
# into numpy arrays.<br>
# The datafiles are :
# - randomammal_train.csv
# - randomammal_test.csv

# In[2]:


# Load the train and test set into a pandas dataframe and convert them into a numpy array.
train_set = pd.read_csv('randomammal_train.csv').values
test_set = pd.read_csv('randomammal_test.csv').values


# # Data Visualization
# Draw a scatter plot of the training data where __x__=Temerature and **y**=Humidity. <br>
# Use color to distinguish points from different classes.<br>
# Stop for a minute to think about Alice and Bob's approaches and which one you expect to work better.

# In[3]:



X_1 = (train_set[train_set[:,-1] == 1])[:,0]
X_0 = (train_set[train_set[:,-1] == 0])[:,0]

y_1 = (train_set[train_set[:,-1] == 1])[:,1]
y_0 = (train_set[train_set[:,-1] == 0])[:,1]

plt.plot(X_1, y_1, 'ro', ms=1, color = 'b')
plt.plot(X_0, y_0, 'ro', ms =1, color = 'r')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Training Data')
plt.legend(['class = 1', 'class = 0'])
plt.show()


# ## Bob's Naive Model
# 
# Start with implementing the [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) probability density function in the next cell: 
# $$ \frac{1}{\sqrt{2\pi \sigma^2}} \cdot e ^{-\frac{(x - \mu)^2}{2\sigma^2}} $$
# Where :
# * $\mu$ is the distribution mean.
# * $\sigma$ is the distribution standard deviation.

# Recall that when using the naive assumption, we assume our features are indepenent given the class. Meaning:
# $$ P(x_1, x_2 | Y) = p(x_1 | Y) \cdot p(x_2 | Y)$$
# 
# 
# Since we assume our features are normally distributed we need to find the mean and std for each feature in order for us to compute those probabilites. Implement the **NaiveNormalClassDistribution** in the next cell and build a distribution object for each class.

# In[4]:


def normal_pdf(x, mean, std):

    std_square = np.square(std) 
    e_power = -((np.square(x - mean)) / (2 * std_square))
    denomin = np.sqrt (2 * np.pi * std_square)
    
    return (np.power(np.e, e_power)) * (1 / denomin)
    
class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):

        self.dataset = dataset
        self.class_value = class_value
        self.classmates = self.dataset[self.dataset[:,-1] == class_value][:, :-1] # array of "similiar-classed" instances
        self.mean = np.mean(self.classmates, axis = 0)
        self.std = np.std(self.classmates, axis = 0)
        
    def get_prior(self):

        # size of class divided by size of data
        return self.classmates.shape[0] / self.dataset.shape[0] 
    
    def get_instance_likelihood(self, x):

        # gather the probabilty of every feature and make them one product - the likelihood
        prob_array = [normal_pdf(x[i], self.mean[i], self.std[i]) for i in range(len(self.classmates[0]))] # gather probabilty of every feature
        likelihood = np.prod(prob_array)
        return likelihood
        
    def get_instance_posterior(self, x):

        # posterior is P(A)*P(x|A) - i.e.( prior )*( likelihood )
        return (self.get_prior()) * (self.get_instance_likelihood(x))


# In[5]:


# Build the a NaiveNormalClassDistribution for each class.
naive_normal_CD_0 = NaiveNormalClassDistribution(train_set, 0)
naive_normal_CD_1 = NaiveNormalClassDistribution(train_set, 1)


# Implement the **MAPClassifier** class and build a MAPClassifier object contating the 2 distribution objects you just made above.

# In[6]:


class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    
    def predict(self, x):
        
        # compares between the probabilties in order to classify
        posterior_0 = self.ccd0.get_instance_posterior(x)
        posterior_1 = self.ccd1.get_instance_posterior(x)
        
        if posterior_0 > posterior_1:
            return 0
        else:
            return 1


# In[7]:


naive_normal_classifier = MAPClassifier(naive_normal_CD_0, naive_normal_CD_1)


# ### Evaluate model
# Implement the **compute_accuracy** function in the next cell. Use it and the 2 distribution objects you created to compute the accuracy on the test set.

# In[8]:


def compute_accuracy(testset, map_classifier):
    
    # check each instance's class in the test set and count the succesfull predictions
  
    correct_pred = 0
    for instance in testset :
        pred = map_classifier.predict(instance)
        if pred == instance[-1]:
            correct_pred += 1
    
    #accuracy = #correctly classified / #testset size  
    return (correct_pred / testset.shape[0])


# In[9]:


# Compute the naive model accuracy and store it in the naive accuracy variable.
naive_accuracy = compute_accuracy(test_set, naive_normal_classifier)
naive_accuracy


# ## Alice's Full Model
# 
# Start with Implementing the [multivariate normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) distribution probability density function in the next cell.
# 
# ## $$ (2\pi)^{-\frac{d}{2}} det(\Sigma )^{-\frac{1}{2}} \cdot e ^{-\frac{1}{2}(x-\mu)^T \Sigma ^ {-1} (x - \mu) }$$
# 
# Where : 
# * $\mu$ is the distribution mean vector. (length 2 in our case)
# * $\Sigma$ Is the distribution covarince matrix. (size 2x2 in our case)

# In the full bayes model we will not make any simplyfing assumptions, meaning, we will use a multivariate normal distribution. <br>
# And so, we'll need to compute the mean of each feature and to compute the covariance between the features to build the covariance matrix.
# Implement the **MultiNormalClassDistribution** and build a distribution object for each class.

# In[10]:


def multi_normal_pdf(x, mean, cov):
    
    d = len(cov)
    fin_vector = x - mean
    inv_matrix = np.linalg.inv(cov)
    det = np.linalg.det(cov) ** -0.5
    exponent = -0.5 * np.dot(np.transpose(fin_vector), np.dot(inv_matrix, fin_vector))

    return np.power(2 * np.pi,(-d / 2)) * det * np.exp(exponent) 

class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
       """
        
        self.dataset = dataset
        self.class_value = class_value
        self.classmates = self.dataset[self.dataset[:,-1] == class_value][:, :-1] # array of "similiar-classed" instances
        self.mean = np.mean(self.classmates, axis = 0) 
        self.cov = np.cov(self.classmates, rowvar = False)
    def get_prior(self):
        
         return self.classmates.shape[0] / self.dataset.shape[0]
    
    def get_instance_likelihood(self, x):
      
        likelihood =  multi_normal_pdf(x[:-1], self.mean, self.cov)
        return likelihood
    def get_instance_posterior(self, x):
        
        
        # posterior is P(A)*P(x|A) - i.e.( prior )*( likelihood )
        return (self.get_prior()) * (self.get_instance_likelihood(x))


# In[11]:


# Build the a MultiNormalClassDistribution for each class.
multi_normal_CD_0 = MultiNormalClassDistribution(train_set, 0)
multi_normal_CD_1 = MultiNormalClassDistribution(train_set, 1)


# build a MAPClassifier object contating the 2 distribution objects you just made above.

# In[12]:


multi_normal_classifier = MAPClassifier(multi_normal_CD_0, multi_normal_CD_1)


# ### Evaluate model
# Use the **compute_accuracy** function and the 2 distribution objects you created to compute the accuracy on the test set.

# In[13]:


# Compute the naive model accuracy and store it in the naive accuracy variable.
full_accuracy = compute_accuracy(test_set, multi_normal_classifier)
full_accuracy


# ## Results

# Use a plot bar to showcase the models accuracy.

# In[14]:


# Bar plot of accuracy of each model side by side.
plt.bar(x=['Naive', 'Full'], height=[naive_accuracy, full_accuracy])
plt.title("Naive vs Full accuracy comparison")
plt.ylabel("Accuracy")


# # Discrete Naive Bayes Classifier 

# We will now build a discrete naive Bayes based classifier using **Laplace** smoothing.
# In the recitation, we saw how to compute the probability for each attribute value under each class:

# $$ P(x_j | A_i) = \frac{n_{ij} + 1}{n_i + |V_j|} $$
# Where:
# * $n_{ij}$ The number of training instances with the class $A_i$ and the value $x_j$ in the relevant attribute.
# * $n_i$ The number of training instances with the class $A_i$
# * $|V_j|$ The number of possible values of the relevant attribute.
# 
# In order to compute the likelihood we assume:
# $$ P(x| A_i) = \prod\limits_{j=1}^{n}P(x_j|A_i) $$
# 
# And to classify an instance we will choose : 
# $$\arg\!\max\limits_{i} P(A_i) \cdot P(x | A_i)$$
# 

# ## Data
# We will try to predict breast cancer again only this time from a different dataset, 
# <br> you can read about the dataset here : [Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer)<br>
# Load the training set and test set provided for you in the data folder.
#  - breast_trainset.csv
#  - breast_testset.csv
#  

# In[16]:


# Load the train and test set into a pandas dataframe and convert them into a numpy array.
train_set = pd.read_csv('breast_trainset.csv').values
test_set = pd.read_csv('breast_testset.csv').values


# ## Build A Discrete Naive Bayes Distribution for each class
# Implement the **DiscreteNBClassDistribution** in the next cell and build a distribution object for each class.

# In[17]:


EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
       
        self.dataset = dataset
        self.class_value = class_value
        self.classmates = self.dataset[self.dataset[:,-1] == class_value][:, :-1] # array of "similiar-classed" instances
    
    def get_prior(self):
      
        return self.classmates.shape[0] / self.dataset.shape[0]
    
    def get_instance_likelihood(self, x):
        
        likelihood = 1
        laplace_est = 0
        
        # for each value xj - loop through the data to retrieve the relevant numbers for laplace
        for j in self.classmates[0][:-1]:
            n_i = self.classmates.shape[0]
            n_i_j = len(self.classmates[self.classmates[:, j] == x[j]])
            v_j = len(np.unique(self.dataset[:, j]))
            
            if n_i_j == 0:
                    laplace_est = EPSILLON
            else:
                    laplace_est = (1 + n_i_j) / (n_i + v_j)
            likelihood *= laplace_est
        
        return likelihood
    
    def get_instance_posterior(self, x):
        
        return (self.get_prior()) * (self.get_instance_likelihood(x))


# In[18]:


discrete_naive_CD_0 = DiscreteNBClassDistribution(train_set, 0)
discrete_naive_CD_1 = DiscreteNBClassDistribution(train_set, 1)


# build a MAPClassifier object contating the 2 distribution objects you just made above.

# In[19]:


discrete_naive_classifier = MAPClassifier(discrete_naive_CD_0, discrete_naive_CD_1)


# Use the **compute_accuracy** function and the 2 distribution objects you created to compute the accuracy on the test set.

# In[20]:


compute_accuracy(test_set, discrete_naive_classifier)


# In[ ]:





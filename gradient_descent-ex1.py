#!/usr/bin/env python
# coding: utf-8

# # Exercise 1: Linear Regression
# 
# ### This notebook is executed automatically. Failure to comply with the following instructions will result in a massive penalty. Appeals regarding your failure to read the following instructions will be denied. Kindly reminder: the homework assignments grade is 50% of the final grade. 
# 
# ### Do not start the exercise until you fully understand the submission guidelines.
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
# 8. Submission includes this notebook only with the exercise number and your ID as the filename. For example: `hw1_123456789_987654321.ipynb` if you submitted in pairs and `hw1_123456789.ipynb` if you submitted the exercise alone.
# 9. Answers to qualitative questions should be written in **markdown** cells (with $\LaTeX$ support). Answers that will be written in commented code blocks will not be checked.
# 
# ## In this exercise you will perform the following:
# 1. Load a dataset and perform basic data exploration using a powerful data science library called [pandas](https://pandas.pydata.org/pandas-docs/stable/).
# 2. Preprocess the data for linear regression.
# 3. Compute the cost and perform gradient descent in pure numpy in vectorized form.
# 4. Fit a linear regression model using a single feature.
# 5. Visualize your results using matplotlib.
# 6. Perform multivariate linear regression.
# 7. Pick the best three features in the dataset.
# 

# # I have read and understood the instructions:203378039

# In[1]:


import numpy as np # used for scientific computing
import pandas as pd # used for data analysis and manipulation
import matplotlib.pyplot as plt # used for visualization and plotting

np.random.seed(42) 

# make matplotlib figures appear inline in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (14.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# ## Part 1: Data Preprocessing (10 Points)
# 
# For the following exercise, we will use a dataset containing housing prices in King County, USA. The dataset contains 5,000 observations with 18 features and a single target value - the house price. 
# 
# First, we will read and explore the data using pandas and the `.read_csv` method. Pandas is an open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

# In[2]:


# Read comma separated data
df = pd.read_csv('data.csv') # Make sure this cell runs regardless of your absolute path.
# df stands for dataframe, which is the default format for datasets in pandas


# ### Data Exploration
# A good practice in any data-oriented project is to first try and understand the data. Fortunately, pandas is built for that purpose. Start by looking at the top of the dataset using the `df.head()` command. This will be the first indication that you read your data properly, and that the headers are correct. Next, you can use `df.describe()` to show statistics on the data and check for trends and irregularities.

# In[3]:


df.head(5)


# In[4]:


df.describe()


# We will start with one variable linear regression by extracting the target column and the `sqft_living` variable from the dataset. We use pandas and select both columns as separate variables and transform them into a numpy array.

# In[5]:


X = df['sqft_living'].values
y = df['price'].values


# ## Preprocessing
# 
# As the number of features grows, calculating gradients gets computationally expensive. We can speed this up by normalizing the input data to ensure all values are within the same range. This is especially important for datasets with high standard deviations or differences in the ranges of the attributes. Use [mean normalization](https://en.wikipedia.org/wiki/Feature_scaling) for the fearures (`X`) and the true labels (`y`).
# 
# Implement the cost function `preprocess`.

# In[6]:


def preprocess(X, y):
   
    X = (X - X.mean(axis = 0)) / (X.max(axis = 0) - X.min(axis = 0)) # Perform mean normalization on the features
    y = (y - y.mean()) / (y.max() - y.min()) # Perform mean normalization on the target values
    
    return X, y


# In[7]:


X, y = preprocess(X, y)
print(X)


# We will split the data into two datasets: 
# 1. The training dataset will contain 80% of the data and will always be used for model training.
# 2. The validation dataset will contain the remaining 20% of the data and will be used for model evaluation. For example, we will pick the best alpha and the best features using the validation dataset, while still training the model using the training dataset.

# In[8]:


# training and validation split
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
X_train, X_val = X[idx_train], X[idx_val]
y_train, y_val = y[idx_train], y[idx_val]


# ## Data Visualization
# Another useful tool is data visualization. Since this problem has only two parameters, it is possible to create a two-dimensional scatter plot to visualize the data. Note that many real-world datasets are highly dimensional and cannot be visualized naively. We will be using `matplotlib` for all data visualization purposes since it offers a wide range of visualization tools and is easy to use.

# In[9]:


plt.plot(X_train, y_train, 'ro', ms=1, mec='b') # the parameters control the size, shape and color of the scatter plot
plt.ylabel('Price in USD')
plt.xlabel('sq.ft')
plt.show()


# ## Bias Trick
# 
# Make sure that `X` takes into consideration the bias $\theta_0$ in the linear model. Hint, recall that the predications of our linear model are of the form:
# 
# $$
# \hat{y} = h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1
# $$
# 
# Add columns of ones as the zeroth column of the features (do this for both the training and validation sets).

# In[10]:


X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
X_val = np.column_stack((np.ones(X_val.shape[0]), X_val))
print(X_train)


# ## Part 2: Single Variable Linear Regression (40 Points)
# Simple linear regression is a linear regression model with a single explanatory varaible and a single target value. 
# 
# $$
# \hat{y} = h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1
# $$
# 
# ## Gradient Descent 
# 
# Our task is to find the best possible linear line that explains all the points in our dataset. We start by guessing initial values for the linear regression parameters $\theta$ and updating the values using gradient descent. 
# 
# The objective of linear regression is to minimize the cost function $J$:
# 
# $$
# J(\theta) = \frac{1}{2m} \sum_{i=1}^{n}(h_\theta(x^{(i)})-y^{(i)})^2
# $$
# 
# where the hypothesis (model) $h_\theta(x)$ is given by a **linear** model:
# 
# $$
# h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1
# $$
# 
# $\theta_j$ are parameters of your model. and by changing those values accordingly you will be able to lower the cost function $J(\theta)$. One way to accopmlish this is to use gradient descent:
# 
# $$
# \theta_j = \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
# $$
# 
# In linear regresion, we know that with each step of gradient descent, the parameters $\theta_j$ get closer to the optimal values that will achieve the lowest cost $J(\theta)$.

# Implement the cost function `compute_cost`. (10 points)

# In[11]:


def compute_cost(X, y, theta):
    
    # Inner product of - features vector & weights(parameters) vector
    innr_prdct = X.dot(theta)                        
    sqr_error= np.square(np.subtract(innr_prdct, y)) # Computes the squared difference 
    J = np.sum(sqr_error) / (2 * len(X))

# Returning The cost associated with the current set of parameters
    return J


# In[12]:


theta = np.array([-1, 2])
J = compute_cost(X_train, y_train, theta)


# Implement the gradient descent function `gradient_descent`. (10 points)

# In[13]:


def gradient_descent(X, y, theta, alpha, num_iters):

    J_history = [] # Use a python list to save cost in every iteration
    theta = theta.copy() # avoid changing the original thetas
    m = len(X)
    
    # Starting the gradient descent iterations
    for itr in range(num_iters):
        h_theta = X.dot(theta)
        theta = theta - alpha * (1 / m) * X.T.dot(h_theta - y)
        J_history.append(compute_cost(X,y,theta))
    return theta, J_history


# In[14]:


np.random.seed(42)
theta = np.random.random(size=2)
iterations = 40000
alpha = 0.1
theta, J_history = gradient_descent(X_train ,y_train, theta, alpha, iterations)


# You can evaluate the learning process by monitoring the loss as training progress. In the following graph, we visualize the loss as a function of the iterations. This is possible since we are saving the loss value at every iteration in the `J_history` array. This visualization might help you find problems with your code. Notice that since the network converges quickly, we are using logarithmic scale for the number of iterations. 

# In[15]:


plt.plot(np.arange(iterations), J_history)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations')
plt.show()


# Implement the pseudo-inverse function `pinv`. **Do not use `np.linalg.pinv`**, instead use only direct matrix multiplication as you saw in class (you can calculate the inverse of a matrix using `np.linalg.inv`). (10 points)

# In[16]:


def pinv(X, y):
    
    pinv_theta = []
    pinv_theta = (np.linalg.inv(X.T.dot(X))).dot(X.T) # Acoording the formula - inversing and transposing
    pinv_theta = pinv_theta.dot(y)
    return pinv_theta


# In[17]:


theta_pinv = pinv(X_train ,y_train)
J_pinv = compute_cost(X_train, y_train, theta_pinv)


# We can add the loss value for the theta calculated using the psuedo-inverse to our graph. This is another sanity check as the loss of our model should converge to the psuedo-inverse loss.

# In[18]:


plt.plot(np.arange(iterations), J_history)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations')
plt.hlines(y = J_pinv, xmin = 0, xmax = len(J_history), color='r',
           linewidth = 1, linestyle = 'dashed')
plt.show()


# We can use a better approach for the implementation of `gradient_descent`. Instead of performing 40,000 iterations, we wish to stop when the improvement of the loss value is smaller than `1e-8` from one iteration to the next. Implement the function `efficient_gradient_descent`. (5 points)

# In[19]:


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
   
    J_history = [] # Use a python list to save cost in every iteration
    theta = theta.copy() # avoid changing the original thetas
    
    
    # Starting the gradient descent iterations
    m = len(y)
    for itr in range(num_iters):
        h_theta = X.dot(theta)
        theta = theta - alpha * (1 / m) * X.T.dot(h_theta - y)
        J_history.append(compute_cost(X,y,theta))
        
    # Stop if improvement of the loss value is smaller than 1e-8
        if (not itr == 0) and (J_history[itr-1] - J_history[itr] < 1e-8):
            return theta, J_history
    return theta, J_history


# The learning rate is another factor that determines the performance of our model in terms of speed and accuracy. Complete the function `find_best_alpha`. Make sure you use the training dataset to learn the parameters (thetas) and use those parameters with the validation dataset to compute the cost.

# In[20]:


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
  
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}
    theta = np.random.random(size=2)
    
    # Creating the dictionary - 'alpha' as key, 'cost' as value
    for i in range(len(alphas)):
        temp, J_history = efficient_gradient_descent(X_train, y_train, theta, alphas[i], iterations)
        cost = compute_cost(X_val, y_val, temp)
        alpha_dict.update( { alphas[i] : cost } )
    
    return alpha_dict


# In[21]:


alpha_dict = find_best_alpha(X_train, y_train, X_val, y_val, 40000)


# Obtain the best learning rate from the dictionary `alpha_dict`. This can be done in a single line using built-in functions.

# In[22]:


best_alpha= min(alpha_dict, key=alpha_dict.get)
print(best_alpha)


# Pick the best three alpha values you just calculated and provide **one** graph with three lines indicating the training loss as a function of iterations (Use 10,000 iterations). Note you are required to provide general code for this purpose (no hard-coding). Make sure the visualization is clear and informative. (5 points)

# In[23]:


# Sorting and using the annonymous function in order to obtain the best three alphas
three_best = (sorted(alpha_dict.items(), key = lambda kv:(kv[1], kv[0])))[:3]

for i in range(3):
    temp_alpha = (three_best[i][0])
    np.random.seed(42)
    theta = np.random.random(size=2)
    iters = 10000
    theta, J_history =gradient_descent(X_train ,y_train, theta, temp_alpha, iters)
    plt.plot(np.arange(iters), J_history)

plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss of three best alphas')
plt.legend(['1st', '2nd', 'rd'])
plt.show()


# This is yet another sanity check. This function plots the regression lines of your model and the model based on the pseudoinverse calculation. Both models should exhibit the same trend through the data. 

# In[24]:


plt.figure(figsize=(7, 7))
plt.plot(X_train[:,1], y_train, 'ro', ms=1, mec='k')
plt.ylabel('Price in USD')
plt.xlabel('sq.ft')
plt.plot(X_train[:, 1], np.dot(X_train, theta), 'o')
plt.plot(X_train[:, 1], np.dot(X_train, theta_pinv), '-')

plt.legend(['Training data', 'Linear regression', 'Best theta']);


# ## Part 2: Multivariate Linear Regression (30 points)
# 
# In most cases, you will deal with databases that have more than one feature. It can be as little as two features and up to thousands of features. In those cases, we use a multiple linear regression model. The regression equation is almost the same as the simple linear regression equation:
# 
# $$
# \hat{y} = h_\theta(\vec{x}) = \theta^T \vec{x} = \theta_0 + \theta_1 x_1 + ... + \theta_n x_n
# $$
# 
# 
# If you wrote vectorized code, this part should be straightforward. If your code is not vectorized, you should go back and edit your functions such that they support both multivariate and single variable regression. **Your code should not check the dimensionality of the input before running**.

# In[25]:


# Read comma separated data
df = pd.read_csv('data.csv')
df.head()


# ## Preprocessing
# 
# Like in the single variable case, we need to create a numpy array from the dataframe. Before doing so, we should notice that some of the features are clearly irrelevant.

# In[26]:


X = df.drop(columns=['price', 'id', 'date']).values
y = df['price'].values


# Use the **same** `preprocess` function you implemented previously. Notice that proper vectorized implementation should work regardless of the dimensionality of the input. You might want to check that your code in the previous parts still works.

# In[27]:


# preprocessing
X, y = preprocess(X, y)


# In[28]:


# training and validation split
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
idx_train, idx_val = indices[:int(0.8*X.shape[0])], indices[int(0.8*X.shape[0]):]
X_train, X_val = X[idx_train,:], X[idx_val,:]
y_train, y_val = y[idx_train], y[idx_val]


# Using 3D visualization, we can still observe trends in the data. Visualizing additional dimensions requires advanced techniques we will learn later in the course.

# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
import mpl_toolkits.mplot3d.axes3d as p3
fig = plt.figure(figsize=(5,5))
ax = p3.Axes3D(fig)
xx = X_train[:, 1][:1000]
yy = X_train[:, 2][:1000]
zz = y_train[:1000]
ax.scatter(xx, yy, zz, marker='o')
ax.set_xlabel('bathrooms')
ax.set_ylabel('sqft_living')
ax.set_zlabel('price')
plt.show()


# Use the bias trick again (add a column of ones as the zeroth column in the both the training and validation datasets).

# In[30]:


X_train = np.column_stack((np.ones(X_train.shape[0]), X_train))
X_val = np.column_stack((np.ones(X_val.shape[0]), X_val))
print(X_train)


# Make sure the functions `compute_cost` (10 points), `gradient_descent` (15 points), and `pinv` (5 points) work on the multi-dimensional dataset. If you make any changes, make sure your code still works on the single variable regression model. 

# In[31]:


shape = X_train.shape[1]
theta = np.ones(shape)
J = compute_cost(X_train, y_train, theta)


# In[32]:


np.random.seed(42)
shape = X_train.shape[1]
theta = np.random.random(shape)
iterations = 40000
theta, J_history = gradient_descent(X_train ,y_train, theta, best_alpha, iterations)


# In[33]:


theta_pinv = pinv(X_train ,y_train)
J_pinv = compute_cost(X_train, y_train, theta_pinv)


# We can use visualization to make sure the code works well. Notice we use logarithmic scale for the number of iterations, since gradient descent converges after ~500 iterations.

# In[34]:


plt.plot(np.arange(iterations), J_history)
plt.xscale('log')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss as a function of iterations - multivariate linear regression')
plt.hlines(y = J_pinv, xmin = 0, xmax = len(J_history), color='r',
           linewidth = 1, linestyle = 'dashed')
plt.show()


# ## Part 3: Find best features for regression (20 points)
# 
# Adding additional features to our regression model makes it more complicated but does not necessarily improves performance. Find the combination of two features that best minimizes the loss. First, we will reload the dataset as a dataframe in order to access the feature names. Use the dataframe with the relevant features as the input to the `generate_couples` and obtain a list of all possible feature couples.

# In[35]:


columns_to_drop = ['price', 'id', 'date']
all_features = df.drop(columns=columns_to_drop)
all_features.head(5)


# In[36]:


import itertools

def generate_couples(features):

# Creating the python list of tuples to be returned, using combinations method
    couples = []
    couples = list(itertools.combinations(features, 2))
   
  
    return couples


# In[37]:


couples = generate_couples(all_features)
print("Number of couples: {}".format(len(couples)))


# Complete the function `find_best_couple`. You are free to use any arguments you need.

# In[38]:


def find_best_couple():

    # Define the parameters
    alpha = best_alpha
    theta = np.random.random(size = 3)
    min_cost = float ('inf')
        
    for couple in couples:
        
        # Collects training  and validation of current couple
        couple_train = X_train[:,[0,all_features.columns.get_loc(couple[0]) + 1,all_features.columns.get_loc(couple[1]) + 1]]
        couple_val = X_val[:,[0,all_features.columns.get_loc(couple[0]) + 1,all_features.columns.get_loc(couple[1]) + 1]]
        
        # Random theta
        np.random.seed(42)
        shape = couple_train.shape[1]
        theta = np.random.random(shape)
        
     # Computes cost and saves the result if there's an improvement
        temp_theta, J_history = efficient_gradient_descent(couple_train, y_train, theta, alpha,  40000)
        current_cost = compute_cost(couple_val, y_val, temp_theta)
        
        if current_cost < min_cost:
            min_cost = current_cost
            best_couple = list(couple)   
    return best_couple


# In[39]:


best_couple = find_best_couple()
print(best_couple)


# ### Backward Feature Selection
# 
# Complete the function `backward_selection`. Train the model with all but one of the features at a time and remove the worst feature. Next, remove an additional feature along the feature you previously removed. Repeat this process until you reach two features + bias. You are free to use any arguments you need.

# In[42]:


def backward_selection():

    np.random.seed(42)
    best_couples = None
    features = list(all_features.columns.values)
    X_train_copy = X_train
    X_val_copy = X_val
    
    while len(features) > 2:
        
        min_cost = float("inf")
        worst_feature = None
        for feature in features:
            
            ## Collects training & validation of current features
            worst_feature_train = np.delete(X_train_copy, features.index(feature) + 1, axis=1)
            worst_feature_val = np.delete(X_val_copy, features.index(feature) + 1, axis=1)
            
            # Random theta
            shape = worst_feature_train.shape[1]
            theta = np.random.random(shape)
            
            iterations = 40000
            # Computes error
            theta = efficient_gradient_descent(worst_feature_train ,y_train, theta, best_alpha, iterations)[0]
            current_cost = compute_cost(worst_feature_val, y_val, theta)
            
            # Determines worst feature out of current features
            if current_cost < min_cost:
                min_cost = current_cost
                worst_feature = feature
            
        # Removes unnecessary features from the list
        X_train_copy = np.delete(X_train_copy, features.index(worst_feature) + 1, axis=1)
        X_val_copy = np.delete(X_val_copy, features.index(worst_feature) + 1, axis=1)
        features.remove(worst_feature)
        
    best_couples = features
    return best_couples


# In[43]:


backward_selection()


# Give an explanations to the results. Do they make sense? How could you further improve this linear regression model?

# ## The result makes sense, since the size of the living room and the height of the house/apratment (indicates the potential view seen from the apartment, are key parameters for affecting potential buyers' decision. Should we improve this model , we would would use more features for comparing - best trio, best quartet. It could optimize accuartion
# 

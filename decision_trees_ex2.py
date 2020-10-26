#!/usr/bin/env python
# coding: utf-8

# # Exercise 2: Decision Trees
# 
# In this assignment you will implement a Decision Tree algorithm as learned in class.
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
# 1. Practice OOP in python.
# 2. Implement two impurity measures: Gini and Entropy.
# 3. Construct a decision tree algorithm.
# 4. Prune the tree to achieve better results.
# 5. Visualize your results.

# # I have read and understood the instructions:203378039

# In[1]:


import numpy as np
import pandas as pd
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


# ## Warmup - OOP in python
# 
# Our desicion tree will be implemented using a dedicated python class. Python classes are very similar to classes in Java.
# 
# 
# You can use the following [site](https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/) to learn about classes in python.

# In[2]:


class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, node):
        self.children.append(node)


# In[3]:


n = Node(5)
p = Node(6)
q = Node(7)
n.add_child(p)
n.add_child(q)
n.children


# ## Data preprocessing
# 
# For the following exercise, we will use a dataset containing mushroom data `agaricus-lepiota.csv`. 
# 
# This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family. Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous
# one (=there are only two classes **edible** and **poisonous**). 
#     
# The dataset contains 8124 observations with 22 features:
# 1. cap-shape: bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s
# 2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# 3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
# 4. bruises: bruises=t,no=f
# 5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
# 6. gill-attachment: attached=a,descending=d,free=f,notched=n
# 7. gill-spacing: close=c,crowded=w,distant=d
# 8. gill-size: broad=b,narrow=n
# 9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# 10. stalk-shape: enlarging=e,tapering=t
# 11. stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r
# 12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# 15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# 16. veil-type: partial=p,universal=u
# 17. veil-color: brown=n,orange=o,white=w,yellow=y
# 18. ring-number: none=n,one=o,two=t
# 19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
# 20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
# 21. population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
# 22. habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
# 
# First, we will read and explore the data using pandas and the `.read_csv` method. Pandas is an open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.

# In[4]:


# load dataset
data = pd.read_csv('agaricus-lepiota.csv')


# One of the advantages of the Decision Tree algorithm is that almost no preprocessing is required. However, finding missing values is always required.

# In[5]:


data.dropna(axis=1, how='any', thresh=None, subset=None, inplace=True)


# We will split the dataset to `Training` and `Test` sets

# In[6]:


from sklearn.model_selection import train_test_split
# Making sure the last column will hold the labels
X, y = data.drop('class', axis=1), data['class']
X = np.column_stack([X,y])
# split dataset using random_state to get the same split each time
X_train, X_test = train_test_split(X, random_state=99)

print("Training dataset shape: ", X_train.shape)
print("Testing dataset shape: ", X_test.shape)


# In[7]:


y.shape


# ## Impurity Measures
# 
# (5 points each - 10 points total)
# 
# Impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. Implement the functions `calc_gini` and `calc_entropy`. You are encouraged to test your implementation.

# In[8]:


def calc_gini(data):

    gini = 0.0
    y = data[:,-1:]
    num_of_classes = np.unique(y, return_counts=True)[1]
    
        #Gini formula
    impurity = np.square(num_of_classes / y.shape[0]).sum() 
    gini = 1 - impurity

    return gini


# In[9]:


def calc_entropy(data):
        
    entropy = 0.0
    y = data[:,-1:]
    num_of_classes = np.unique(y, return_counts=True)[1]
    
        #Entropy formula
    impurity = -1 * (num_of_classes / y.shape[0] * np.log2(num_of_classes / y.shape[0])).sum() 
    return impurity


# In[10]:


calc_gini(X), calc_entropy(X)


# ## Goodness of Split
# 
# (10 Points)
# 
# Given a feature the Goodnees of Split measures the reduction in the impurity if we split the data according to the feature.
# $$
# \Delta\varphi(S, A) = \varphi(S) - \sum_{v\in Values(A)} \frac{|S_v|}{|S|}\varphi(S_v)
# $$
# 
# In our implementation the goodness_of_split function will return either the Goodness of Split or the Gain Ratio as learned in class. You'll control the return value with the `gain_ratio` parameter. If this parameter will set to False (the default value) it will return the regular Goodness of Split. If it will set to True it will return the Gain Ratio.
# $$
# GainRatio(S,A)=\frac{InformationGain(S,A)}{SplitInformation(S,A)}
# $$
# Where:
# $$
# InformationGain(S,A)=Goodness\ of\ Split\ calculated\ with\ Entropy\ as\ the\ Impurity\ function \\
# SplitInformation(S,A)=- \sum_{a\in A} \frac{|S_a|}{|S|}\log\frac{|S_a|}{|S|}
# $$
# NOTE: you can add more parameters to the function and you can also add more returning variables (The given parameters and the given returning variable should not be touch).

# In[11]:


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):

    goodness = 0.0
    x = data[:, feature]
    diff_values = np.unique(x)
    weighted_average = 0.0;
    if gain_ratio:
        split_info = 0.0
        for value in diff_values:
            sub_data = data[data[:, feature] == value]
            
            #compute the weighted_average impurity after split using Entropy
            prob = sub_data.shape[0] / x.shape[0]
            sub_data = data[data[:, feature] == value]
            weighted_average += prob * calc_entropy(sub_data)
            
            #split information formula
            split_info -= prob * np.log2(prob)
            
        info_gain = calc_entropy(data) - weighted_average
        goodness = info_gain / split_info
    else:  
        
        #compute weighted_average impurity after split using Gini
        for value in diff_values:
            sub_data = data[data[:, feature] == value]
            prob = sub_data.shape[0] / x.shape[0]
            weighted_average += prob * impurity_func(sub_data)
        goodness = impurity_func(data) - weighted_average
    return goodness    


# ## Building a Decision Tree
# 
# (30 points)
# 
# Use a Python class to construct the decision tree. Your class should support the following functionality:
# 
# 1. Initiating a node for a decision tree. You will need to use several class methods and class attributes and you are free to use them as you see fit. We recommend that every node will hold the feature and value used for the split and its children.
# 2. Your code should support both Gini and Entropy as impurity measures. 
# 3. The provided data includes categorical data. In this exercise, when splitting a node create the number of children needed according to the attribute unique values.
# 
# Complete the class `DecisionNode`. The structure of this class is entirely up to you. 
# 
# Complete the function `build_tree`. This function should get the training dataset and the impurity as inputs, initiate a root for the decision tree and construct the tree according to the procedure you learned in class.

# In[38]:


class DecisionNode:

    # This class will hold everything you require to construct a decision tree.
    # The structure of this class is up to you. However, you need to support basic 
    # functionality as described above. It is highly recommended that you 
    # first read and understand the entire exercise before diving into this class.
    
    def __init__(self, data, feature = None, value = None , feature_val = None):
        self.feature = feature # column index of criteria being tested
        self.value = value
        self.data = data
        self.children = []
        self.depth = 0
        self.feature_val = feature_val
        
    def add_child(self, node):
        self.children.append(node)
        
    def is_leaf(self):
        return (len(self.children) == 0)
        


# In[39]:


import queue

# During the tree constructing -the "best attribute function" determines favorable attribute each step

def best_attribute(data, impurity_func, gain_ratio = False):
    best_goodness = -1.0
    best_feature_ind = 0
    for feature in range(data.shape[1] - 1):
        curr_goodness = goodness_of_split(data, feature, impurity_func, gain_ratio)
        if curr_goodness > best_goodness:
            best_goodness = curr_goodness
            best_feature_ind = feature
    return best_feature_ind, best_goodness

# the "chi square test" helps in pruning effectively, to acheive high accuracy in prediction 

def chi_square_test(node):
    num_of_each_class = np.unique(node.data[:, -1], return_counts = True)[1]
    diff_values, num_of_each_value = np.unique(node.data[:, node.feature], return_counts = True)
    p_0 = num_of_each_class[0] / len(node.data)
    p_1 = num_of_each_class[1] / len(node.data)
    diff_classes = np.unique(node.data[:, -1])
    chi_square = 0
    
    for i, value in enumerate(diff_values): # using enumerate built-in to handle easily dynamic sizes
        d_f = num_of_each_value[i]
        p_f = np.count_nonzero((node.data[:, node.feature] == value) & (node.data[:, -1] == diff_classes[0]))
        n_f = np.count_nonzero((node.data[:, node.feature] == value) & (node.data[:, -1] == diff_classes[1]))
        E_0 = d_f * p_0
        E_1 = d_f * p_1
        
        # the statistic formula
        chi_square += ((p_f - E_0)**2 / E_0) + ((n_f - E_1)**2 / E_1)
        
    return chi_square

def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
   
    #According to the manipulated constructor, defining the starting root
    root = None
    data_copy = data
    root = DecisionNode(data_copy)
    root.tree_depth = 0
    node_queue = queue.SimpleQueue()
    node_queue.put(root)
   
    # keep the tree growing until all leaves are pure
    while(not node_queue.empty()):
        curr_node = node_queue.get()
       
        # stop if reached maximum depth
        if curr_node.depth == max_depth:
            continue
        # stop if reached the last feature (the classification)
        elif curr_node.data.shape[1] == 1:
            continue
        
        # stop if the node is perfectly classified
        elif impurity(curr_node.data) == 0: 
            continue
      
        # use the best attribute function to assign best feature for the node
        best_feature_ind, best_goodness = best_attribute(curr_node.data, impurity, gain_ratio)
        curr_node.feature = best_feature_ind
        curr_node.value = best_goodness        
        
        # stop if acheived perfect impurity for the node
        if(best_goodness == 0):
            continue
            
        # create the children according to the best attribute's values
        sorted_feature_values = np.unique(curr_node.data[:, best_feature_ind])
        
        # chi square test
        if(chi != 1):
            chi_square = chi_square_test(curr_node)
            if chi_square < chi_table[len(sorted_feature_values) - 1][chi]:
                continue
        
        for value in sorted_feature_values:
            new_node = DecisionNode(curr_node.data[np.where(curr_node.data[:, best_feature_ind] == value)], feature_val = value)
            new_node.depth = curr_node.depth + 1
            root.tree_depth = max(root.tree_depth, new_node.depth)
            curr_node.add_child(new_node)
            node_queue.put(new_node)

    return root


# In[40]:


# python support passing a function as arguments to another function.
tree_gini = build_tree(data=X_train, impurity=calc_gini) # gini and goodness of split
tree_entropy = build_tree(data=X_train, impurity=calc_entropy) # entropy and goodness of split
tree_entropy_gain_ratio = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True) # entropy and gain ratio


# ## Tree evaluation
# 
# (10 points) 
# 
# Complete the functions `predict` and `calc_accuracy`. 

# In[41]:


def predict(node, instance):

# work thorough the decision tree to predict the intstance's class 
    prediction = None
    instance_copy = instance
    curr_node = node
    prev_node = None
    while len(curr_node.children) != 0 and prev_node != curr_node:
        found_child = False
        for i in curr_node.children:
          
        # if there's a value-match beetween a child and the current node, move to the next "question"
            if i.feature_val == instance_copy[curr_node.feature]: 
                prev_node = curr_node
                curr_node = i
                found_child = True
                break;
        if not found_child: break
        
    diff_values, num_each_value = np.unique(curr_node.data[:,-1], return_counts = True)
    
    #prediction according to the "highest score"
    prediction = diff_values[np.argmax(num_each_value)]
    return prediction


# In[42]:


def calc_accuracy(node, dataset):
 
    # checks how accurate is the prediction
    accuracy = 0
    num_of_successes = 0
    for instance in dataset:
        prediction = predict(node, instance) 
        if prediction == instance[len(instance) - 1]:
            num_of_successes = num_of_successes + 1
    accuracy = num_of_successes / dataset.shape[0] * 100
    return accuracy 


# After building the three trees using the training set, you should calculate the accuracy on the test set. For each tree print the training and test accuracy. Select the tree that gave you the best test accuracy. For the rest of the exercise, use that tree (when you asked to build another tree use the same impurity function and same gain_ratio flag). 

# In[43]:


print("tree_gini accuracy on training set is:", calc_accuracy(tree_gini, X_train))
print("tree_gini accuracy on test set is:", calc_accuracy(tree_gini, X_test))
print("tree_entropy accuracy on training set is:", calc_accuracy(tree_entropy, X_train))
print("tree_entropy accuracy on test set is:", calc_accuracy(tree_entropy, X_test))
print("tree_entropy_gain_ratio accuracy on training set is:", calc_accuracy(tree_entropy_gain_ratio, X_train))
print("tree_entropy_gain_ratio accuracy on test set is:", calc_accuracy(tree_entropy_gain_ratio, X_test))


# ## Depth pruning
# 
# (15 points)
# 
# Consider the following max_depth values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]. For each value, construct a tree and prune it according to the max_depth value = don't let the tree to grow beyond this depth. Next, calculate the training and testing accuracy.<br>
# On a single plot, draw the training and testing accuracy as a function of the max_depth. Mark the best result on the graph with red circle.

# In[44]:


# building a tree for each depth , using the accuracy function to mesure and point out the overfitting
depths = [1,2,3,4,5,6,7,8,9,10]
test_set = []
training_set = []
best_accuracy = 0.0
best_depth = 0
for i in depths:
    d_tree = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, max_depth = i)
    test_accuracy = calc_accuracy(d_tree, X_test)
    test_set.append(test_accuracy)
    training_set.append(calc_accuracy(d_tree, X_train))
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_depth = i

plt.scatter(best_depth, best_accuracy, s=100, color="red")
plt.plot(depths, training_set)
plt.plot(depths, test_set)
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Depth')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy as a function of the max_depth')
plt.legend(['Train accuracy', 'Test accuracy']);
plt.show()
    
    


# ## Chi square pre-pruning
# 
# (15 points)
# 
# Consider the following p-value cut-off values: [1 (no pruning), 0.5, 0.25, 0.1, 0.05, 0.0001 (max pruning)]. For each value, construct a tree and prune it according to the cut-off value. Next, calculate the training and testing accuracy. <br>
# On a single plot, draw the training and testing accuracy as a function of the tuple (p-value, tree depth). Mark the best result on the graph with red circle.

# In[46]:


### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning
chi_table = {1: {0.5 : 0.45,
                 0.25 : 1.32,
                 0.1 : 2.71,
                 0.05 : 3.84,
                 0.0001 : 100000},
             2: {0.5 : 1.39,
                 0.25 : 2.77,
                 0.1 : 4.60,
                 0.05 : 5.99,
                 0.0001 : 100000},
             3: {0.5 : 2.37,
                 0.25 : 4.11,
                 0.1 : 6.25,
                 0.05 : 7.82,
                 0.0001 : 100000},
             4: {0.5 : 3.36,
                 0.25 : 5.38,
                 0.1 : 7.78,
                 0.05 : 9.49,
                 0.0001 : 100000},
             5: {0.5 : 4.35,
                 0.25 : 6.63,
                 0.1 : 9.24,
                 0.05 : 11.07,
                 0.0001 : 100000},
             6: {0.5 : 5.35,
                 0.25 : 7.84,
                 0.1 : 10.64,
                 0.05 : 12.59,
                 0.0001 : 100000},
             7: {0.5 : 6.35,
                 0.25 : 9.04,
                 0.1 : 12.01,
                 0.05 : 14.07,
                 0.0001 : 100000},
             8: {0.5 : 7.34,
                 0.25 : 10.22,
                 0.1 : 13.36,
                 0.05 : 15.51,
                 0.0001 : 100000},
             9: {0.5 : 8.34,
                 0.25 : 11.39,
                 0.1 : 14.68,
                 0.05 : 16.92,
                 0.0001 : 100000},
             10: {0.5 : 9.34,
                  0.25 : 12.55,
                  0.1 : 15.99,
                  0.05 : 18.31,
                  0.0001 : 100000},
             11: {0.5 : 10.34,
                  0.25 : 13.7,
                  0.1 : 17.27,
                  0.05 : 19.68,
                  0.0001 : 100000}}


# In[25]:


# building a tree for each p-value, demonstrating the best combniation for max depth and chi statistic value
p_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
training_set = []
test_set = []
tupples = []
best_accuracy = 0.0
ideal_p = -1
for val in p_values:
    decision_tree = build_tree(X_train, calc_entropy, gain_ratio=True, chi = val)
    test_accuracy = calc_accuracy(decision_tree, X_test)
    test_set.append(test_accuracy)
    training_set.append(calc_accuracy(decision_tree, X_train))
    tupples.append((val, decision_tree.tree_depth))
    
    if test_accuracy >= best_accuracy:
        best_accuracy = test_accuracy
        ideal_p = val
        
plt.figure(figsize=(15, 10))
plt.xscale('log')
plt.xticks(p_values,tupples)
plt.plot(p_values, training_set)
plt.plot(p_values, test_set)
plt.scatter(ideal_p, best_accuracy, s=100, color="red")
plt.xlabel('Chi, Tree Depth')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy as a function of the chi value and tree depth')
plt.legend(['Train accuracy', 'Test accuracy']);
plt.show()


# Build the best 2 trees:
# 1. tree_max_depth - the best tree according to max_depth pruning
# 1. tree_chi - the best tree according to chi square pruning

# In[47]:


tree_max_depth = build_tree(X_train, calc_entropy, gain_ratio=True, max_depth = best_depth)
tree_chi = build_tree(X_train, calc_entropy, gain_ratio=True,chi = ideal_p)


# ## Number of Nodes
# 
# (5 points) 
# 
# Of the two trees above we will choose the one with fewer nodes.
# 
# Complete the function counts_nodes and print the number of nodes in each tree

# In[50]:


def count_nodes(node):
    
    num_nodes = 1
    for i in node.children:
        num_nodes += count_nodes(i)
    return num_nodes
    
print("Tree Max Depth has", count_nodes(tree_max_depth), "nodes")
print("Tree Chi has", count_nodes(tree_chi), "nodes")


# ## Print the tree
# 
# (5 points)
# 
# Complete the function `print_tree` and execute it on your chosen tree. Your code should do print:
# ```
# [ROOT, feature=X0],
#   [X0=a, feature=X2]
#     [X2=c, leaf]: [{1.0: 10}]
#     [X2=d, leaf]: [{0.0: 10}]
#   [X0=y, feature=X5], 
#        [X5=a, leaf]: [{1.0: 5}]
#        [X5=s, leaf]: [{0.0: 10}]
#   [X0=e, leaf]: [{0.0: 25, 1.0: 50}]
# ```
# In each brackets:
# * The first argument is the parent feature with the value that led to current node
# * The second argument is the selected feature of the current node
# * If the current node is a leaf, you need to print also the labels and their counts

# In[49]:


# you can change the function signeture
def print_tree(node, labels, depth=0, parent_feature='ROOT'):
    num_of_spaces = depth * "  "
    
    # root
    if node.feature_val == None:
        print("[ROOT, feature={0}],".format(node.feature))
    
    # leaf     
    elif node.is_leaf():     
        count_class_0 = np.count_nonzero(node.data[:, -1] == labels[0])
        count_class_1 = np.count_nonzero(node.data[:, -1] == labels[1])
        class_0 = "{0}: {1}".format(labels[0], count_class_0) if count_class_0 != 0 else ""
        class_1 = "{0}: {1}".format(labels[1], count_class_1) if count_class_1 != 0 else ""
        comma = "" if (count_class_0 == 0 or count_class_1 == 0) else ", "
        
        print(num_of_spaces, "[X{0}={1}, leaf]: [{2}{3}{4}]".format(parent_feature, node.feature_val, class_0, comma, class_1))
    
    # others
    else:
        print(num_of_spaces, "[X{0}={1}, feature=X{2}],".format(parent_feature, node.feature_val, node.feature))
    
    for i in node.children:
        print_tree(i, labels, depth=depth + 1, parent_feature=node.feature)
 
print_tree(tree_chi, ('p', 'e'))


# In[ ]:





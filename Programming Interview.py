#!/usr/bin/env python
# coding: utf-8

# # 1) Generate random graph:
# Write a function to generate a random directed acyclic graph with V number of vertices and E number of edges.

# In[1]:


import random

def generate_dag(V, E, seed=None):
    graph = {i: set() for i in range(V)}
    edges = []
    for i in range(V):
        for j in range(i+1, V):
            edges.append((i,j))
    if seed is not None:
        random.seed(seed)
    random.shuffle(edges)
    for i in range(E):
        u, v = edges[i]
        if u < v and not any(u in graph[p] for p in graph[v]):
            graph[u].add(v)
    return graph


# # 2) Visualize graph:
# Generate one random directed acyclic graph with 20 vertices and 20 edges. Generate a graphical representation of the graph: use circles to represent vertices and arrow to represent edges. Label the circles with the name of the corresponding vertices

# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt

random.seed(666)
graph = generate_dag(20, 20)

G = nx.DiGraph()
for u in graph:
    G.add_node(u)
for u in graph:
    for v in graph[u]:
        G.add_edge(u, v)

pos = nx.spring_layout(G)  # Change layout to spring_layout
node_size = 750  # Reduce node size
node_color = 'green'  # Change node color
edge_color = 'black'  # Change edge color

plt.figure(figsize=(15,15))  # Adjust figure size
nx.draw_networkx(G, pos, with_labels=True, arrows=True, node_size=node_size, node_color=node_color, edge_color=edge_color)
plt.title('Directed Acyclic Graph')
plt.axis('off')  # Turn off axis
plt.show()


# # 3) Enumerate paths:
# Write a function to enumerate all directed paths between two variables Vi and Vj in a directed acyclic graph G. (e.g. in the graph from 2), all directed paths between a and d are: a->b->d and a->c->d). Demonstrate how your function works using the graph you generated from step 2).

# In[ ]:


def enumerate_directed_paths(G, u, v):
    if u == v:
        return [[u]]
    paths = []
    for neighbor in G[u]:
        for path in enumerate_directed_paths(G, neighbor, v):
            paths.append([u] + path)
    return paths


# In[ ]:


enumerate_directed_paths(G,4,17)


# In[ ]:


for i in range(20):
    for j in range(20):
        paths = enumerate_directed_paths(G, i, j)
        print(f"All paths from {i} to {j}: {paths}")


# # 4) Generate data according to the structure of the graph:
# Generate data based on the graph you created in 2) in the following fashion: Any variable without any parent are random variable following Gaussian distribution with mean 0 standard deviation of 1 ; any variable with parents are the sum of their parents plus a Gaussian noise term with mean 0 and standard deviation of 1. Generate 1000 observations.

# In[ ]:


import numpy as np

np.random.seed(666)
# Simulate data for each vertices
n = 1000
data = np.zeros((n, len(graph)))

for i in range(len(graph)):
    parents = list(graph[i])
    if len(parents) == 0:
        # If the vertex has no parents, draw from standard normal
        data[:, i] = np.random.normal(size=n)
    else:
        # If the vertex has parents, sum their values plus noise
        parent_values = np.zeros((n, len(parents)))
        for j, parent in enumerate(parents):
            parent_values[:, j] = data[:, parent]
        noise = np.random.normal(size=n)
        data[:, i] = np.sum(parent_values, axis=1) + noise
        
import pandas as pd
col=['V0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19']
data=pd.DataFrame(data,columns=col)
data


# # 5) Predictive modeling:
# Create predictive model using the data you generated in step 4) with the following specifications: <br>
# Target variable for prediction: randomly pick a variable that have more than one neighbor as the target of prediction. <br>
# Features: all other variables that are not the target are potential predictors/features. <br>
# Feature selection and regression algorithm: use your favorite methods for feature selection and regression. You can use a pre-existing implementation. <br>
# Validation: use 800 samples for training and 200 sample for validation. <br>
# Performance metric: use your favorite metric(s) <br>
# Report the following: (1) the performance metric (2) what features are selected by the feature selector, and (3) the graph distances of the selected features to the target.

# In[ ]:


graph


# In[ ]:


# Let's choose vertices 17 as the target variable, since it's connected to many vertex before

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = data.drop('V17', axis=1)  # Features
y = data['V17']  # Target variable

# Split the data into train and validation sets with 80/20 ratio
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=666)

lasso = Lasso(alpha=0.05) # Default regularized parameter alpha=1
lasso.fit(X_train, y_train)

# Get the predicted values on the validation set
y_pred = lasso.predict(X_val)

# Calculate MSE on the validation set
mse = mean_squared_error(y_val, y_pred)
print('Mean Squared Error (MSE):', mse)

# Display selected features and coefficients
selected_features = X.columns[lasso.coef_ != 0]
coefficients = lasso.coef_[lasso.coef_ != 0]
intercept = lasso.intercept_

# Create a DataFrame to display the selected features, coefficients, and intercept
coef_df = pd.DataFrame({'Feature': ['Intercept'] + selected_features.tolist(), 
                        'Coefficient': [intercept] + coefficients.tolist()})
print(coef_df)


# In[ ]:


enumerate_directed_paths(G,11,17)
# The graph distance is 1


# # 6) Data generation Graph and Predictive Performance:
# Generate 100 random directed acyclic graph with 20 vertices and 20 edges. Go through step 4)
# and 5) for each random graph. Plot the histogram for the performance metric(s) you obtain for
# the 100 random graphs. What factors may contribute to the variability in the performance
# metric? Demonstrate with the data/results you have.

# In[ ]:


# Generate 100 datasets from DAG
n_datasets = 100
V = 20
E = 20
mse_list = []

for i in range(n_datasets):
    # Generate DAG
    graph = generate_dag(V, E, seed=i)
    
    # Simulate data for each vertex
    n = 1000
    data = np.zeros((n, len(graph)))
    for i in range(len(graph)):
        parents = list(graph[i])
        if len(parents) == 0:
            data[:, i] = np.random.normal(size=n)
        else:
            parent_values = np.zeros((n, len(parents)))
            for j, parent in enumerate(parents):
                parent_values[:, j] = data[:, parent]
            noise = np.random.normal(size=n)
            data[:, i] = np.sum(parent_values, axis=1) + noise
    
    # Convert data to DataFrame
    col = ['V{}'.format(i) for i in range(V)]
    data = pd.DataFrame(data, columns=col)
    
    # Choose vertex 17 as the target variable
    X = data.drop('V17', axis=1)
    y = data['V17']
    
    # Split the data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=666)
    
    # Perform Lasso regression
    lasso = Lasso(alpha=0.05)
    lasso.fit(X_train, y_train)
    
    # Get the predicted values on the validation set
    y_pred = lasso.predict(X_val)
    
    # Calculate MSE on the validation set
    mse = mean_squared_error(y_val, y_pred)
    mse_list.append(mse)

# Plot histogram of MSEs
plt.hist(mse_list, bins=20)
plt.xlabel('Mean Squared Error (MSE)')
plt.ylabel('Frequency')
plt.title('Performance of Lasso Regression on 100 Datasets')
plt.show()


# # Data Visualization

# In[ ]:


import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.figure(figsize = (20, 15))
plotnumber = 1

for col in data.columns:
    if plotnumber <= 30:
        ax = plt.subplot(5, 5, plotnumber)
        sns.distplot(data[col],rug=False,hist=False)
        plt.xlabel(col, fontsize = 15)
    
    plotnumber += 1
plt.tight_layout()
plt.show()

# no scaling required as data seems to follow normal distribution, I think the imperfect bell shape is due to correlation as some vertices are connected


# In[ ]:


plt.figure(figsize = (18, 12))

corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))

sns.heatmap(data = corr, mask = mask, annot = True, fmt = '.2g', linewidth = 1)
plt.show()

# Low correlation between predictors is good but low correlation between predictors and any target variable is bad for linear model
# There's slightly more correlation between 2 vertices when they are connected than when they are disconnected


# # Testing some models

# In[ ]:


# Let's choose vertices 0 as the target variable, which is isolated

X = data.drop('V0', axis=1)  # Features
y = data['V0']  # Target variable

# Split the data into train and validation sets with 80/20 ratio
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=666)

lasso = Lasso(alpha=0.05) # Default regularized parameter alpha=1
lasso.fit(X_train, y_train)

# Get the predicted values on the validation set
y_pred = lasso.predict(X_val)

# Calculate MSE on the validation set
mse = mean_squared_error(y_val, y_pred)
print('Mean Squared Error (MSE):', mse)

# Display selected features and coefficients
selected_features = X.columns[lasso.coef_ != 0]
coefficients = lasso.coef_[lasso.coef_ != 0]
intercept = lasso.intercept_

# Create a DataFrame to display the selected features, coefficients, and intercept
coef_df = pd.DataFrame({'Feature': ['Intercept'] + selected_features.tolist(), 
                        'Coefficient': [intercept] + coefficients.tolist()})
print(coef_df)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()
knn.fit(X_train, y_train) # Default 5 nearest neighbourhood

# Get the predicted values on the validation set
y_pred = knn.predict(X_val)

# Calculate MSE on the validation set
mse = mean_squared_error(y_val, y_pred)
print('Mean Squared Error (MSE):', mse)

# Non linear model does not perform better than linear model


# # Code Testing

# In[ ]:


from sklearn.datasets import load_boston
boston_dataset = load_boston()
data = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
print(data.columns)
# Load your data into a pandas DataFrame
# Assuming you have a DataFrame called 'data' with your features and target variable
X = data.drop('LSTAT', axis=1)  # Features
y = data['LSTAT']  # Target variable

# Split the data into train and validation sets with 80/20 ratio
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=666)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

# Fit Lasso regression model
lasso = Lasso(alpha=0.05)
lasso.fit(X_train, y_train)

# Get the predicted values on the validation set
y_pred = lasso.predict(X_val)

# Calculate MSE on the validation set
mse = mean_squared_error(y_val, y_pred)
print('Mean Squared Error (MSE):', mse)

# Display selected features and coefficients
selected_features = X.columns[lasso.coef_ != 0]
coefficients = lasso.coef_[lasso.coef_ != 0]
intercept = lasso.intercept_

# Create a DataFrame to display the selected features, coefficients, and intercept
coef_df = pd.DataFrame({'Feature': ['Intercept'] + selected_features.tolist(), 
                        'Coefficient': [intercept] + coefficients.tolist()})
print(coef_df)

# Test data show code and result are valid


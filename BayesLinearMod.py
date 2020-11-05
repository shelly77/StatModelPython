#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:47:05 2020

@author: yihanc
"""


# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np
np.random.seed(42)

#sklearn for data set split
from sklearn.model_selection import train_test_split
 
#matplotlib for plotting figures
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['figure.figsize'] = (9, 9)

import seaborn as sns


# PyMC3 for Bayesian Inference
import pymc3 as pm

"""
load data
the data can be download from https://archive.ics.uci.edu/ml/datasets/student+performance#
make sure that .csv file is kept in your working directory
"""
#%%
df = pd.read_csv('student-mat.csv', sep = ";")

df = df[~df['G3'].isin([0, 1])]

df = df.rename(columns={'G3': 'Grade'})

df = df.drop(columns=['G1', 'G2'])

df.head()
#%%

"""
check corerlation between all the continuous variables
check numerical correaltion between all the continuous variables and "Grade"
"""
#%%
df.corr()

# Correlations of continuous variables
df.corr()['Grade'].sort_values()
#%%

"""
check corerlation between all the continuous variables
check numerical correaltion between all the continuous variables and "Grade"
"""
#%%
# Select only categorical variables
category_df = df.select_dtypes('object')
# One hot encode the variables
dummy_df = pd.get_dummies(category_df)
# Put the grade back in the dataframe
dummy_df['Grade'] = df['Grade']
dummy_df.head()
# Correlations in one-hot encoded dataframe
dummy_df.corr()['Grade'].sort_values()
#%%

"""
select from all the variable four variables which have highest correlation
with "Grade" and split the data into traning set and test set
"""

#%%
# grade and returns training and testing datasets
def format_data(df):
    # Targets are final grade of student
    labels = df['Grade']
    
    # Drop the school from features
    df = df.drop(columns=['school'])
    
    # One-Hot Encoding of Categorical Variables
    df = pd.get_dummies(df)
    
    # Find correlations with the Grade
    allcor = df.corr().abs()['Grade'].sort_values(ascending=False)
    
    # Maintain the top 6 most correlation features with Grade
    most_correlated = allcor[:6]
    
    df = df.loc[:, most_correlated.index]
    
    df = df.drop(columns = 'schoolsup_no')
    
    # Split into training/testing sets with 25% split
    X_train, X_test, y_train, y_test = train_test_split(df, labels, 
                                                        test_size = 0.25,
                                                        random_state=42)
    #X_train and X_test contains all the variables including the response variable  
    #y_train and y_test only contains the response variable
    return X_train, X_test, y_train, y_test

#%%

"""
Create linear regression model in PyMC3
and sample posterior estimates
"""
#%%

# run customized function format_data first
format_data(df)

# Formula for Bayesian Linear Regression (follows R formula syntax
formula = 'Grade ~ ' + ' + '.join(['%s' % variable for variable in X_train.columns[1:]])
formula

with pm.Model() as normal_model:
    
    # The prior for the model parameters will be a normal distribution
    family = pm.glm.families.Normal()
    
    # Creating the model requires a formula and data (and optionally a family)
    pm.GLM.from_formula(formula, data = X_train, family = family)
    
    # Perform Markov Chain Monte Carlo sampling
    normal_trace = pm.sample(draws=2000, chains = 2, tune = 500,cores=1)

#%%


"""
Some useful functions for the posterior inspection
"""
#%%
pm.summary(normal_trace)
plot_trace(normal_trace)
pm.forestplot(normal_trace)
pm.plot_posterior(normal_trace, figsize = (14, 14), text_size=20)

#%%

"""
make predictions with test data set
"""
#%%
#Make a new prediction from the test set and compare to actual value
def test_model(trace, test_observation):
    
    # Print out the test observation data
    print('Test Observation:')
    print(test_observation)
    var_dict = {}
    for variable in trace.varnames:
        var_dict[variable] = trace[variable]

    # Results into a dataframe
    var_weights = pd.DataFrame(var_dict)
    
    # Standard deviation of the likelihood
    sd_value = var_weights['sd'].mean()

    # Actual Value
    actual = test_observation['Grade']
    
  
    # Add in intercept term
    test_observation['Intercept'] = 1
    test_observation = test_observation.drop('Grade')
    
    # Align weights and test observation
    var_weights = var_weights[test_observation.index]

    # Means for all the weights
    var_means = var_weights.mean(axis=0)

    # Location of mean for observation
    mean_loc = np.dot(var_means, test_observation)
    
    # Estimates of grade
    estimates = np.random.normal(loc = mean_loc, scale = sd_value,
                                 size = 1000)

    # Plot all the estimates
    sns.distplot(estimates, hist = True, kde = True, bins = 19,
                 hist_kws = {'edgecolor': 'k', 'color': 'darkblue'},
                kde_kws = {'linewidth' : 4},
                label = 'Estimated Dist.')
    # Plot the actual grade
    plt.vlines(x = actual, ymin = 0, ymax = 5, 
               linestyles = '--', colors = 'red',
               label = 'True Grade',
              linewidth = 2.5)
    
    # Plot the mean estimate
    plt.vlines(x = mean_loc, ymin = 0, ymax = 5, 
               linestyles = '-', colors = 'orange',
               label = 'Mean Estimate',
              linewidth = 2.5)
    
    plt.legend(loc = 1)
    plt.title('Density Plot for Test Observation');
    plt.xlabel('Grade'); plt.ylabel('Density');
    
    # Prediction information
    print('True Grade = %d' % actual)
    print('Average Estimate = %0.4f' % mean_loc)
    print('5%% Estimate = %0.4f    95%% Estimate = %0.4f' % (np.percentile(estimates, 5),
                                       np.percentile(estimates, 95)))








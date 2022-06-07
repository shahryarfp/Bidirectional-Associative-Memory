#!/usr/bin/env python
# coding: utf-8

# # Q4

# In[1]:


import numpy as np
import copy
import random


# In[2]:


# Data

name_1 = np.array([1,0,0,0,0,1,1,  1,1,0,1,1,0,0,  1,1,0,1,0,0,1,  1,1,0,1,1,1,0,  1,1,1,0,1,0,0,                   1,1,0,1,1,1,1,  1,1,0,1,1,1,0])
name_2 = np.array([1,0,0,1,0,0,0,  1,1,0,1,0,0,1,  1,1,0,1,1,0,0,  1,1,0,1,1,0,0,  1,1,0,0,0,0,1,                   1,1,1,0,0,1,0,  1,1,1,1,0,0,1])
name_3 = np.array([1,0,0,1,0,1,1,  1,1,0,0,1,0,1,  1,1,0,1,1,1,0,  1,1,1,0,0,1,1,  1,1,1,0,1,0,0,                   1,1,0,0,0,0,1,  1,1,1,0,0,1,0])

feature_1 = np.array([1,0,1,0,0,0,0, 1,1,1,0,0,1,0, 1,1,0,0,1,0,1, 1,1,1,0,0,1,1, 1,1,0,1,0,0,1,                      1,1,0,0,1,0,0, 1,1,0,0,1,0,1, 1,1,0,1,1,1,0, 1,1,1,0,1,0,0])
feature_2 = np.array([1,0,0,0,1,1,0, 1,1,0,1,0,0,1, 1,1,1,0,0,1,0, 1,1,1,0,0,1,1, 1,1,1,0,1,0,0,                      1,0,0,1,1,0,0, 1,1,0,0,0,0,1, 1,1,0,0,1,0,0, 1,1,1,1,0,0,1])
feature_3 = np.array([1,0,0,0,1,1,1, 1,1,0,0,1,0,1, 1,1,0,1,1,1,0, 1,1,1,0,1,0,0, 1,1,0,1,1,0,0,                      1,1,0,0,1,0,1, 1,1,0,1,1,0,1, 1,1,0,0,0,0,1, 1,1,0,1,1,1,0])

my_dictionary = {
    'Clinton':name_1,
    'Hillary':name_2,
    'Kenstar':name_3,
    
    'President': feature_1,
    'FirstLady': feature_2,
    'Gentleman': feature_3
}


# In[3]:


#converting datas to bipolar

def convert_to_bipolar(vector_):
    for i in range(len(vector_)):
        if vector_[i] == 0:
            vector_[i] = -1

convert_to_bipolar(name_1)
convert_to_bipolar(name_2)
convert_to_bipolar(name_3)
convert_to_bipolar(feature_1)
convert_to_bipolar(feature_2)
convert_to_bipolar(feature_3)


# # Part 1
# ### Creating Weight Matrix

# In[4]:


weight_matrix = np.zeros((len(name_1),len(feature_1)))
def update_weight(name_, feature_):
    for i in range(len(name_)):
        for j in range(len(feature_)):
            weight_matrix[i][j] += name_[i]*feature_[j]

update_weight(name_1, feature_1)
update_weight(name_2, feature_2)
update_weight(name_3, feature_3)
print('Weight Matrix Shape:', weight_matrix.shape)
weight_matrix


# # Part 2

# In[5]:


def my_sign(vector_):
    for i in range(len(vector_)):
        if vector_[i] >=0:
            vector_[i] = 1
        else :
            vector_[i] = -1
    return vector_

def is_same(v1, v2):
    if len(v1) != len(v2):
        return False
    for i in range(len(v1)):
        if v1[i] != v2[i]:
            return False
    return True


# In[6]:


predicted_name_1 = my_sign(np.matmul(name_1, weight_matrix))
if is_same(predicted_name_1, feature_1):
    print('name_1 predicted correctly')
predicted_name_2 = my_sign(np.matmul(name_2, weight_matrix))
if is_same(predicted_name_2, feature_2):
    print('name_2 predicted correctly')
predicted_name_3 = my_sign(np.matmul(name_3, weight_matrix))
if is_same(predicted_name_3, feature_3):
    print('name_3 predicted correctly')

predicted_feature_1 = my_sign(np.matmul(feature_1, weight_matrix.T))
if is_same(predicted_feature_1, name_1):
    print('feature_1 predicted correctly')
predicted_feature_2 = my_sign(np.matmul(feature_2, weight_matrix.T))
if is_same(predicted_feature_2, name_2):
    print('feature_2 predicted correctly')
predicted_feature_3 = my_sign(np.matmul(feature_3, weight_matrix.T))
if is_same(predicted_feature_3, name_3):
    print('feature_3 predicted correctly')


# # Part 3
# ### Adding Noise

# In[7]:


def add_noise(vector__, noise):
    vector_ = copy.deepcopy(vector__)
    for i in range(len(vector_)):
        rand = random.random()
        if rand < noise:
            vector_[i] *= -1
    return vector_

def activation_func(y_in_, y_in_prev):
    y_in = copy.deepcopy(y_in_)
    for i in range(len(y_in)):
        if y_in[i] > 0:
            y_in[i] = 1
        elif y_in[i] == 0:
            y_in[i] = y_in_prev[i]
        elif y_in[i] < 0:
            y_in[i] = -1
    return y_in

def calculate_accuracy(predicted, real):
    count = 0
    for i in range(len(real)):
        if predicted[i] == real[i]:
            count += 1
    return count/len(real)


# ### Bidirectional Associative Memory

# In[8]:


def BAM(inputs, weight_matrix, outputs, noise, results_size):
    results = np.zeros(results_size)
    for i in range(len(inputs)):
        for j in range(100):
            n10_name_ = add_noise(inputs[i], noise)

            #step 2a
            x = n10_name_
            x_prev = copy.deepcopy(x)
            #step 2b
            y_prev = np.zeros(len(outputs[0]))

            #step 3
            while True:
                #step 4
                y_in = np.matmul(x, weight_matrix)
                y = activation_func(y_in, y_prev)

                #step 5
                x_in = np.matmul(y, weight_matrix.T)
                x = activation_func(x_in, x_prev)
                y_prev = copy.deepcopy(y)

                #step 6
                predicted_ans = activation_func(np.matmul(y, weight_matrix.T), x_prev)
                if (x - x_prev).any() == 0:
                    results[i] += calculate_accuracy(predicted_ans, names[i])
                    break

                x_prev = copy.deepcopy(x)
                
    return results


# ### 10% Noise

# In[9]:


names = [name_1, name_2, name_3]
features = [feature_1, feature_2, feature_3]
noise = 0.1

print('Noise:', noise*100,'%')
results = BAM(names, weight_matrix, features, noise, 3)
print('Name 1 Accuracy:', results[0])
print('Name 2 Accuracy:', results[1])
print('Name 3 Accuracy:', results[2])

results = BAM(features, weight_matrix.T, names, noise, 3)
print('Feature 1 Accuracy:', results[0])
print('Feature 2 Accuracy:', results[1])
print('Feature 3 Accuracy:', results[2])


# ### 20% Noise

# In[10]:


noise = 0.2

print('Noise:', noise*100,'%')
results = BAM(names, weight_matrix, features, noise, 3)
print('Name 1 Accuracy:', results[0])
print('Name 2 Accuracy:', results[1])
print('Name 3 Accuracy:', results[2])

results = BAM(features, weight_matrix.T, names, noise, 3)
print('Feature 1 Accuracy:', results[0])
print('Feature 2 Accuracy:', results[1])
print('Feature 3 Accuracy:', results[2])


# # Part 4
# ### Adding another Data and doing all the previous steps again

# In[11]:


#Lewisky
name_4 = np.array([1,0,0,1,1,0,0, 1,1,0,0,1,0,1, 1,1,1,0,1,1,1, 1,1,0,1,0,0,1,                   1,1,1,0,0,1,1, 1,1,0,1,0,1,1, 1,1,1,1,0,0,1])
#SweetGirl
feature_4 = np.array([1,0,1,0,0,1,1, 1,1,1,0,1,1,1, 1,1,0,0,1,0,1, 1,1,0,0,1,0,1, 1,1,1,0,1,0,0,                      1,0,0,0,1,1,1, 1,1,0,1,0,0,1, 1,1,1,0,0,1,0, 1,1,0,1,1,0,0])


# In[12]:


convert_to_bipolar(name_4)
convert_to_bipolar(feature_4)


# In[13]:


weight_matrix = np.zeros((len(name_1),len(feature_1)))
def update_weight(name_, feature_):
    for i in range(len(name_)):
        for j in range(len(feature_)):
            weight_matrix[i][j] += name_[i]*feature_[j]

update_weight(name_1, feature_1)
update_weight(name_2, feature_2)
update_weight(name_3, feature_3)
update_weight(name_4, feature_4)


# In[14]:


predicted_name_1 = my_sign(np.matmul(name_1, weight_matrix))
if is_same(predicted_name_1, feature_1):
    print('name_1 predicted correctly')
predicted_name_2 = my_sign(np.matmul(name_2, weight_matrix))
if is_same(predicted_name_2, feature_2):
    print('name_2 predicted correctly')
predicted_name_3 = my_sign(np.matmul(name_3, weight_matrix))
if is_same(predicted_name_3, feature_3):
    print('name_3 predicted correctly')
predicted_name_4 = my_sign(np.matmul(name_4, weight_matrix))
if is_same(predicted_name_4, feature_4):
    print('name_4 predicted correctly')

predicted_feature_1 = my_sign(np.matmul(feature_1, weight_matrix.T))
if is_same(predicted_feature_1, name_1):
    print('feature_1 predicted correctly')
predicted_feature_2 = my_sign(np.matmul(feature_2, weight_matrix.T))
if is_same(predicted_feature_2, name_2):
    print('feature_2 predicted correctly')
predicted_feature_3 = my_sign(np.matmul(feature_3, weight_matrix.T))
if is_same(predicted_feature_3, name_3):
    print('feature_3 predicted correctly')
predicted_feature_4 = my_sign(np.matmul(feature_4, weight_matrix.T))
if is_same(predicted_feature_4, name_4):
    print('feature_4 predicted correctly')


# In[15]:


#10 % noise
names = [name_1, name_2, name_3, name_4]
features = [feature_1, feature_2, feature_3, feature_4]
noise = 0.1

print('Noise:', noise*100,'%')
results = BAM(names, weight_matrix, features, noise, 4)
print('Name 1 Accuracy:', results[0])
print('Name 2 Accuracy:', results[1])
print('Name 3 Accuracy:', results[2])
print('Name 3 Accuracy:', results[3])

results = BAM(features, weight_matrix.T, names, noise, 4)
print('Feature 1 Accuracy:', results[0])
print('Feature 2 Accuracy:', results[1])
print('Feature 3 Accuracy:', results[2])
print('Feature 3 Accuracy:', results[3])


# In[16]:


#20 % noise
noise = 0.2

print('Noise:', noise*100,'%')
results = BAM(names, weight_matrix, features, noise, 4)
print('Name 1 Accuracy:', results[0])
print('Name 2 Accuracy:', results[1])
print('Name 3 Accuracy:', results[2])
print('Name 3 Accuracy:', results[3])

results = BAM(features, weight_matrix.T, names, noise, 4)
print('Feature 1 Accuracy:', results[0])
print('Feature 2 Accuracy:', results[1])
print('Feature 3 Accuracy:', results[2])
print('Feature 3 Accuracy:', results[3])


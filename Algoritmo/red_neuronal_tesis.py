#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:24:48 2019

@author: aldo_mellado
"""
# =============================================================================
# Importing the libraries
# =============================================================================
import os
import numpy as np
import pandas as pd
from fixrows import fixrows
from merge_csv import fusionar_csv
from create_cords import createcords

# =============================================================================
# Importing the dataset
# =============================================================================
   
#Run this code, only if the file with all the dataframes were deleted

dir_pot = os.getcwd() + '/Potencias/'
 
df1 = fixrows(dir_pot+'Potencia_r1')
num_row = np.shape(df1)[0]
coords  = ['(1,0)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df1 = coords.join(df1, how='left')
df1.to_csv(dir_pot + 'Potencia_R1.csv')

df2 = fixrows(dir_pot+'Potencia_r2')
num_row = np.shape(df2)[0]
coords  = ['(0,0)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df2 = coords.join(df2, how='left')
df2.to_csv(dir_pot + 'Potencia_R2.csv')

df3 = fixrows(dir_pot+'Potencia_r3')
num_row = np.shape(df3)[0]
coords  = ['(0,1)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df3 = coords.join(df3, how='left')
df3.to_csv(dir_pot + 'Potencia_R3.csv')

df4 = fixrows(dir_pot+'Potencia_r4')
num_row = np.shape(df4)[0]
coords  = ['(1,1)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df4 = coords.join(df4, how='left')
df4.to_csv(dir_pot + 'Potencia_R4.csv')
 
 #Ingresar a carpeta potencias
fusionar_csv('Potencia_R1','Potencia_R2','Potencia_R3','Potencia_R4')

df0 = fixrows(dir_pot+'/potencias_fusionado').iloc[:,1:]
# =============================================================================

#df0 = pd.read_csv(dir_pot+'potencias_fusionado.csv').iloc[:,1:]

X = df0.iloc[:,1:].values #variables Dependientes (Potencias)
y = df0.iloc[:,0].values #values Independientes (Posición)

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)

#returns a list with the positions (0,0),(0,1),(1,0),(1,1)
#list(encoder.classes_)
#encoder.transform(['(0,0)','(0,0)','(1,0)','(1,1)']) 
#list(encoder.inverse_transform([2, 2, 1]))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.3, random_state = 0)

# =============================================================================
# Feature Scaling (Standarization)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Feature Scaling (Normalization)
from sklearn import preprocessing
# Normalize total_bedrooms column
X_trainn = preprocessing.normalize(X_train)
X_testn = preprocessing.normalize(X_test)
# =============================================================================

# Part 2 - Now let's make the ANN!
# Importing the Keras libraries and packages

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19)) # input dim is the number of nodes in the input layer, or independant variables 
classifier.add(Dropout(rate = 0.1))

classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# Adding the first hidden layer
classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_trainn, y_train, batch_size = 32, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_testn)
y_pred = (y_pred > 0.5)

#1,0
new_prediction = classifier.predict(np.array([[-67,-7   3,-79,-72,-79,-76,-79,-73,-78,-52,-65,-73,-72,-72,-65,-72,-73]]))

#0,0
new_prediction = classifier.predict_classes(np.array([[-47,-67,-59,-63,-65,-75,-71,-83,-67,-75,-79,-57,-81,-77,-75,-75,-73]]),2)
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19))

    classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.1))
    
    classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.1))
    
    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.1))
    
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.1))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 32, epochs = 300)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

new_prediction = classifier.predict(np.array([[-67.0, -69, -73, -75, -81, -61, -71, -65, -75, -85, -73, -63, -53,
       -71, -71, -81, -60]]))


# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
#1
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19))
    classifier.add(Dropout(p = 0.1))
#2
    classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
#3
    classifier.add(Dense(units = 25, kernel_initializer = 'uniform', activation = 'relu'))
#    classifier.add(Dropout(p = 0.1))
#4
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
#5
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))    
#6
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [32, 64],
              'epochs': [100, 300],
#              'optimizer': ['adam', 'adamax','rmsprop']}
                'optimizer': ['adam', 'adamax']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_trainn, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
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
 
df1 = fixrows(dir_pot+'Potencia__r1').iloc[:,3:]
num_row = np.shape(df1)[0]
coords  = [[(1,0)] for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df1 = coords.join(df1, how='left')
df1.to_csv(dir_pot + 'Potencia_R1.csv')

df2 = fixrows(dir_pot+'Potencia__r2').iloc[:,3:]
num_row = np.shape(df2)[0]
coords  = [[(0,0)] for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df2 = coords.join(df2, how='left')
df2.to_csv(dir_pot + 'Potencia_R2.csv')

df3 = fixrows(dir_pot+'Potencia__r3').iloc[:,3:]
num_row = np.shape(df3)[0]
coords  = [[(0,1)] for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df3 = coords.join(df3, how='left')
df3.to_csv(dir_pot + 'Potencia_R3.csv')

df4 = fixrows(dir_pot+'Potencia__r4').iloc[:,3:]
num_row = np.shape(df4)[0]
coords  = [[(1,1)] for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df4 = coords.join(df4, how='left')
df4.to_csv(dir_pot + 'Potencia_R4.csv')
 
 #Ingresar a carpeta potencias
fusionar_csv('Potencia_R1','Potencia_R2','Potencia_R3','Potencia_R4')

df0 = fixrows(dir_pot+'potencias_fusionado').iloc[:,1:]
    # =============================================================================

df0 = pd.read_csv(dir_pot+'potencias_fusionado.csv')

X = df0.iloc[:,1:].values #variables Dependientes (Potencias)
y = df0.iloc[:,0].values #values Independientes (Posición)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
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

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))
# input dim is the number of nodes in the input layer, or independant variables 
classifier.add(Dropout(rate = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 40, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# Adding the third hidden layer
classifier.add(Dense(units = 25, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# Adding the third hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

new_prediction = classifier.predict(np.array([[-67, -69, -73, -75, -81, -61, -71, -65, -75, -85, -73, -63, -53,
       -71, -71, -81, -77]]))
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
    classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 25, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
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
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
#1
    classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 17))
    classifier.add(Dropout(p = 0.1))
#2
    classifier.add(Dense(units = 40, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
#3
    classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
#4
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
#5
    classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 50],
              'epochs': [100, 300],
#              'optimizer': ['adam', 'adamax','rmsprop']}
                'optimizer': ['adam', 'adamax']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
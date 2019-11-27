#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:14:06 2019

@author: aldo_mellado
"""

# =============================================================================
# Importing the libraries
# =============================================================================
import os
import numpy as np
import pandas as pd
# =============================================================================
# from fixrows import fixrows
# from merge_csv import fusionar_csv
# from create_cords import createcords
# =============================================================================

# =============================================================================
# Importing the dataset
# =============================================================================
   
#Run this code, only if the file with all the dataframes were deleted

# =============================================================================
# dir_pot = os.getcwd() + '/Potencias/'
#  
# df1 = fixrows(dir_pot+'Potencia_r1')
# num_row = np.shape(df1)[0]
# coords  = ['(1,0)' for j in range(num_row)]
# coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
# df1 = coords.join(df1, how='left')
# df1.to_csv(dir_pot + 'Potencia_R1.csv')
# 
# df2 = fixrows(dir_pot+'Potencia_r2')
# num_row = np.shape(df2)[0]
# coords  = ['(0,0)' for j in range(num_row)]
# coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
# df2 = coords.join(df2, how='left')
# df2.to_csv(dir_pot + 'Potencia_R2.csv')
# 
# df3 = fixrows(dir_pot+'Potencia_r3')
# num_row = np.shape(df3)[0]
# coords  = ['(0,1)' for j in range(num_row)]
# coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
# df3 = coords.join(df3, how='left')
# df3.to_csv(dir_pot + 'Potencia_R3.csv')
# 
# df4 = fixrows(dir_pot+'Potencia_r4')
# num_row = np.shape(df4)[0]
# coords  = ['(1,1)' for j in range(num_row)]
# coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
# df4 = coords.join(df4, how='left')
# df4.to_csv(dir_pot + 'Potencia_R4.csv')
#  
#  #Ingresar a carpeta potencias
# fusionar_csv('Potencia_R1','Potencia_R2','Potencia_R3','Potencia_R4')
# 
# df0 = fixrows(dir_pot+'/potencias_fusionado').iloc[:,1:]
# =============================================================================

df0 = pd.read_csv('potencias_fusionado_corregido.csv').iloc[:,1:]

X = df0.iloc[:,1:].values #variables Dependientes (Potencias)
y = df0.iloc[:,0].values #values Independientes (Posición)

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)
y_encoded = np_utils.to_categorical(y_encoded)


#returns a list with the positions (0,0),(0,1),(1,0),(1,1)
#list(encoder.classes_)
#encoder.transform(['(0,0)','(0,0)','(1,0)','(1,1)']) 
#list(encoder.inverse_transform([2, 2, 1]))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.2, random_state = 0)

# =============================================================================
# Feature Scaling (Standarization)
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainn = sc.fit_transform(X_train)
X_testn = sc.transform(X_test)
# =============================================================================


# Feature Scaling (Normalization)
#from sklearn import preprocessing
#X_trainn = preprocessing.normalize(X_train)
#X_testn = preprocessing.normalize(X_test)
# =============================================================================

import keras
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
    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
#3
#    classifier.add(Dense(units = 25, kernel_initializer = 'uniform', activation = 'relu'))
#    classifier.add(Dropout(p = 0.1))
#4
    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = 0.1))
#5
#    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))    
#    classifier.add(Dropout(p = 0.1))
#6
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [32, 64],
              'epochs': [100, 150],
#              'optimizer': ['adam', 'adamax','rmsprop']}
                'optimizer': ['adam', 'adamax']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_search = grid_search.fit(X_trainn, y_train)
best_parameters = grid_search.best_params_


print(f"best_parameters = {best_parameters}")
best_accuracy = grid_search.best_score_
print(f"best_accuracy = {best_accuracy}")


from matplotlib import pyplot as plt

plt.subplot(121)
plt.plot(grid_search.grid_search['acc'])
plt.plot(grid_search.grid_search['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='center right')

plt.subplot(122)
plt.plot(grid_search.grid_search['loss'])
plt.plot(grid_search.grid_search['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='center right')
plt.show()


#from ann_visualizer.visualize import ann_viz
#ann_viz(classifier, view=True, filename='network.gv', title='MyNeural Network_variations')

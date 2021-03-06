#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:14:06 2019

@author: aldo_mellado
"""
# ============================================================================
# Importing the libraries
# =============================================================================
import numpy as np
import pandas as pd
from fixrows import fixrows
from pytictoc import TicToc
from merge_csv import fusionar_csv


# =============================================================================
# Importing the dataset
# =============================================================================

t = TicToc()
df1 = fixrows('Potencia_r1').iloc[:25000,:]
num_row = np.shape(df1)[0]
coords  = ['(1,0)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df1 = coords.join(df1, how='left')
df1.to_csv('Potencia_R1.csv')
 
df2 = fixrows('Potencia_r2').iloc[:25000,:]
num_row = np.shape(df2)[0]
coords  = ['(0,0)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df2 = coords.join(df2, how='left')
df2.to_csv('Potencia_R2.csv')
 
df3 = fixrows('Potencia_r3').iloc[:25000,:]
num_row = np.shape(df3)[0]
coords  = ['(0,1)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df3 = coords.join(df3, how='left')
df3.to_csv('Potencia_R3.csv')

df4 = fixrows('Potencia_r4').iloc[:25000,:]
num_row = np.shape(df4)[0]
coords  = ['(1,1)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df4 = coords.join(df4, how='left')
df4.to_csv('Potencia_R4.csv')
  
# Fusionar archivos corregidos para obtener el archivo de potencias final
fusionar_csv('Potencia_R1','Potencia_R2','Potencia_R3','Potencia_R4')
 
df0 = fixrows('potencias_fusionado').iloc[:,1:]

df0 = pd.read_csv('potencias_fusionado_corregido.csv').iloc[3:,1:]

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
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# =============================================================================


# Feature Scaling (Normalization)
#from sklearn import preprocessing
#X_trainn = preprocessing.normalize(X_train)
#X_testn = preprocessing.normalize(X_test)
# =============================================================================

# =============================================================================
#  Buscar mejores parámetros
# =============================================================================

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense

print("Comenzando Grid_search\n")
t.tic()

def build_classifier(optimizer):
    classifier = Sequential()
#1
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19))
    classifier.add(Dropout(rate = 0.3))
#2
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.3))
#3
    classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.3))
#4
    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.3))
#5
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))    
    classifier.add(Dropout(rate = 0.3))
#6
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [16,32, 64],
              'epochs': [15, 25, 40],
                'optimizer': ['adam', 'adamax','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
#                           scoring = 'accuracy',
                           cv = 15,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_parameters  = grid_search.best_params_
print(f"best_parameters = {grid_search.best_params_}")
print(f"best_accuracy =   {grid_search.best_score_}")
t.toc('Finalizado - Grid_search')

# =============================================================================
# Cross Validation
# =============================================================================

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

t.tic() 
print("\nEntrando en Cross Validation\n")

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19))
    classifier.add(Dropout(rate = 0.3))
    
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.3))
    
    classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.3))
    
    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.3))
    
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))   
    classifier.add(Dropout(rate = 0.3))

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = best_parameters['optimizer'], loss = 'categorical_crossentropy', metrics = ['accuracy'])
#    classifier.compile(optimizer = 'adamax', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = best_parameters['batch_size'], epochs = best_parameters['epochs'])
#classifier = KerasClassifier(build_fn = build_classifier, batch_size = 32, epochs = 25)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
ac = list(accuracies)
mean = accuracies.mean()
variance = accuracies.std()
t.toc('\nTiempo de red neuronal: ')

history = classifier.fit(X_test, y_test, batch_size = best_parameters['batch_size'], epochs = best_parameters['epochs'], validation_split=0.2)

from matplotlib import pyplot as plt

ax1 = plt.subplot(121)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='center right')
plt.grid()
ax1.set_ylim([0.9, 1.02])

plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='center right')
plt.grid()
plt.subplots_adjust(wspace =0.4, hspace= 2.5)
plt.savefig('accuracy_over_epochs_train.pdf')

# =============================================================================
#                             Prediction
# =============================================================================
for i in range(1,20):
    y_pred = classifier.predict(np.array([X_test[i]]))
    predictions = list(encoder.inverse_transform(y_pred))
    y_pred_prob = classifier.predict_proba(np.array([X_test[i]]))
    print(f"The position is: {predictions}, and its accuracy was: {np.amax(y_pred_prob):.3g}")

a = np.array([[-54,-77,-82,-85,-75,-75,-91,-63,-65,-67,-89,-68,-85,-83,-84,-83,-84,-83,-83]])
b = np.array([[-52,-70,-87,-85,-74,-75,-91,-61,-62,-67,-89,-67,-87,-85,-84,-84,-77,-77,-83]])
c = np.array([[-51,-70,-85,-73,-76,-63,-64,-67,-93,-68,-85,-80,-80,-81,-81,-81,-83,-82,-84]])

y_pred = classifier.predict(b)
predictions = list(encoder.inverse_transform(y_pred))
y_pred_prob = classifier.predict_proba(b)
print(f"The position is: {predictions}, and its accuracy was: {np.amax(y_pred_prob):.3g}")
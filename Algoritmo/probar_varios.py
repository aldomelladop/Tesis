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
from fixrows import fixrows
from merge_csv import fusionar_csv
# =============================================================================

# =============================================================================
# Importing the dataset
# =============================================================================
   
#Run this code, only if the file with all the dataframes were deleted
num_test = [1000,5000,10000,15000,20000,25000,30000]

for i in num_test:

    # =============================================================================
    df1 = fixrows( 'Potencia_r1').iloc[:5000,:]
    num_row = np.shape(df1)[0]
    coords  = ['(1,0)' for j in range(num_row)]
    coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
    df1 = coords.join(df1, how='left')
    df1.to_csv('Potencia_R1.csv')
    # 
    df2 = fixrows( 'Potencia_r2').iloc[:5000,:]
    num_row = np.shape(df2)[0]
    coords  = ['(0,0)' for j in range(num_row)]
    coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
    df2 = coords.join(df2, how='left')
    df2.to_csv('Potencia_R2.csv')
     
    df3 = fixrows( 'Potencia_r3').iloc[:5000,:]
    num_row = np.shape(df3)[0]
    coords  = ['(0,1)' for j in range(num_row)]
    coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
    df3 = coords.join(df3, how='left')
    df3.to_csv('Potencia_R3.csv')

    df4 = fixrows( 'Potencia_r4').iloc[:5000,:]
    num_row = np.shape(df4)[0]
    coords  = ['(1,1)' for j in range(num_row)]
    coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
    df4 = coords.join(df4, how='left')
    df4.to_csv('Potencia_R4.csv')
      
    #Ingresar a carpeta potencias
    fusionar_csv('Potencia_R1','Potencia_R2','Potencia_R3','Potencia_R4')
    df0 = fixrows('potencias_fusionado').iloc[3:,1:]

    df0 = pd.read_csv('potencias_fusionado_corregido.csv').iloc[3:,1:]

    X = df0.iloc[:,1:].values #variables Dependientes (Potencias)
    y = df0.iloc[:,0].values #values Independientes (Posición)

    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoder.fit(y)
    y_encoded = encoder.transform(y)
    y_encoded = np_utils.to_categorical(y_encoded)

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
        classifier.add(Dropout(p = 0.3))
    #2
        classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dropout(p = 0.3))
    #3
        classifier.add(Dense(units = 25, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dropout(p = 0.3))
    #4
        classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dropout(p = 0.3))
    #5
        classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))    
        classifier.add(Dropout(p = 0.3))
    #6
        classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
        classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return classifier
    classifier = KerasClassifier(build_fn = build_classifier)
    parameters = {'batch_size': [16,32, 64],
                  'epochs': [20, 45, 70],
                    'optimizer': ['adam', 'adamax','rmsprop']}

    grid_search = GridSearchCV(estimator = classifier,
                               param_grid = parameters,
    #                           scoring = 'accuracy',
                               cv = 15,
                               n_jobs = -1)

    grid_search = grid_search.fit(X_trainn, y_train)


    print(f"best_parameters = {grid_search.best_params_}")
    #globals()['best_param_{}'.format(i)] = globals()['best_param_{}'.format(i)].append(grid_search.best_params_)
    print(f"best_accuracy =   {grid_search.best_score_}")
    #globals()['accuracies_{}'.format(i)] = globals()['accuracies_{}'.format(i)].append(grid_search.best_score)


    # =============================================================================
    # Neural network to use
    # =============================================================================


    import keras
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score
    from keras.models import Sequential
    from keras.layers import Dropout
    from keras.layers import Dense

    def build_classifier():
        classifier = Sequential()
        classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19))
        classifier.add(Dropout(rate = 0.3))
        
        classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dropout(rate = 0.3))
        
        classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dropout(rate = 0.3))
        
        classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dropout(rate = 0.3))
        
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))   
        classifier.add(Dropout(rate = 0.3))

        classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
        classifier.compile(optimizer = 'adamax', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return classifier

    classifier = KerasClassifier(build_fn = build_classifier, batch_size = 16, epochs = 20)
    accuracies = cross_val_score(estimator = classifier, X = X_trainn, y = y_train, cv = 10, n_jobs = -1)
    mean = accuracies.mean()
    variance = accuracies.std()

    history = classifier.fit(X_trainn, y_train, batch_size = 16, epochs = 20, validation_split=0.2)

    # =============================================================================
    # Prediction
    # =============================================================================
    for i in range(1,20):
        y_pred = grid_search.predict(np.array([X_testn[i]]))
        predictions = list(encoder.inverse_transform(y_pred))
        y_pred_prob = grid_search.predict_proba(np.array([X_testn[i]]))
        print(f"The position is: {predictions}, and its accuracy was: {np.amax(y_pred_prob):.3g}")
        #global()['predictions_{}'.format(i)] = global()['predictions_{}'.format(i)].append((predi))

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
plt.savefig('accuracy_over_epochs_'+ str(i) +'.pdf')

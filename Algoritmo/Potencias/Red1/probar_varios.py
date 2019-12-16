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
import random
import numpy as np
import pandas as pd
from pytictoc import TicToc
from fixrows  import fixrows
from merge_csv import fusionar_csv

# =============================================================================
# Importing the dataset
# =============================================================================
   
#Run this code, only if the file with all the dataframes were deleted
num_test = [5000,10000,15000,20000,23000,25000,30000]

t = TicToc()

t.tic()
for j in num_test:

    print(f"Probando {j}\n")
    
    df1 = fixrows( 'Potencia_r1').iloc[:j,:]
    num_row = np.shape(df1)[0]
    coords  = ['(1,0)' for j in range(num_row)]
    coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
    df1 = coords.join(df1, how='left')
    df1.to_csv('Potencia_R1.csv')
     
    df2 = fixrows( 'Potencia_r2').iloc[:j,:]
    num_row = np.shape(df2)[0]
    coords  = ['(0,0)' for j in range(num_row)]
    coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
    df2 = coords.join(df2, how='left')
    df2.to_csv('Potencia_R2.csv')
     
    df3 = fixrows( 'Potencia_r3').iloc[:j,:]
    num_row = np.shape(df3)[0]
    coords  = ['(0,1)' for j in range(num_row)]
    coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
    df3 = coords.join(df3, how='left')
    df3.to_csv('Potencia_R3.csv')

    df4 = fixrows( 'Potencia_r4').iloc[:j,:]
    num_row = np.shape(df4)[0]
    coords  = ['(1,1)' for j in range(num_row)]
    coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
    df4 = coords.join(df4, how='left')
    df4.to_csv('Potencia_R4.csv')
      
    #Ingresar a carpeta potencias
    t.tic()
    fusionar_csv('Potencia_R1','Potencia_R2','Potencia_R3','Potencia_R4')
    t.toc('\nTiempo Archivos Fusionados\n')
    df0 = fixrows('potencias_fusionado').iloc[3:,1:]

#    df0 = pd.read_csv('potencias_fusionado_corregido.csv').iloc[3:,1:]

    X = df0.iloc[:,1:].values #variables Dependientes (Potencias)
    y = df0.iloc[:,0].values #values Independientes (Posición)

    import keras
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
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import GridSearchCV
    from keras.models import Sequential
    from keras.layers import Dropout
    from keras.layers import Dense

    print("Comenzando cross_validation\n")
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
        classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
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
                  'epochs': [20, 45, 70],
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

    t.toc('\nTiempo en cross_validation\n')

    # =============================================================================
    # Neural network to use
    # =============================================================================

    
    from sklearn.model_selection import cross_val_score

    t.tic()
    print("Entrando en Red Neuronal\n")
    
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
        classifier.compile(optimizer = best_parameters['optimizer'], loss = 'categorical_crossentropy', metrics = ['accuracy'])
        return classifier

    classifier = KerasClassifier(build_fn = build_classifier, batch_size = best_parameters['batch_size'], epochs = best_parameters['epochs'])
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
    ac = list(accuracies)
    mean = accuracies.mean()
    variance = accuracies.std()
    t.toc('\nTiempo de red neuronal: ')

    history  = classifier.fit(X_train, y_train, batch_size = best_parameters['epochs'], epochs = best_parameters['epochs'], validation_split=0.2)

    # =============================================================================
    # Graficos
    # =============================================================================
    from matplotlib import pyplot as plt
    
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='center right')
    plt.grid()
    
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='center right')
    plt.grid()
    plt.subplots_adjust(wspace =0.4, hspace= 2.5)
    
    directory = os.getcwd()

    if os.path.isdir(os.getcwd() + '/' + str(j)) == True:
            pass
    else:
            os.mkdir(os.getcwd() + '/' + str(j))
            
    plt.savefig( os.getcwd() + '/' + str(j) + '/accuracy_over_epochs_train_' + str(j) + '.pdf')
        
    # =============================================================================
    #     Escritura de archivo
    # =============================================================================
    print(f"\nnp.shape(X_test)[0] = {np.shape(X_test)[0]}\n")

    outFileName= directory + '/' + str(j) + "/resultados_"+str(j)+".txt"
    f = open(outFileName,"w")
    f.write("El número de elementos usados es: " + repr(j) +'\n'
            "Los mejores parámetros son: "+ repr(best_parameters) +'\n'
            "La media obtenida es: " + repr(mean) + '\n'
            )
    f.write("La varianza obtenida es: " + repr(variance) + '\n')
    
    for i in ac:
        f.write("\tac: " +  repr(round((i*100),2)) +"%" + '\n')

    # =============================================================================
    #                             Prediction
    # =============================================================================
    for i in range(1,15):
    	r = random.randint(0, np.shape(X_test)[0]) 
    	y_pred = grid_search.predict(np.array([X_test[r]]))
    	predictions = list(encoder.inverse_transform(y_pred))
    	y_pred_prob = grid_search.predict_proba(np.array([X_test[i]]))
    	f.write("\nFor the vector: ["+ repr(X_test[r])+ "]\t the predicted position is:" +  repr(predictions) +  "and its accuracy was:" + repr(round(np.amax(y_pred_prob),2)))
    f.close()

    print("Archivo escrito\n")
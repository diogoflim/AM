import pandas as pd
import numpy as np

def normaliza_maxmin (M):
    return (M - np.min(M))/(np.max(M)-np.min(M))

def sigmoid (z):
    return 1/(1 + np.exp(-z)) 

def inicializa_parametros (d):
    w = np.zeros(d)
    b = 0
    return w, b

def fit (X_train, y_train, w, b, n_iter, taxa_aprendizado):
    n = X_train.shape[0]
    d = X_train.shape[1]
    
    cost = []
    for i in range(n_iter):            
        
        Z = X_train.dot(w) + b
        A = sigmoid (Z)
        
        #Calcula o custo e anexa a lista
        J = (-1/n) * np.sum(y_train * np.log(A) + (1-y_train) * np.log(1-A))
        
        cost.append(J) 

        # Calcula dz
        dz = (A - y_train.T)        
        dz = np.reshape(dz, n)        
        
        #Calcula dw
        dw = np.dot(X_train.T, dz)/n         
        db = np.sum(dz)/n 
        
        #Atualiza parâmetros
        w = w - taxa_aprendizado * dw  
        b = b - taxa_aprendizado * db      
                       
    return w, b, cost

def previsao(X_test, w_final, b_final):    
        Z = X_test.dot(w_final) + b_final
        A = sigmoid(Z)        
        y_previsto = np.where(A > 0.5, 1, 0)        
        return A, y_previsto

def acuracia(y_test, y_previsto):
    n=len(y_test)
    conta_correto = 0
    for i in range(n):
        if y_test[i] == y_previsto[i]:
            conta_correto += 1
    return conta_correto / n
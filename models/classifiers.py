#%%
import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

distance = ['eucliean', 'deepmetric']
similarity = ['cosine']
metric = {'euclidean':euclidean_distances, 'cosine':cosine_similarity, 'deepmetric': None}

def measure(x, y, method='similarity'):
    
    if method != 'deepmetric':
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
    
    return metric[method](x, y)

def signature(sn, nu, method='similarity'):

    coef = None
    if method in distance:
        coef = 1
    else: coef = -1
 
    if  sn < nu:
        return 1*coef
    else: return -1*coef

def predict_example(spreader, no_spreader, u, checkp=0.25, method='euclidean'):
    
    spreader_aster = spreader[list( np.random.choice( range(len(spreader)), int(checkp*len(spreader)), replace=False) )]

    y_hat = 0
    for s in spreader_aster:
        
        no_spreader_aster = no_spreader[list(np.random.choice( range(len(spreader)), int(checkp*len(no_spreader)), replace=False))]
        y_hat_aster = 0
        sn = measure(s, u, method)
        for n in no_spreader_aster:
            nu = measure(n, u, method)

            y_hat_aster += signature(sn, nu, method)
        y_hat = y_hat + (y_hat_aster >= 0) - (y_hat_aster < 0)
    # print(y_hat, len(spreader_aster))
    return (y_hat >= 0)


def K_Impostor(spreader, no_spreader, unk, checkp=0.25, method='euclidean', model=None):

    if method == 'deepmetric':
        metric['deepmetric'] = lambda x, y : model.forward(torch.unsqueeze(torch.tensor(x), 0), torch.unsqueeze(torch.tensor(y), 0))

    Y = np.zeros((len(unk), ))
    for i, u in zip(range(len(unk)), unk):
        Y[i] = predict_example(spreader, no_spreader, u, checkp, method)
    # print(Y)
    return Y


spreader = np.random.randn(100, 64)
no_spreader = np.random.randn(100, 64)
unk = np.random.randn(100, 64)



# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 08:52:08 2020

@author: RAVJ


Script to check internal methods
"""
import PuntosInteriores as PI
import Descenso2Pasos as d2p
import numpy as np
import pandas as pd

# =============================================================================
# MAIN
# =============================================================================
#solve paraboloid in the first cuadrant
A = np.array([[1, 1, 1], 
              [1, -1, 1]])
Q = 2 * np.eye(3)
c = np.zeros((3,1))
b = np.ones((2,1))

x, y, mu = PI.interior_points(Q, A, c, b, plot=True)

# %% Pruebas para el descensop en 2 pasos
df_ratings_raw = pd.read_csv('Data/ratings_small.csv')

# change from long to wide
df_ratings = df_ratings_raw.pivot(index='userId',
                                  columns='movieId',
                                  values='rating')

X_ratings = df_ratings.to_numpy()

# %% hiperparams1
# optimize
np.random.seed(42)
k = 2
lambd = 1
maxiter = 4
tol = 1e-4

W, H = d2p.gradient2steps(X_ratings, k, lambd, maxiter, tol)


# %% hiperparams2
# optimize
np.random.seed(42)
k = 20
lambd = 1
maxiter = 4
tol = 1e-4

# W, H = d2p.gradient2steps(X_ratings, k, lambd, maxiter, tol)

# %% hiperparams3
# optimize
np.random.seed(42)
k = 20
lambd = 0
maxiter = 4
tol = 1e-4

# W, H = d2p.gradient2steps(X_ratings, k, lambd, maxiter, tol)

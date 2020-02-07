import numpy as np
import numpy.linalg as la
from .. import extensions


def flip_and_swap(X, x_flip, y_flip, z_flip):
    X = extensions.numpy.atleast_float_2d(X)
    X_clone = np.copy(X)
    x = np.copy(X_clone[:, 0])
    y = np.copy(X_clone[:, 1])
    z = np.copy(X_clone[:, 2])
    x_flip = x_flip.lower()
    y_flip = y_flip.lower()
    z_flip = z_flip.lower()
    if x_flip == 'x':
        X_clone[:, 0] = x
    elif x_flip == '-x':
        X_clone[:, 0] = -x
    elif x_flip == 'y':
        X_clone[:, 0] = y
    elif x_flip == '-y':
        X_clone[:, 0] = -y
    elif x_flip == 'z':
        X_clone[:, 0] = z
    elif x_flip == '-z':
        X_clone[:, 0] = -z

    if y_flip == 'x':
        X_clone[:, 1] = x
    elif y_flip == '-x':
        X_clone[:, 1] = -x
    elif y_flip == 'y':
        X_clone[:, 1] = y
    elif y_flip == '-y':
        X_clone[:, 1] = -y
    elif y_flip == 'z':
        X_clone[:, 1] = z
    elif y_flip == '-z':
        X_clone[:, 1] = -z

    if z_flip == 'x':
        X_clone[:, 2] = x
    elif z_flip == '-x':
        X_clone[:, 2] = -x
    elif z_flip == 'y':
        X_clone[:, 2] = y
    elif z_flip == '-y':
        X_clone[:, 2] = -y
    elif z_flip == 'z':
        X_clone[:, 2] = z
    elif z_flip == '-z':
        X_clone[:, 2] = -z

    return X_clone


def vector_magnitude(X):
    X = extensions.numpy.atleast_float_2d(X)
    result = la.norm(X, ord=2, axis=1, keepdims=True)
    return result

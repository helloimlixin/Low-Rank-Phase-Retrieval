#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:03:27 2021

@author: soominkwon
"""

import numpy as np

def rand_src(alphabet, n1, n2):
    """Helper function to simulate a random signal source that generates the
    modulation signal from a prescribed alphabet randomly.

    Args:
        alphabet (list): a prescribed alphabet of modulation codes
        n1 (int): the first dimension of the video frame
        n2 (int): the second dimension of the video frame

    Returns:
        numpy array: an n1 x n2 numpy array representing the modulation matrix
    """
    return np.random.choice(alphabet, size=(n1, n2), replace=True)

def generate_cdp_masks(alphabet, n1, n2, L):
    """Helper function to generate the Coded Diffraction Pattern Measurement
    masks with the prescribed alphabet.

    Args:
        frame (numpy array): raw image frame
        alphabet (list): a prescribed alphabet of modulation codes

    Returns:
        numpy array: modulation mask matrix
    """
    masks = np.zeros((n1, n2, L), dtype=complex)

    for l in range(L):
        masks[:, :, l] = rand_src(alphabet, n1, n2)

    return np.reshape(masks, (n1, n2, L))

def generateLRPRMeasurements(image_name, m_dim, L):
    """ Function to obtain measurements y's (m x q) and A's (m x n x q).
    
        Arguments:
            image_name: name of .npz file to load (n1 x n2 x q)
            m_dim: dimensions of m
    
    """
    
    with np.load(image_name) as data:
        tensor = data['arr_0']
    
    n1, n2, q_dim = tensor.shape
    n_dim = n1 * n2
    vec_images = np.reshape(tensor, (n_dim, q_dim), order='F')
    
    A_tensor = np.zeros((n_dim, m_dim, q_dim), dtype=complex)
    Y = np.zeros((m_dim, q_dim))

    # Prescribed alphabet.
    alphabet = [1, -1, complex(0, 1), complex(0, -1)]

    # Modulation patterns.
    D = generate_cdp_masks(alphabet, n1, n2, L)
    # DFT matrix.
    f = np.fft.fft2(np.eye(n_dim, q_dim))

    for k in range(q_dim):
        # A_k = f_k * D_l
        # where f_k is the kth row of the DFT matrix, and D_l is the ith frontal
        # slice of the modulation tensor.
        A_k = np.multiply(f[:, k], np.diag(np.reshape(D[:, :, 0], (n_dim,))))
        
        for l in range(1, L):
            A_k = np.hstack((A_k, np.multiply(
                f[:, k], np.diag(np.reshape(D[:, :, l], (n_dim,))))))

        A_tensor[:, :, k] = A_k
        x_k = vec_images[:, k]
        
        norm_x_k = np.linalg.norm(x_k)
        x_k = x_k / norm_x_k

        # Perform the 2D Discrete Fourier Transform.
        Ax_k = np.fft.fft2(
            np.multiply(D, np.reshape(np.tile(x_k, (1, L)), (n1, n2, L))))

        y_k = np.abs(np.reshape(Ax_k, (n_dim * L)))**2
        Y[:, k] = y_k
    
    return tensor, Y, A_tensor

#! /usr/bin/env python

"""
Obtain low rank approximation based on L1 and L2 norm. 
The function is inspired by the VIP-HCI Annular PCA function
    https://github.com/vortex-exoplanet/VIP

Cite: 

"""

__author__ = 'Hazan Daglayan'
__all__ = ['norm_low_rank', 'low_rank_adi']

import numpy as np
from multiprocessing import cpu_count
from vip_hci.preproc import (cube_derotate, cube_collapse)
from vip_hci.var import get_annulus_segments
from l1lracd import L1LRAcd
from vip_hci.psfsub.svd import svd_wrapper

def norm_low_rank(cube, angle_list, inner_radius=0, outer_radius=None, asize=4, 
                ncomp=1, norm=2, nproc=1, imlib='vip-fft', svd_mode='lapack', 
                collapse='median', full_output=False):
    """ 
    Parameters
    ----------
    cube : numpy ndarray, 3d or 4d
        Input cube.
    angle_list : numpy ndarray, 1d
        Corresponding parallactic angle for each frame.
    inner_radius : float, optional
        The radius of the innermost annulus. 
    outer_radius : float, optional
        The radius of the outermost annulus. If it is not given, it is used as the 
        half of the image. If it is given, the annuli between inner_radius and 
        outer_radius are determined.
    asize : float, optional
        The size of the annuli, in pixels.
    ncomp : int, optional
        the number of PCs.
    norm : 1 for L1-LRA, 2 for PCA 
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to (cpu_count()/2).
    imlib : str, optional
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    svd_mode : {'lapack', 'arpack', 'eigen', 'randsvd', 'cupy', 'eigencupy',
        'randcupy', 'pytorch', 'eigenpytorch', 'randpytorch'}, str optional
        Switch for the SVD method/library to be used. It calculates using VIP_HCI
    collapse : {'median', 'mean', 'sum', 'trimmean'}, str optional
        Sets the way of collapsing the frames for producing a final image.
    full_output: boolean, optional
        Whether to return the final median combined image only or with 
        residual cube and rotated residual cube.

    Returns
    -------
    frame : numpy ndarray, 2d
        [full_output=False] Median combination of the de-rotated cube.
    array_out : numpy ndarray, 3d or 4d
        [full_output=True] Cube of residuals.
    array_der : numpy ndarray, 3d or 4d
        [full_output=True] Cube residuals after de-rotation.
    frame : numpy ndarray, 2d
        [full_output=True] Median combination of the de-rotated cube.
    """


    # 3D cube
    if cube.ndim == 3:
        res = low_rank_adi(cube, angle_list, inner_radius, outer_radius, asize,
                    ncomp, norm, nproc, imlib, svd_mode, collapse, full_output)

        cube_out, cube_der, frame = res
        if full_output:
            return cube_out, cube_der, frame
        else:
            return frame

    # 4D cube, but no mSDI desired
    elif cube.ndim == 4:
        nch, nz, ny, nx = cube.shape
        cube_frames = np.zeros([nch, ny, nx])

        cube_out = []
        cube_der = []
        # ADI in each channel
        for ch in range(nch):
            res = low_rank_adi(cube[ch], angle_list, inner_radius, outer_radius, asize, 
                            ncomp, norm, nproc, imlib, svd_mode, collapse, full_output)
            cube_out.append(res[0])
            cube_der.append(res[1])
            cube_frames[ch] = res[-1]

        frame = cube_collapse(cube_frames, mode=collapse)

        # convert to numpy arrays
        cube_out = np.array(cube_out)
        cube_der = np.array(cube_der)
        if full_output:
            return cube_out, cube_der, frame
        else:
            return frame

    else:
        raise TypeError('Input array is not a 4d or 3d array')


################################################################################
# Function solves the low rank approximation problem according to norm
################################################################################

def low_rank_adi(cube, angle_list, inner_radius=0, outer_radius=None, asize=4, ncomp=1, 
                norm=1, nproc=None, imlib='vip-fft', svd_mode='lapack', collapse='median', 
                full_output=True):
    if cube.ndim != 3:
        raise TypeError("Input array should be a 3d array")
    if cube.shape[0] != len(angle_list):
        msg = "Number of frames ({}) and number of parallactic angles "
        msg += "({}) should be same".format(cube.shape[0], len(angle_list))
        raise TypeError(msg)
    
    nfr, y, _ = cube.shape
    
    if outer_radius is not None:
        n_annuli = round((outer_radius - inner_radius) / asize)
    else:
        n_annuli = int((y / 2 - inner_radius) / asize)
    
    if nproc is None:   
        nproc = cpu_count() // 2
    
    cube_out = np.zeros_like(cube)
    for ann in range(n_annuli):
        inner_radius_ = inner_radius + ann * asize
        
        indices = get_annulus_segments(cube[0], inner_radius_, asize, 1, 0)
        
        yy = indices[0][0]
        xx = indices[0][1]
        matrix_segm = cube[:, yy, xx]  

        if norm == 1:        
            U_, S_, VT_ = np.linalg.svd(matrix_segm,full_matrices=0)
            V = np.dot(np.diag(S_[:ncomp]),VT_[:ncomp])  #truncated SVD
            U = U_[:, :ncomp]
            U,V = L1LRAcd(matrix_segm, ncomp, maxiter=5, U0=U, V0=V)
            reconstructed = np.dot(U,V)
        elif norm == 2:
            V = svd_wrapper(matrix_segm, svd_mode, ncomp, verbose=False)
            transformed = np.dot(V, matrix_segm.T)
            reconstructed = np.dot(transformed.T, V)
        else:
            raise TypeError("The norm should be 1 or 2.")
        
        residuals = cube[:,yy,xx] - reconstructed

        for fr in range(nfr):
            cube_out[fr][yy, xx] = residuals[fr]


    # Cube is derotated according to the parallactic angle and collapsed
    cube_der = cube_derotate(cube_out, angle_list, nproc=nproc, imlib=imlib)
    frame = cube_collapse(cube_der, mode=collapse)

    if full_output:
        return cube_out, cube_der, frame
    else:
        return frame
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Obtain L1 norm low rank approximation. This code uses and Python version of the MATLAB code provided
by Nicolas Gillis, Component-Wise l1-Norm Low-Rank Matrix Approximation  
https://sites.google.com/site/nicolasgillis/code?authuser=0

Cite: 

"""

__author__ = 'Hazan Daglayan'
__all__ = ['L1LRAcd', 'wmedian']
from hciplot import plot_frames
from matplotlib.pyplot import *
from matplotlib import pyplot as plt

import vip_hci as vip

from vip_hci.var import frame_center 
import matplotlib.pyplot as plt

from vip_hci.var import get_annulus_segments

import numpy as np
from copy import deepcopy
from photutils import detect_sources
from vip_hci.var import get_circle





def remove_blob(frame_loc, blob_xy, fwhm):
    frame_mask = get_circle(np.ones_like(frame_loc), radius=fwhm/2,
                                cy=blob_xy[1], cx=blob_xy[0],
                                mode="mask")
    if ~np.isnan(frame_loc[frame_mask==1]).any():
        frame_loc[frame_mask==1] = np.nan
        return True
    else:
        return False
def remove_blob_enforce(frame_loc, blob_xy, fwhm):
    frame_mask = get_circle(np.ones_like(frame_loc), radius=fwhm/2,
                                cy=blob_xy[1], cx=blob_xy[0],
                                mode="mask")

    frame_loc[frame_mask==1] = np.nan
    return True

# This function is taken from the inner funcion of vip_hci.metrics.roc.compute_binary_map
def _overlap_injection_blob(injection, fwhm, blob_mask):

    injection_mask = get_circle(np.ones_like(blob_mask), radius=fwhm,
                                cy=injection[1], cx=injection[0],
                                mode="mask")

    intersection = injection_mask & blob_mask
    smallest_area = min(blob_mask.sum(), injection_mask.sum())
    return intersection.sum() / smallest_area



import itertools
marker = itertools.cycle(('s', 'p', 'o', '*')) 
    
    
# --------------------------------------------------------------------------
cmap = plt.get_cmap('tab10')


plot=False

def build_roc_curve(loglrmaps, pos_yxs, thresholds=None, fwhm=4, sqrt=True, plot=False, vmax=30, label=None, fig=None):
    
    if thresholds is None:
        thresholds = (np.append(0, np.arange(0,301,5)+0.1))

    total_det = np.zeros((1,len(thresholds)))
    total_fps = np.zeros((1,len(thresholds)))
    total_fprl2l1 = np.zeros((1,len(thresholds)))

            
        
                
        
    for i, th in enumerate(range(0, 900, 36)):
        loglr21_1 = loglrmaps[i]            
        frame_loc = deepcopy(loglr21_1)
        cy, cx = frame_center(frame_loc)
        m, n = frame_loc.shape
        mask = get_annulus_segments(np.ones_like(frame_loc), fwhm*1,n/2, mode="mask")
        frame_loc[mask[0]==0] = np.nan
        for j in range(len(pos_yxs[0])):
            remove_blob_enforce(frame_loc, (pos_yxs[i][j][0][1], pos_yxs[i][j][0][0]) , fwhm*1.5)
        if plot:
            plot_frames(loglr21_1, title=label, vmax=vmax)
            plot_frames(frame_loc, title=label, vmax=vmax)
                
            
        falses = []
        for sep_i in range(2, int((n/2)/fwhm)):#range(10,11):
            source_rad = (sep_i)*fwhm

            sourcex, sourcey = vip.var.pol_to_cart(source_rad, th)
            sourcey, sourcex = sourcey+cy, sourcex+cx 
        
            sep = vip.var.dist(cy, cx, float(sourcey), float(sourcex))
        
            angle = np.arcsin(1.5*fwhm/sep)*1
            number_apertures = int(np.floor(2*np.pi/angle))
        
        
            yy = np.zeros((number_apertures))
            xx = np.zeros((number_apertures))
            cosangle = np.cos(angle)
            sinangle = np.sin(angle)
            xx[0] = sourcex - cx
            yy[0] = sourcey - cy
            for ap in range(number_apertures-1):
                xx[ap+1] = cosangle*xx[ap] + sinangle*yy[ap]
                yy[ap+1] = cosangle*yy[ap] - sinangle*xx[ap]                 
        
            xx += cx
            yy += cy

        
            for r_i in range(0,number_apertures):
                if remove_blob(frame_loc, (xx[r_i],yy[r_i]), fwhm*1.5):
                    falses.append((xx[r_i],yy[r_i]))

        list_detections=[]
        list_fps = []
        list_fpr = []
        
        debug=False 
        injections = []
        for pos in pos_yxs[i]:
            y,x = pos[0][0], pos[0][1]
            injections.append((x,y))
        for ithr, threshold in enumerate(thresholds):
            if debug:
                print("\nprocessing threshold #{}: {}".format(ithr + 1, threshold))
        
            segments = detect_sources(loglr21_1, threshold-0.0001, 1, connectivity=4)
            detections = 0
            fps = 0
            if segments != []:
                binmap = (segments.data != 0)
        
            
                for injection in injections:
                    overlap = _overlap_injection_blob(injection, fwhm, binmap)
                    if overlap > 0:
                        if debug:
                            print("\toverlap of {}! (+1 detection)"
                                    "".format(overlap))
        
                        detections += 1
                for false in falses:
                    overlap = _overlap_injection_blob(false, fwhm, binmap)
                    if overlap > 0:
                        fps = fps+1 

            list_detections.append(detections)
            list_fps.append(fps)
            list_fpr.append(fps/(len(falses)))
        
        
        total_det += np.array(list_detections)
        total_fps += np.array(list_fps)
        total_fprl2l1 += np.array(list_fpr)

    total_tprl2l1 = total_det/100

    #fig, ax = fig.subplots()

    
    if sqrt:
        plt.plot(np.sqrt((total_fprl2l1/25).T),np.sqrt(total_tprl2l1.T), '-', label=label, markersize=5)#, marker=next(marker)), color=cmap(vers-1))
        plt.ylabel(r'$\sqrt{\rm TPR}$')
        plt.xlabel(r'$\sqrt{\rm FPR}$')
    else:
        plt.plot((total_fprl2l1/25).T,np.sqrt(total_tprl2l1.T), '-o',label=label, markersize=10)#, marker=next(marker))#, color=cmap(vers-1))
        plt.xlabel('FPR')
        plt.xlabel('TPR')
        
        
    plt.legend()
    return fig




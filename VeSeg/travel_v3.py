#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 07:56:27 2020

@author: mib
"""
import nibabel as nib

import cv2
from skimage import measure, filters, morphology, exposure, restoration
import numpy as np
from scipy.ndimage.morphology import binary_closing, binary_fill_holes, binary_erosion, binary_dilation
from scipy.ndimage import label
import scipy.ndimage as ndi
from scipy.signal import fftconvolve
from skimage.morphology import skeletonize_3d
from skimage.feature import structure_tensor, structure_tensor_eigenvalues


import mclahe

import time



def rescale_affine(affine, shape, zooms, new_shape=None):
    """ Return a new affine matrix with updated voxel sizes (zooms)
    This function preserves the rotations and shears of the original
    affine, as well as the RAS location of the central voxel of the
    image.
    Parameters
    ----------
    affine : (N, N) array-like
        NxN transform matrix in homogeneous coordinates representing an affine
        transformation from an (N-1)-dimensional space to an (N-1)-dimensional
        space. An example is a 4x4 transform representing rotations and
        translations in 3 dimensions.
    shape : (N-1,) array-like
        The extent of the (N-1) dimensions of the original space
    zooms : (N-1,) array-like
        The size of voxels of the output affine
    new_shape : (N-1,) array-like, optional
        The extent of the (N-1) dimensions of the space described by the
        new affine. If ``None``, use ``shape``.
    Returns
    -------
    affine : (N, N) array
        A new affine transform with the specified voxel sizes
    """
    shape = np.array(shape, copy=False)
    new_shape = np.array(new_shape if new_shape is not None else shape)

    s = nib.affines.voxel_sizes(affine)
    rzs_out = affine[:3, :3] * zooms / s

    # Using xyz = A @ ijk, determine translation
    centroid = nib.affines.apply_affine(affine, (shape - 1) // 2)
    t_out = centroid - rzs_out @ ((new_shape - 1) // 2)
    return nib.affines.from_matvec(rzs_out, t_out)


def preprocessing(img_nii):
    mask = np.zeros_like(img_nii, dtype = np.uint8)
    avg_otsu = 0.0
    # slice wise iteration
    # TODO make this faster!
    nr_slices = np.shape(img_nii)[2]
    for s in range(nr_slices):
        img = img_nii[:,:,s] * 255.0
        img = img.astype(np.uint8)

        # Otsu's thresholding
        otsu_th, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        avg_otsu += otsu_th

        # morph operations
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

        # compute contours
        contours, hierarchy = cv2.findContours(otsu.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # compute max contour area id
        areas = []
        area_idx = []
        hidx = 0
        while hidx >= 0:
            cnt = contours[hidx]
            area = np.abs(cv2.contourArea(cnt))
            areas.append(area)
            area_idx.append(hidx)
            hidx = hierarchy[0][hidx][0]
        areaidx = np.argmax(areas)
        areaidx = area_idx[areaidx]

        cont = np.zeros_like(img)
        # draw mask of biggest area
        cont_umat=cv2.drawContours(cv2.UMat(cont), contours, areaidx, (255, 255, 255), -1)
        cont = cont_umat.get()
        # filter slice on mask
        masked = np.maximum((255 - cont).astype(np.uint8), (img_nii[:,:,s] * 255).astype(np.uint8))
        mask[:,:,s] = np.array(masked)

    avg_otsu /= 1.0 * nr_slices
    return mask, avg_otsu


def largest_label_volume(im, background=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != background]
    vals = vals[vals != background]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung(img, threshold):
    thresholded = np.zeros_like(img, dtype = np.uint8)
    nr_slices = np.shape(img)[2]
    # slice wise thresholding with global avg. otsu
    for s in range(nr_slices):
        otsu_th, otsu = cv2.threshold(img[:,:,s], int(threshold), 255, cv2.THRESH_BINARY)
        thresholded[:,:,s]=  255 - otsu

    # compute connected regions
    labels, numlabels = measure.label(thresholded, background=0, return_num=True)
    # get biggest volume
    l_max = largest_label_volume(labels, background=0)
    mask = np.zeros_like(thresholded, dtype=np.uint8)
    if l_max is not None:  # There is at least one ait pocket
        mask[labels == l_max] = 255

    return mask

def slice_binary_closing(img):
    
    nr_slices = np.shape(img)[2]
    img_close = np.zeros_like(img, dtype = bool)
    # slice wise thresholding with global avg. otsu
    for s in range(nr_slices):
        
        #img_close[:,:,s]= binary_fill_holes(binary_closing(img[:,:,s], iterations=2))
        img_close[:,:,s]= binary_fill_holes(img[:,:,s])
    
    return img_close

def getInsert(labels, nr_objects, Size):
    '''Extrahiert von den zuvor detektierten Strukturen (gespeichert in labels, Anzahl der Strukturen: nr_objects),
    das gewünschte Objekt. Die Auswahl erfolgt aufgrund der Größe der Objekte. Pos = -1 heißt hier extrahiere das größte
    und Pos = 0 das kleinste Objekt.
    Extracts from the previously detected structures (stored in labels, number of structures: nr_objects),
    the desired object. The selection is based on the size of the objects. Pos = -1 here means extract the largest
    and Pos = 0 the smallest object.
    
    Parameters:
    -----------
    labels, nr_objects: output of the scipy ndimage function "label"
    
    Size:  Size of the object to extract. -1: largest object, 0: smallest object (numpy array [Size])
    
    Returns:
    --------
    
    I: The object selected due to its size
    '''
    
    hist = np.histogram(labels.ravel(),bins=np.arange(-0.5,nr_objects+1,1))
    Lbl = np.squeeze(np.where(hist[0]==np.sort(hist[0][1::])[Size]))
    if np.shape(Lbl):
        print("Warnung: Es wurden mehrere gleich großes Labels detektiert")
        Lbl = Lbl[0]
    I = labels == Lbl
    
    return I 

def tol_region_growing(img, msk, tol_p, tol_m, abs_val=False):
    
    if abs_val:
        val_max = tol_p
        val_min = tol_m
    else:
        val_max = np.max(img[msk]) + tol_p
        val_min = np.min(img[msk]) - tol_m
        
    img_thr = np.logical_and(img <= val_max, img >= val_min)
    
    labels, nr_objects = label(img_thr, structure=np.ones((3,3,3)))
    unique_elements, counts_elements = np.unique(labels[msk], return_counts=True)
    
    #all labels with connection are allowed
    msk_gro = np.isin(labels,unique_elements[np.where(unique_elements>0)])
    
    return msk_gro

def ves_cor(msk_ves, size=4):
    
    #remove all too large areas
    msk_cor = binary_dilation(binary_erosion(msk_ves, iterations=size), iterations=size)
    
    msk_cor = np.logical_and(msk_ves, np.logical_not(msk_cor))
    
    #remove areas below a minimum size
    labels, nr_objects = label(msk_cor)    
    unique_elements, counts_elements = np.unique(labels, return_counts=True)
    msk_cor = np.isin(labels,unique_elements[np.where(np.logical_and(counts_elements>(size**3)*3, unique_elements > 0))])
    
    return msk_cor

def adp_res(iterations, fac_mrp = 1):
    return int(round(iterations*fac_mrp))

def itr_ves_grow(raw, msk_ves, max_val, min_val, step, cor_thr =-200, cor_msk=0, ero=1):
    
    if np.sum(cor_msk)==0:
        cor_msk_flg = False
    else:
        cor_msk_flg = True
    
    min_HUs = np.arange(min_val, max_val-1, step)[::-1]
    
    msk = np.array(msk_ves)
    
    msk_HU = raw>cor_thr
    
    # if cor_msk_flg:
    #      cor_msk_dil = binary_closing(cor_msk, iterations=adp_res(1),structure = np.ones((3,3,3)))
    
    for i in min_HUs:
        tmp = tol_region_growing(raw, msk, tol_p=max_val, tol_m = i, abs_val=True)
        #Gefäße sind nicht großflächig
        msk_ves_cor = np.logical_and(tmp, np.logical_not(msk))
        msk_ves_cor = binary_dilation(binary_erosion(msk_ves_cor, iterations=adp_res(ero)),iterations=adp_res(ero+3)) #*binary_erosion(msk, iterations=adp_res(2))

        if cor_msk_flg:
            msk_ves_cor[cor_msk] = False
            msk_ves_cor[np.logical_not(cor_msk)]=True
            
        #remove areas only below a minimum_HU
        msk_ves_cor[msk_HU] = False
        
        msk = np.logical_or(msk, np.logical_and(tmp, np.logical_not(msk_ves_cor)))
        
        if cor_msk_flg:
            #Add areas in cor_msk in any case if they are connected to msk
            msk_gro = np.logical_or(msk, cor_msk)
            labels, nr_objects = label(msk_gro, structure=np.ones((3,3,3)))
            unique_elements, counts_elements = np.unique(labels[msk], return_counts=True)
            
            #all labels with connection are allowed
            msk = np.isin(labels,unique_elements[np.where(unique_elements>0)])
        
    return msk

def itr_tra_grow(raw, msk_tra, max_val, min_val, step, cor_thr, cor_msk=0, min_HU=-1200, ero=1):
    
    if np.sum(cor_msk)==0:
        cor_msk_flg = False
    else:
        cor_msk_flg = True
        
    HUs = -np.arange(abs(max_val), abs(min_val), step)[::-1]
    
    msk = np.array(msk_tra)
    
    # if cor_msk_flg:
    #     cor_msk_dil = binary_closing(cor_msk, iterations=adp_res(1),structure = np.ones((3,3,3)))
    
    msk_HU = raw < cor_thr
    for i in HUs:
        tmp = tol_region_growing(raw, msk, tol_p=i, tol_m = min_HU, abs_val=True)
        #Vessels are not too large
        msk_ves_cor = np.logical_and(tmp, np.logical_not(msk))
        msk_ves_cor = binary_dilation(binary_erosion(msk_ves_cor, iterations=adp_res(ero)),iterations=adp_res(ero+3)) #*binary_erosion(msk, iterations=adp_res(2))

        if cor_msk_flg:
            msk_ves_cor[cor_msk] = False
            msk_ves_cor[np.logical_not(cor_msk)]=True
            
        #Apply corrections only below a HU threshold
        msk_ves_cor[msk_HU] = False
            
        msk = np.logical_or(msk, np.logical_and(tmp, np.logical_not(msk_ves_cor)))
        
        if cor_msk_flg:
            #Add areas in cor_msk in any case if they are connected to msk
            msk_gro = np.logical_or(msk, cor_msk)
            labels, nr_objects = label(msk_gro, structure=np.ones((3,3,3)))
            unique_elements, counts_elements = np.unique(labels[msk], return_counts=True)
            
            #all labels with connection are allowed
            msk = np.isin(labels,unique_elements[np.where(unique_elements>0)])
        
    return msk


def all_lung_msk(raw, luvox_msk, msk_tra, adv_den=False):
    
    '''
    DEBUG
    '''
    #raw=img_int; luvox_msk=luvox_msk_int; msk_tra=trachea_msk_int; adv_den=False
    
    # nii_img = nib.Nifti1Image(np.array(tube_meij_msk, dtype=np.uint8), affine_int)
    # nib.save(nii_img, pth_out+'tube_meij_msk.nii.gz')
    
    
    #add HU area
    img_con = mclahe.mclahe(raw, kernel_size=(33,33,33), n_bins=512, adaptive_hist_range=True)#exposure.rescale_intensity(raw, in_range=(-400, 100))
    img_con = exposure.rescale_intensity(img_con, in_range=(0.5, 0.9))
    img_con_thr =  filters.threshold_multiotsu(img_con,classes=5)
    img_con_msk = img_con > img_con_thr[3]
    
    
    img_con_lun = exposure.rescale_intensity(raw, in_range=(-400, 100))
    msk_lun = img_con_lun<filters.threshold_multiotsu(img_con_lun, classes=4)[0]
    msk_lun = np.logical_or(msk_lun, luvox_msk)
    labels, nr_objects = label(msk_lun)
    #welcher Bereich stimmt mit der Lunge überein
    unique_elements, counts_elements = np.unique(labels[luvox_msk], return_counts=True)
    msk_2 = np.isin(labels,unique_elements[np.where(unique_elements>0)])
    msk_2 = binary_fill_holes(msk_2)
    
    
    #edge detection
    #raw_edg = raw*binary_erosion(msk_2,iterations = adp_res(2))
    msk_2e = binary_erosion(msk_2,iterations = adp_res(2))
    raw_edg = img_con*msk_2e
    edg_x = ndi.filters.sobel(raw_edg, axis=0)
    edg_y = ndi.filters.sobel(raw_edg, axis=1)
    edg_z = ndi.filters.sobel(raw_edg, axis=2)
    
    edg_sob =np.sqrt((np.square(edg_x) + np.square(edg_y) + np.square(edg_z)))
    edg_sob = exposure.equalize_hist(edg_sob,mask=luvox_msk)
    
    thr_edg = filters.threshold_multiotsu(edg_sob, classes=4)
    msk_edg = binary_fill_holes(binary_closing(edg_sob > thr_edg[0],iterations=adp_res(5)))
    #Edges that probably belong to vessels
    msk_edg_ves = edg_sob > thr_edg[2]
    
    #lung mask including vessels:
    msk=binary_fill_holes(np.logical_or(msk_edg, msk_2))
    msk = np.logical_or(msk, luvox_msk)

    #considered shape of the lungs
    msk_luv_clo=binary_closing(luvox_msk, iterations=adp_res(14))
    
    #msk_lrg_ves = np.logical_and(msk_edg_ves_lrg, np.logical_and(msk_bro_dil, msk_luv_clo))
    msk_lrg_ves = np.logical_and(img_con_msk, msk_2)
    labels, nr_objects = label(msk_lrg_ves)    
    #which label has which size?
    unique_elements, counts_elements = np.unique(labels, return_counts=True)
    msk_lrg_ves = np.isin(labels,unique_elements[np.where(np.logical_and(counts_elements>200, unique_elements>0))])
    msk_lrg_ves = binary_fill_holes(msk_lrg_ves)
    

    msk_lun_ves = binary_closing(msk, iterations=adp_res(20))
    msk_lun_ves = binary_erosion(msk_lun_ves, iterations=adp_res(5))
    msk_lun_ves = np.logical_and(msk_lun_ves, np.logical_not(msk_tra))


    if adv_den:
        raw_gau = filters.gaussian(raw,sigma=1)
        raw_con = mclahe.mclahe(raw_gau, kernel_size=(23,23,23), n_bins=256, adaptive_hist_range=False)
        #raw_con_den = restoration.denoise_tv_chambolle(raw_con,weight=0.1)
        lpl = filters.laplace(raw_con)
        lpl_den = restoration.denoise_tv_chambolle(lpl,weight=0.05)
        lpl_thr = filters.threshold_multiotsu(lpl[msk_lun_ves],classes=4)
        #lpl_2 = exposure.rescale_intensity(lpl*msk, in_range=(0, lpl_thr[2]))
        ves_sml = morphology.white_tophat(lpl_den,selem=morphology.ball(3))
        ves_sml_msk = ves_sml > filters.threshold_multiotsu(ves_sml,classes=3)[0]
    else:
        lpl = filters.laplace(raw)
        lpl_thr = filters.threshold_multiotsu(lpl[msk_lun_ves],classes=4)
        lpl_2 = exposure.rescale_intensity(lpl*msk, in_range=(0, lpl_thr[2]))
        #remove artifacts
        lpl_2_den = restoration.denoise_tv_chambolle(lpl_2,weight=0.15)
        ves_sml = morphology.white_tophat(lpl_2_den,selem=morphology.ball(2))
        ves_sml_msk = ves_sml > filters.threshold_multiotsu(ves_sml,classes=5)[1]
        
    #Consider areas with low HU
    #msk_ves3 = itr_ves_grow(raw_ves, msk_ves2,100,-800,100,cor_thr=-200,cor_msk=ves_sml_msk)
    raw_ves = np.array(raw)
    raw_ves[np.logical_not(msk_lun_ves)] = np.min(raw)
    #msk_ves3 = itr_ves_grow(raw_ves, msk_lrg_ves,100,-100,100,cor_thr=100)
    msk_ves3 = itr_ves_grow(raw_ves, msk_lrg_ves,100,-800,100,cor_thr=-200,cor_msk=ves_sml_msk)
    msk_ves3 = np.logical_and(msk_ves3, np.logical_or(msk_lrg_ves, ves_sml_msk))
    
    #msk_ves4 = itr_ves_grow(raw_ves, msk_ves3,150,-50,200,cor_thr=200,ero=2)
    msk_ves3 = binary_fill_holes(msk_ves3)
    #remove areas below a minimum size
    labels, nr_objects = label(msk_ves3)    
    unique_elements, counts_elements = np.unique(labels, return_counts=True)
    msk_ves3 = np.isin(labels,unique_elements[np.where(np.logical_and(counts_elements>400, unique_elements>0))])

    

    
    if not adv_den:
        #remove areas below a minimum size
        labels, nr_objects = label(msk_ves3, structure = np.ones((3,3,3)))    
        unique_elements, counts_elements = np.unique(labels, return_counts=True)
        msk_ves3= np.isin(labels,unique_elements[np.where(np.logical_and(counts_elements>800, unique_elements>0))])
        
    #supress edge effects
    msk_ves4 = msk_ves3*msk_lun_ves
    msk = np.logical_or(msk, msk_ves4)
    
    #remove tumors (large areas that are unlikely to be large vessels)
    msk_ves_cor = np.logical_and(msk_ves4, np.logical_not(msk_lrg_ves))
    msk_ves_cor = binary_dilation(binary_erosion(msk_ves_cor, iterations=adp_res(4)),iterations=adp_res(10))#*msk_sob
    msk_ves5 = np.logical_and(msk_ves4, np.logical_not(msk_ves_cor))
    
    #remove areas classified as vessels that are structurally similar to a plate
    #calculate the structure tensor
    sigma = (3, 3, 3)
    A_elems = structure_tensor(msk_ves5, sigma=sigma)
    e1, e2, e3 = structure_tensor_eigenvalues(A_elems)
    
    FA = FA_map(e1,e2,e3)*msk_ves5
    FA = np.nan_to_num(FA)
    msk_plate = FA > 0.85
    msk_plate[msk_lrg_ves]=False
    
    msk_ves6 = np.logical_and(msk_ves5, np.logical_not(msk_plate))
    
    #suppress edge effects
    msk_lung_edge = np.logical_and(luvox_msk, np.logical_not(binary_erosion(luvox_msk, iterations=4)))
    msk_lung_edge = np.logical_and(msk_lung_edge, raw > -300)
    
    msk_ves6 = np.logical_and(msk_ves6, np.logical_not(msk_lung_edge))
    
    
    #remove areas below a minimum size
    labels, nr_objects = label(msk_ves6)    
    unique_elements, counts_elements = np.unique(labels, return_counts=True)
    msk_ves6 = np.isin(labels,unique_elements[np.where(np.logical_and(counts_elements>400, unique_elements>0))])
    
    #remove bronchial walls
    #msk_ves6 = np.logical_and(msk_ves6, np.logical_not(binary_dilation(msk_tra,iterations=3)))
    msk_ves6 = np.logical_and(msk_ves6, np.logical_not(binary_dilation(np.logical_and(binary_closing(msk_ves6, iterations=3), msk_tra), iterations=3)))
    
    return np.array(msk,dtype=bool), msk_ves6

    
def vessel_det(msk, msk_main_ves, raw):
    
    raw_den = np.array(raw*msk)

    raw_den_con = exposure.rescale_intensity(raw_den, in_range=(-900, -300))
    raw_den_con_thr =  filters.threshold_multiotsu(raw_den_con[msk],classes=3)

    
    msk_HU = raw>-800
    arr_low = filters.gaussian(raw_den_con, sigma=adp_res(5))
    arr_low[msk_HU] = np.min(raw_den_con)
    val_high = raw_den_con_thr[1]
    #arr_low[np.where(arr_low<filters.threshold_multiotsu(raw_den_con[msk],classes=3)[0])]=np.max(raw_den_con)
    tisX=filters.apply_hysteresis_threshold(raw_den_con, low=arr_low, high=val_high)
    tisX = tisX*binary_erosion(msk, iterations=2)

    
    #remove areas that are vessels
    les = np.logical_and(tisX,np.logical_not(msk_main_ves))
    
    
    #Classify "large-scale" contiguous areas as suspect tissue
    les = binary_dilation(binary_erosion(np.logical_and(tisX, np.logical_not(msk_main_ves)), iterations=adp_res(3)),iterations=adp_res(12))*tisX
    #remove small areas (vessels) from les
    les = binary_dilation(binary_erosion(les, iterations=adp_res(2)),iterations=adp_res(2))*tisX
    
    #remove main vessels
    les = np.logical_and(les, np.logical_not(binary_fill_holes(binary_dilation(msk_main_ves))))
    
    ves = np.logical_and(msk_main_ves, np.logical_not(les))

    
    return les, ves
    
    
    

    

def trachea_det(msk, raw):
    
    
    #msk_dil = binary_dilation(msk, iterations=adp_res(6))
    

    #trachea has the lowest pixelvalues within the mask
    #values to analyse:
    val = raw[msk]
    val = val[np.where(val<-900)]
    trachea_thr= filters.threshold_multiotsu(val,classes=3)

    #trachea_msk = (raw < trachea_thr[0])*msk
    
    #supress edge effects
    
    #Sphere starting from the center of the data set
    ori = np.array(np.array(np.shape(msk))/2, dtype=int)
    sphere_radius = int(np.min(np.shape(msk)-ori)-4)
    roi = np.zeros_like(msk)
    roi[ori[0],ori[1],ori[2]] = 1
    roi = fftconvolve(roi, morphology.ball(sphere_radius), mode='same') > 0.5
    #remove upper edge
    roi[:,:80,:] = False
    
    raw_con = mclahe.mclahe(raw, kernel_size=(53,53,53), n_bins=256, adaptive_hist_range=True)
    raw_con_thr =  filters.threshold_multiotsu(raw_con[msk],classes=5)
    
    trachea_msk = raw_con < raw_con_thr[0]

    raw_thr = filters.threshold_multiotsu(raw[trachea_msk],classes=3)
    trachea_msk[raw>raw_thr[0]] = False

    raw_tra = np.copy(raw)
    raw_tra[np.logical_not(trachea_msk)]=np.max(raw)
    trachea_thr_2 = filters.threshold_multiotsu(raw[trachea_msk],classes=5)
    

    
    #compute tubeness
    #downsampling
    raw_tra_int = ArrRes(np.array(raw_tra,dtype=np.float32),res_fac=np.ones(3)*0.5,order=1)
    tube_meij = np.array(filters.meijering(np.array(raw_tra_int, dtype=np.float32), sigmas=[1,2,3,4,5], black_ridges=True),dtype = np.float32)
    #backsampling
    #tube_meij_msk = MskRes(np.array(tube_meij_msk,dtype=np.float32), dim = np.shape(raw),order=1)
    tube_meij = ArrRes(np.array(tube_meij,dtype=np.float32), dim = np.shape(raw),order=1)

    thresholds = filters.threshold_multiotsu(tube_meij[roi], classes=5)
    tube_meij_msk =  (tube_meij > np.max(thresholds))*roi
    #tube_meij_msk[raw>trachea_thr_2[3]]=False
    
     
    trachea = tol_region_growing(raw_tra,trachea_msk, tol_p=trachea_thr_2[1], tol_m = -1200, abs_val=True)
    #trachea = binary_fill_holes(binary_closing(trachea,iterations=1))
    tra_msk = np.copy(trachea_msk)
    #tra_msk[raw>trachea_thr[0]]=False
   # tra_msk[np.logical_not(tube_meij_msk)]=False
    trachea = np.logical_or(trachea, tra_msk)
    #Exclude areas with too low tubeness
    trachea[np.logical_not(tube_meij_msk)] = False
    #label trachea mask
    labels, nr_objects = label(trachea)
    #which label has the highest tubeness?
    unique_elements, counts_elements = np.unique(labels[tube_meij_msk], return_counts=True)

    #Consider the largest labels
    pot_lab = unique_elements[np.argsort(counts_elements)[-5:]]
    men_tub = np.zeros(len(pot_lab))
    for i in range(len(pot_lab)):
        men_tub[i] = np.mean(tube_meij[labels==pot_lab[i]])
    
    lab = pot_lab[np.logical_and(men_tub == np.max(men_tub), pot_lab != 0)]
    
    trachea1=np.zeros_like(trachea)
    if lab.size > 0:
        for i in lab:
            trachea1 = np.logical_or(trachea1, labels == i) 
    
    #trachea = labels == pot_lab[np.argmax(men_tub)]
    
    #exclude areas of too low tubeness
    trachea1[np.logical_not(tube_meij_msk)] = False
        
    trachea_gro = itr_tra_grow(raw,trachea1,trachea_thr_2[1],trachea_thr[1]-50,50,cor_thr=trachea_thr[0],ero=2)#*msk_dil

    img_con = exposure.rescale_intensity(raw, in_range=(trachea_thr_2[0], -100))

    img_con_4 = mclahe.mclahe(img_con, kernel_size=(23,23,23), n_bins=256, adaptive_hist_range=True)#*msk_dil
    img_con_4_den = restoration.denoise_tv_chambolle(img_con_4,weight=0.07)
    img_con_4_thr = filters.threshold_multiotsu(img_con_4_den[msk],classes=5)
    
    
    tra_sml = morphology.black_tophat(img_con_4_den,selem=morphology.ball(adp_res(2)))#np.ones((3,3,3)))
    tra_sml_thr =  filters.threshold_multiotsu(tra_sml[msk],classes=4)
    tra_sml_msk = tra_sml > tra_sml_thr[1]
    
    #add large trachea parts
    img_con_int = ArrRes(np.array(img_con_4_den,dtype=np.float32),res_fac=np.ones(3)*0.3,order=1)
    tra_lrg = morphology.black_tophat(img_con_int,selem=morphology.ball(adp_res(4)))
    tra_lrg = ArrRes(np.array(tra_lrg,dtype=np.float32), dim = np.shape(raw),order=1)
    tra_lrg_thr =  filters.threshold_multiotsu(tra_lrg[msk],classes=4)
    tra_lrg_msk = tra_lrg > tra_lrg_thr[2]
    #remove small areas
    #tra_lrg_msk = binary_dilation(binary_erosion(tra_lrg_msk, iterations=adp_res(2)),iterations=adp_res(2))*tra_lrg_msk    
    #tra_lrg_msk = np.logical_and(tra_lrg_msk, np.logical_not(tra_sml_msk))
    tra_lrg_msk[img_con_4_den > img_con_4_thr[1]] = False
    tra_lrg_msk[np.logical_not(trachea_msk)] = False
    
    #Only allow areas with connection to trachea 
    labels, nr_objects = label(tra_lrg_msk)
    unique_elements, counts_elements = np.unique(labels[trachea_gro], return_counts=True)
    tra_lrg_msk = np.isin(labels,unique_elements[np.where(unique_elements>0)])

    cor_msk = np.copy(np.logical_or(np.logical_or(tra_sml_msk, tube_meij_msk), tra_lrg_msk))
    cor_msk[img_con_4_den > img_con_4_thr[1]] = False
    #cor_msk[np.logical_not(msk_dil)] = False
    
    trachea2 = itr_tra_grow(raw,trachea_gro,trachea_thr_2[2],trachea_thr[0]-50,50,cor_thr=trachea_thr[1],cor_msk=cor_msk,ero=1)#*msk_dil
    tra_cor = np.logical_and(trachea2, np.logical_not(trachea_gro))
    tra_cor = binary_dilation(binary_erosion(tra_cor, iterations=adp_res(3)),iterations=adp_res(5))*msk
    
    #correct mask only above a maximum HU
    tra_cor[img_con_4_den < img_con_4_thr[1]] = False
    
    trachea2 = np.logical_or(np.logical_and(trachea2, np.logical_not(tra_cor)), trachea1)

    #label trachea mask
    labels, nr_objects = label(trachea2)#, structure=np.ones((3,3,3)))
    #label with connection to the mask "trachea" is trachea
    unique_elements, counts_elements = np.unique(labels[trachea1], return_counts=True)
    
    #all labels with connection are allowed
    trachea2 = np.isin(labels,unique_elements[np.where(unique_elements>0)])
    
    
    #take into account the smallest bronchi
    tra_sml = morphology.black_tophat(img_con_4_den,selem=morphology.ball(adp_res(1)))#np.ones((3,3,3)))
    tra_sml_thr =  filters.threshold_multiotsu(tra_sml[msk],classes=4)
    tra_sml_msk = tra_sml > tra_sml_thr[1]
    
    trachea3 = itr_tra_grow(raw,trachea2,trachea_thr_2[2],trachea_thr[0]-50,50,cor_thr=trachea_thr[0],cor_msk=tra_sml_msk,ero=1)#*msk_dil
    
    tra_cor = np.logical_and(trachea3, np.logical_not(trachea2))
    tra_cor = binary_dilation(binary_erosion(tra_cor, iterations=adp_res(1)),iterations=adp_res(2))#*msk
    #correct only range above a maximum HU
    tra_cor[raw<trachea_thr_2[0]] = False
    
    trachea4 = np.logical_or(np.logical_and(trachea3, np.logical_not(tra_cor)), trachea1)
    

    #label trachea mask
    labels, nr_objects = label(trachea4)#, structure=np.ones((3,3,3)))
    #label with connection to the mask "trachea" is trachea
    unique_elements, counts_elements = np.unique(labels[trachea1], return_counts=True)
    
    #alle label mit Verbindung werden zugelassen
    trachea4 = np.isin(labels,unique_elements[np.where(unique_elements>0)])
    
    return trachea4

def ArrRes(arr, res_fac=0, dim=0, order=3):
    '''
    
    Parameters
    ----------
    arr : numpy array
        Zu interpolierendes Array
    res_fac : numpy array
        Interpolationsfaktor entlang jeder Achse
    dim : numpy array oder tuple
        Ziel-'Shape' des interpoliertend Arrays
    order: int
        Ordnung der Spline-Interpolation

    Returns
    -------
    int_arr: numpy array
            interpoliertes Array

    '''
    
    shape = np.shape(arr)
    
    if np.sum(res_fac)!=0:
        new_real_shape = shape * res_fac
        new_shape = np.array(new_real_shape, dtype=int)
        real_resize_factor = new_shape / shape
        return ndi.zoom(arr, real_resize_factor, order=order)
    elif np.sum(dim)!=0:
        real_resize_factor = np.array(dim) / shape
        return ndi.zoom(arr, real_resize_factor, order=order)
        
    
def MskRes(arr, res_fac=0, dim=0, order=0):
    msk_int = ArrRes(np.array(arr,dtype=np.float32),res_fac,dim,order)
    msk_int = msk_int > 0.1
    
    return msk_int

def MskResCrp(arr_crp, crp, dim1, res_fac=0, dim=0, order=0):
    
    msk_int = MskRes(arr_crp, res_fac, dim, order)
    
    msk = np.zeros(np.array(dim1))
    
    msk[crp[0]:crp[1],crp[2]:crp[3], crp[4]:crp[5]] = msk_int
    
    return msk

def CrpCoo(msk):
    
    x = np.where(np.sum(msk,axis=(1,2))>0)[0]
    y = np.where(np.sum(msk,axis=(0,2))>0)[0]
    z = np.where(np.sum(msk,axis=(0,1))>0)[0]
    
    return np.array([x.min(), x.max(), y.min(), y.max(), z.min(), z.max()])

def arrtmp(msk,im_int,crp_int):
    dum = np.zeros_like(im_int)
    dum[crp_int[0]:crp_int[1],crp_int[2]:crp_int[3], crp_int[4]:crp_int[5]]=msk
    
    return np.array(dum, dtype=np.uint8)


def FA_map(e1, e2, e3):
    FA = np.sqrt(1/2)* np.sqrt((e1-e2)**2 + (e2 - e3)**2 + (e3-e1)**2)/np.sqrt(e1**2 + e2**2 + e3**2)
    return FA


def travel(pth_in, pth_out, pfx='', pth_ext_msk = ''):
    '''
    Lung vessel and trachea segmentation.

    Parameters
    ----------
    pth_in : str
        path including complete filename to the input CT dataset (nifti file)
    pth_out : str
        output path for the resulting masks
    pfx : str
        Prefix used in the filenames of the resulting mask. The default is ''.
    pth_ext_msk : str
        path including complete filename to an optional external lung segmentation (nifti file). The default is ''.

    Returns
    -------
    None.

    '''
    
    #load nii-data
    img_nii = nib.load(pth_in)
    img = np.array(np.squeeze(img_nii.get_fdata()), dtype = np.float32)
    affine = img_nii.affine
    
    #rescale image pixel values to be in the range [0,1]
    img_scl = img.astype(np.float32)
    img_scl += np.abs(np.min(img_scl))
    img_scl /= np.max(img_scl)
    
    print('1. detect lung (luvox mask)')
    tme_1 = time.time()
    # preprocess voxels
    case_voxels, avg_otsu = preprocessing(img_scl)
    # compute segmentation
    luvox_msk = np.array(segment_lung(case_voxels, avg_otsu) > 0, dtype=bool)
        
    if len(pth_ext_msk)>0:
        msk_ext_lun_nii = nib.load(pth_ext_msk)
        msk_ext_lun = np.array(np.squeeze(msk_ext_lun_nii.get_fdata()), dtype = np.bool)
        msk_ext_lun = msk_ext_lun != 0 #because lung mask could have a strange format
        eq_affine = msk_ext_lun_nii.affine == affine
        if np.sum(eq_affine) == 16:
            luvox_msk = np.logical_or(luvox_msk, msk_ext_lun)
        elif ~eq_affine[2,2] and affine[2,2]/msk_ext_lun_nii.affine[2,2] == -1:
            msk_ext_lun = ndi.affine_transform(msk_ext_lun, np.array([[1,0,0,0],[0,1,0,0], [0,0,-1,np.shape(msk_ext_lun)[2]-1],[0,0,0,1]]), order=0,mode='constant')
            luvox_msk = np.logical_or(luvox_msk, msk_ext_lun)
        
    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')
    # nii_img = nib.Nifti1Image(np.array(luvox_msk, dtype=np.uint8), affine)
    # nib.save(nii_img, pth_out+'msk_luvox.nii.gz')
    
    
    
    #crop array to relevant area
    crp = CrpCoo(binary_dilation(luvox_msk, iterations = 5))
    
    img_crp = img[crp[0]:crp[1],crp[2]:crp[3], crp[4]:crp[5]]
    
    #get resolution:
    res= np.array(img_nii.header.get_zooms())
    res_min = np.min(res)
    
    #interpolate to isotropic resolution
    com_res = np.ones(3)*0.7
    resize_factor = res / com_res
    
    
    print('2. interpolate image data')
    tme_1 = time.time()
    img_int = ArrRes(img_crp,resize_factor,order=3)
    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')
    
    
    crp_int = np.array([crp[0]*resize_factor[0],crp[2]*resize_factor[1], crp[4]*resize_factor[2]],dtype=int)
    crp_int_2 = np.array([crp_int[0],crp_int[0]+img_int.shape[0],crp_int[1],crp_int[1]+img_int.shape[1], crp_int[2],crp_int[2]+img_int.shape[2]],dtype=int)
    img_int_2 = ArrRes(img,resize_factor,order=0)
    img_int_2[crp_int_2[0]:crp_int_2[1],crp_int_2[2]:crp_int_2[3], crp_int_2[4]:crp_int_2[5]]=img_int
    

    affine_int = rescale_affine(affine, img.shape,zooms=com_res,new_shape=img_int_2.shape)
    #nii_img = nib.Nifti1Image(np.array(img_int_2, dtype=np.int16), affine_int)
    #nib.save(nii_img, pth_out+pfx+'raw_int.nii.gz')
    
    #reference resolution of the algorithm
    res_ref = 0.7
    fac_mrp = res_ref/np.min(com_res)
    
    
    
    print('3. interpolate luvox mask')
    tme_1 = time.time()
    luvox_msk_int = MskRes(luvox_msk,resize_factor)[crp_int_2[0]:crp_int_2[1],crp_int_2[2]:crp_int_2[3], crp_int_2[4]:crp_int_2[5]]
    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')
    luvox_msk = MskResCrp(luvox_msk_int, crp=crp, dim1=img.shape, dim=img_crp.shape)
    nii_img = nib.Nifti1Image(np.array(luvox_msk, dtype=np.uint8), affine)
    nib.save(nii_img, pth_out+pfx+'msk_luv.nii.gz')
    
    #detect trachea
    print('4. detect trachea')
    tme_1 = time.time()
    trachea_msk_int = trachea_det(luvox_msk_int, img_int)
    #backsampling
    trachea_msk = MskResCrp(trachea_msk_int, crp=crp, dim1=img.shape, dim=img_crp.shape)
    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')
    nii_img = nib.Nifti1Image(np.array(trachea_msk, dtype=np.uint8), affine)
    nib.save(nii_img, pth_out+pfx+'msk_tra.nii.gz')
    
    
    
    print('5. adding more tissue and detect main vessels')
    tme_1 = time.time()
    lung_all_msk_int, msk_main_ves_int = all_lung_msk(img_int, luvox_msk_int, trachea_msk_int, adv_den=False)
    #backsampling
    lung_all_msk = MskResCrp(lung_all_msk_int,crp=crp, dim1=img.shape, dim=img_crp.shape)
    msk_main_ves = MskResCrp(msk_main_ves_int,crp=crp, dim1=img.shape, dim=img_crp.shape)
    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')
    nii_img = nib.Nifti1Image(np.array(lung_all_msk, dtype=np.uint8), affine)
    nib.save(nii_img, pth_out+pfx+'msk_all.nii.gz')
    
    if len(pth_ext_msk)>0:
        msk_main_ves = msk_main_ves*msk_ext_lun
        
    nii_img = nib.Nifti1Image(np.array(msk_main_ves, dtype=np.uint8), affine)
    nib.save(nii_img, pth_out+pfx+'msk_ves_main.nii.gz')
    
    
    #detect vessels and lesions
    print('6. detect vessels and lesions')
    tme_1 = time.time()
    msk_les, msk_ves = vessel_det(lung_all_msk_int, msk_main_ves_int, img_int)
    #backsampling
    msk_les = MskResCrp(msk_les,crp=crp, dim1=img.shape, dim=img_crp.shape)
    msk_ves = MskResCrp(msk_ves, crp=crp,dim1=img.shape, dim=img_crp.shape)
    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')
    nii_img = nib.Nifti1Image(np.array(msk_ves, dtype=np.uint8), affine)
    nib.save(nii_img, pth_out+pfx+'msk_ves.nii.gz')
    
    nii_img = nib.Nifti1Image(np.array(msk_les, dtype=np.uint8), affine)
    nib.save(nii_img, pth_out+pfx+'msk_les.nii.gz')



def travel_nora(pth_im, pth_msk_lun, pth_msk_tra, pth_msk_ves):
    '''
    Lung vessel and trachea segmentation for the use with Nora.

    Parameters
    ----------
    pth_im : str
        path including complete filename to the input CT dataset (nifti file)
    pth_msk_lun : str
        path including complete filename to the input lung mask (nifti file)
    pth_msk_tra : str
        path including complete filename to the output trachea mask (nifti file)
    pth_msk_ves : str
        path including complete filename to the output vessel mask (nifti file)

    Returns
    -------
    None.

    '''


    
    #load nii-data
    img_nii = nib.load(pth_im)
    img = np.array(np.squeeze(img_nii.get_fdata()), dtype = np.float32)
    img[img<-1024]=-1024
    affine = img_nii.affine
    
    #rescale image pixel values to be in the range [0,1]
    img_scl = img.astype(np.float32)
    img_scl += np.abs(np.min(img_scl))
    img_scl /= np.max(img_scl)
    
    print('1. detect lung (luvox mask)')
    tme_1 = time.time()
    # preprocess voxels
    case_voxels, avg_otsu = preprocessing(img_scl)
    # compute segmentation
    luvox_msk = np.array(segment_lung(case_voxels, avg_otsu) > 0, dtype=bool)
        
    msk_ext_lun_nii = nib.load(pth_msk_lun)
    msk_ext_lun = np.array(np.squeeze(msk_ext_lun_nii.get_fdata()), dtype = np.bool)
    msk_ext_lun = msk_ext_lun != 0 #da Lungenmaske manchmal komisches format hat
    eq_affine = msk_ext_lun_nii.affine == affine
    if np.sum(eq_affine) == 16:
        luvox_msk = np.logical_or(luvox_msk, msk_ext_lun)
    elif ~eq_affine[2,2] and affine[2,2]/msk_ext_lun_nii.affine[2,2] == -1:
        msk_ext_lun = ndi.affine_transform(msk_ext_lun, np.array([[1,0,0,0],[0,1,0,0], [0,0,-1,np.shape(msk_ext_lun)[2]-1],[0,0,0,1]]), order=0,mode='constant')
        luvox_msk = np.logical_or(luvox_msk, msk_ext_lun)
        
    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')
    # nii_img = nib.Nifti1Image(np.array(luvox_msk, dtype=np.uint8), affine)
    # nib.save(nii_img, pth_out+'msk_luvox.nii.gz')
    
    
    
    #crop array to relevant area
    crp = CrpCoo(binary_dilation(luvox_msk, iterations = 5))
    
    img_crp = img[crp[0]:crp[1],crp[2]:crp[3], crp[4]:crp[5]]
    
    #get resolution:
    res= np.array(img_nii.header.get_zooms())
    res_min = np.min(res)
    
    #interpolate to isotropic resolution
    com_res = np.ones(3)*0.7
    resize_factor = res / com_res
    
    
    print('2. interpolate image data')
    tme_1 = time.time()
    img_int = ArrRes(img_crp,resize_factor,order=3)
    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')
    
    
    crp_int = np.array([crp[0]*resize_factor[0],crp[2]*resize_factor[1], crp[4]*resize_factor[2]],dtype=int)
    crp_int_2 = np.array([crp_int[0],crp_int[0]+img_int.shape[0],crp_int[1],crp_int[1]+img_int.shape[1], crp_int[2],crp_int[2]+img_int.shape[2]],dtype=int)
    img_int_2 = ArrRes(img,resize_factor,order=0)
    img_int_2[crp_int_2[0]:crp_int_2[1],crp_int_2[2]:crp_int_2[3], crp_int_2[4]:crp_int_2[5]]=img_int
    

    affine_int = rescale_affine(affine, img.shape,zooms=com_res,new_shape=img_int_2.shape)
    #nii_img = nib.Nifti1Image(np.array(img_int_2, dtype=np.int16), affine_int)
    #nib.save(nii_img, pth_out+pfx+'raw_int.nii.gz')
    
    #reference resolution of the algorithm
    res_ref = 0.7
    fac_mrp = res_ref/np.min(com_res)
    
    
    
    print('3. interpolate luvox mask')
    tme_1 = time.time()
    luvox_msk_int = MskRes(luvox_msk,resize_factor)[crp_int_2[0]:crp_int_2[1],crp_int_2[2]:crp_int_2[3], crp_int_2[4]:crp_int_2[5]]
    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')
    luvox_msk = MskResCrp(luvox_msk_int, crp=crp, dim1=img.shape, dim=img_crp.shape)
    
    #detect trachea
    print('4. detect trachea')
    tme_1 = time.time()
    trachea_msk_int = trachea_det(luvox_msk_int, img_int)
    #backsampling
    trachea_msk = MskResCrp(trachea_msk_int, crp=crp, dim1=img.shape, dim=img_crp.shape)
    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')
    nii_img = nib.Nifti1Image(np.array(trachea_msk, dtype=np.uint8), affine)
    nib.save(nii_img, pth_msk_tra)
    
    
    
    print('5. adding more tissue and detect main vessels')
    tme_1 = time.time()
    lung_all_msk_int, msk_main_ves_int = all_lung_msk(img_int, luvox_msk_int, trachea_msk_int, adv_den=False)
    #backsampling
    lung_all_msk = MskResCrp(lung_all_msk_int,crp=crp, dim1=img.shape, dim=img_crp.shape)
    msk_main_ves = MskResCrp(msk_main_ves_int,crp=crp, dim1=img.shape, dim=img_crp.shape)
    tme_2 = time.time()
    print('execution time: ' + str(round(tme_2-tme_1))+' s')
    # nii_img = nib.Nifti1Image(np.array(lung_all_msk, dtype=np.uint8), affine)
    # nib.save(nii_img, pth_out+pfx+'msk_all.nii.gz')
    
    #renove vessels outside the lung mask
    msk_main_ves = np.logical_and(msk_ext_lun, msk_main_ves)
    nii_img = nib.Nifti1Image(np.array(msk_main_ves, dtype=np.uint8), affine)
    nib.save(nii_img, pth_msk_ves)
    
    
    # #detect vessels and lesions
    # print('6. detect vessels and lesions')
    # tme_1 = time.time()
    # msk_les, msk_ves = vessel_det(lung_all_msk_int, msk_main_ves_int, img_int)
    # #backsampling
    # msk_les = MskResCrp(msk_les,crp=crp, dim1=img.shape, dim=img_crp.shape)
    # msk_ves = MskResCrp(msk_ves, crp=crp,dim1=img.shape, dim=img_crp.shape)
    # tme_2 = time.time()
    # print('execution time: ' + str(round(tme_2-tme_1))+' s')
    # nii_img = nib.Nifti1Image(np.array(msk_ves, dtype=np.uint8), affine)
    # nib.save(nii_img, pth_out+pfx+'msk_ves.nii.gz')
    
    # nii_img = nib.Nifti1Image(np.array(msk_les, dtype=np.uint8), affine)
    # nib.save(nii_img, pth_out+pfx+'msk_les.nii.gz')


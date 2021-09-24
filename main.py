#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:33:29 2021

@author: mib
"""
import argparse
from pathlib import Path
from VeSeg.travel_v2 import travel_nora

pth_in = '/media/sf_fsc/act/200728_LuHis/dat/SegTest/Thorax_1_0_I26f_3_s005.nii'
pth_ext_msk = '/media/sf_fsc/act/200728_LuHis/dat/SegTest/thorax_pred.nii.gz'
pth_trachea = '/media/sf_fsc/act/200728_LuHis/dat/SegTest/01_trachea.nii.gz'
pth_vessel = '/media/sf_fsc/act/200728_LuHis/dat/SegTest/01_vessels.nii.gz'

acc_nr = ['28661736','28437204','28559098','27010158','28818912','26906063','27010158','26906063','26514851', '25818090', '25583696', '25782823']
pth_in = '/media/sf_fsc/act/200728_LuHis/dat/SegTestExt/' + acc_nr[0] + '/WT1.nii'
pth_out = '/media/sf_fsc/act/200728_LuHis/dat/SegTestExt/'+ acc_nr[0] +'/'
pth_ext_msk = pth_out + 'thorax_pred.nii.gz'

pth_trachea = '/media/sf_fsc/act/200728_LuHis/dat/SegTestExt/' + acc_nr[0] + '/trachea.nii.gz'
pth_vessel = '/media/sf_fsc/act/200728_LuHis/dat/SegTestExt/' + acc_nr[0] + '/vessels.nii.gz'

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="image", help="lung image path", default = pth_in)
    parser.add_argument("-m", dest="mask", help="lung mask path", default = pth_ext_msk)
    parser.add_argument("-t", dest="trachea", help="trachea mask path", default = pth_trachea)
    parser.add_argument("-v", dest="vessels", help="vessel mask path", default = pth_vessel)
    args = parser.parse_args()
    
    
    travel_nora(Path(args.image), Path(args.mask), Path(args.trachea), Path(args.vessels))

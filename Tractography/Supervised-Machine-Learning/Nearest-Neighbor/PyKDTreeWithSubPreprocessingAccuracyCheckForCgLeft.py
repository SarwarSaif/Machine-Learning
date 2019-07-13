# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 23:00:30 2018

@author: majaa
"""


import numpy as np
import nibabel
from nibabel import trackvis
from dipy.viz import window, actor
from dipy.tracking.streamline import transform_streamlines

from dipy.viz import fury
import vtk.util.colors as colors
from dipy.tracking import utils
import time
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.vox2track import streamline_mapping

import scipy
import pickle

from pykdtree.kdtree import KDTree
# Define distance function.
def euclidean(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

"""
def show_tract(segmented_tract, color):
          #Visualization of the segmented tract.
           
          ren = fvtk.ren()           
          fvtk.add(ren, fvtk.line(segmented_tract,
                                    colors.green,
                                    linewidth=2,
                                    opacity=0.3))
          
          fvtk.show(ren)
"""
def show_tract(segmented_tract, path):
          """Visualization of the segmented tract.
          """ 
          
          #fa = dix['fa']

          #affine = dix['affine']
          #bundle = dix['cg.left']
          """Show every streamline with an orientation color"""
          affine=utils.affine_for_trackvis(voxel_size=np.array([1.25,1.25,1.25]))
          bundle_native = transform_streamlines(segmented_tract, np.linalg.inv(affine))

          renderer = window.Renderer()

          stream_actor = actor.line(bundle_native)

          renderer.set_camera(position=(-176.42, 118.52, 128.20),
                    focal_point=(113.30, 128.31, 76.56),
                    view_up=(0.18, 0.00, 0.98))

          renderer.add(stream_actor)

          # Uncomment the line below to show to display the window
          window.show(renderer, size=(600, 600), reset_camera=False)
          window.record(renderer, out_path=path, size=(600, 600))
          renderer.camera_info()

          
          
def compute_dsc(estimated_tract, true_tract):

    aff=utils.affine_for_trackvis(voxel_size=np.array([1.25,1.25,1.25]))
    voxel_list_estimated_tract = streamline_mapping(estimated_tract, affine=aff).keys()
    voxel_list_true_tract = streamline_mapping(true_tract, affine=aff).keys()
    TP = len(set(voxel_list_estimated_tract).intersection(set(voxel_list_true_tract)))
    vol_A = len(set(voxel_list_estimated_tract))
    vol_B = len(set(voxel_list_true_tract))
    DSC = 2.0 * float(TP) / float(vol_A + vol_B)
    return DSC         

          

if __name__ == '__main__':
    
    target="161731"
    target_brain="full1M_"+target+".trk"
    
    subjectList ="15"
    tractList = "_cg.left.trk" 
    
    pick = "preprocessed\\"+subjectList+"_"+target+"_"+tractList+"_PreprocessedData.pickle"
    path= "images\\"+subjectList+"_sub_"+target+"_"+tractList+"_KDPreprocessedResult.png"
    trueTract, hdr = trackvis.read(target+tractList, as_generator=False)
    trueTract = np.array([s[0] for s in trueTract], dtype=np.object)
   
    t1=time.time() 
    #read the whole tractography
    wholeTract, hdr = trackvis.read(target_brain, as_generator=False)
    wholeTract = np.array([s[0] for s in wholeTract], dtype=np.object)
    
    ###set fixed point 
   
    
   
    wholeTract_downsampled = [set_number_of_points(s, 12).ravel() for s in wholeTract]

    tractTrack=[[0 for x in range(5)] for y in range(len(wholeTract))]
        
    s=[]
    tractLen=[]

    pickle_in = open(pick, "rb")
    b = pickle.load(pickle_in)
    a=np.array(wholeTract_downsampled)
    
      
    tree4 = KDTree(a, leafsize=100)
    t2=time.time()
    dist3, ind = tree4.query(b, k=1)
    t3=time.time()
    
    ind_t=np.hstack(ind) 
    
    print("total time")
    print (t3-t1)
    """
    ###### VOTING ADDED
    
    for t in range(len(ind_t)):
        tractTrack[(ind_t[t])][0]=tractTrack[(ind_t[t])][0]+1            
    
    #print (time.time()-t1)
    
    
    tr=[]
    for l in range(len(tractTrack)):
        m,k=max((v,i) for i,v in enumerate(tractTrack[l]))
        ####### VOTING THRESHOLD ##########     
        if m>0 :      
            tr.append(l)
    """
   
    
    # Calculating Dice Similarity Co-efficient 
    ds1=compute_dsc(wholeTract[ind_t],trueTract)
    print("ds1 = ")
    print(ds1)
    #extract tract 
    segmented_tract=wholeTract [ind_t] 
    try:

        #print(tractTrack)    
        pickle_out = open("data\preprocessed\\"+subjectList+"_"+target+tractList+"_KDPreprocessed.pickle", "wb")
            
        pickle.dump(segmented_tract, pickle_out)

        
        
    except EOFError as error:
        print("File Empty")  
    
    ##visualize segemented tract
    show_tract(segmented_tract, path)    
    
    
    
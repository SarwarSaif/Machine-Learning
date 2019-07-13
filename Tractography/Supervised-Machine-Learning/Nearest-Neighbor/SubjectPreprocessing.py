# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:37:16 2018

@author: majaa
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 23:00:30 2018

@author: majaa
"""

import numpy as np
from nibabel import trackvis
#from dipy.viz import fvtk
import vtk.util.colors as colors

import time
from dipy.tracking.streamline import set_number_of_points
#import vptree
import scipy

from pykdtree.kdtree import KDTree

from tkinter import *
from tkinter import filedialog
import numpy as np
from nibabel import trackvis
from dipy.tracking.utils import length
#from dipy.viz import fvtk
from dipy.segment import *
#import vtk.util.colors as colors
from dipy.tracking.distances import mam_distances, bundles_distances_mam
import pickle
#for downsampling
from dipy.tracking.metrics import downsample

# Define distance function.


          

if __name__ == '__main__':
    
    #target_brain="full1M_100307.trk"
    
    target_brain="161731"
    
    #subjectList= [ "124422","111312","100408","100307","856766"]
    #subjectList= [ "124422","111312","100408","100307","856766","201111","106016","105115","199655","126325"]
    subjectList= [ "124422","111312","100408","100307","856766","201111","106016","105115","199655","126325",
                  "127933","128632","136833","110411","192540"]
    tractList = "_cg.left.trk" 
        
    s=[]
    tractLen=[]
    
    T_filename=subjectList[0]+tractList
    print(T_filename)
    subTract,subHdr= trackvis.read(T_filename, as_generator=False)
    subTract= np.array([s[0] for s in subTract], dtype=np.object)
			
    ###set fix point
    bundle_downsampled1 = [set_number_of_points(s, 12)  for s in subTract]
    #print(bundle_downsampled1[0:20])    
    tractLen.append(len(bundle_downsampled1))
    for i in range(1,len(subjectList)) :
       
        #read tract
        
        T_filename=subjectList[i]+tractList
        print(T_filename)
        subTract,subHdr= trackvis.read(T_filename, as_generator=False)
        subTract= np.array([s[0] for s in subTract], dtype=np.object)
			
        ###set fix point
        bundle_downsampled2 = [set_number_of_points(s, 12)  for s in subTract]
        tractLen.append(len(subTract))
        
        minval=[]
        
        minval2=[]
        DM = bundles_distances_mam(bundle_downsampled1, bundle_downsampled2 )
        for l in range(0,len(bundle_downsampled1)) :
            
            m,k=min((v,i) for i,v in enumerate(DM[l]))
            minval.append(k)
            m,k=max((v,i) for i,v in enumerate(DM[l]))
            minval.append(k)
        
       
        
        minval=list(set(minval))
        
        
        
        ### Deleting nearest tracts
        bundle_downsampled2=np.array(bundle_downsampled2)  
        print(len(bundle_downsampled2))
        bundle_downsampled2=np.delete(bundle_downsampled2, minval, axis=0)
        print(len(bundle_downsampled2))
        
    
        bundle_downsampled1=bundle_downsampled1+bundle_downsampled2.tolist()
        
        print(len(bundle_downsampled1))
        
    
    #bundle_downsampled1=np.array(bundle_downsampled1, dtype=np.object) 
    #print(bundle_downsampled1)
    Y=[]
    """
    print(bundle_downsampled1[0])
    bundle_downsampled1[0]=bundle_downsampled1[0].ravel()
    print(bundle_downsampled1[0])
    """
    for j in range(0,len(bundle_downsampled1)):
        bundle_downsampled1[j]=np.array(bundle_downsampled1[j])
        bundle_downsampled1[j]=bundle_downsampled1[j].ravel()
        
        
    
    b=np.array(bundle_downsampled1, dtype=np.float32)
    print(tractLen)
    
    pick = "preprocessed\\"+str(len(subjectList))+"_"+target_brain+"_"+tractList+"_PreprocessedData.pickle"
    
    try:

        #print(tractTrack)    
        pickle_out = open(pick, "wb")
            
        pickle.dump(b, pickle_out)

        
        
    except EOFError as error:
        print("File Empty")        
        

    
   
import os
import cv2
import glob
import numpy as np
from PIL import Image, ImageOps
from skimage import color
from skimage import io

def getStareData(IMAGE_SIZE):
    
    AMD_w_others = []
    normal = []
    
    img_dir_AMDwOthers = '../data/preprocessing/CLAHE/STARE/AMD w others'
    img_dir_Normal = '../data/preprocessing/CLAHE/STARE/Normal'
    
    data_img_AMD_path = os.path.join(img_dir_AMDwOthers,'*g')
    data_img_Normal_path = os.path.join(img_dir_Normal,'*g')
    
    amd_files = glob.glob(data_img_AMD_path) #error @ rgb_img when removing glob.glob dunno y
    normal_files = glob.glob(data_img_Normal_path)
    
    for f1 in amd_files:
        if(f1.endswith(".jpg") or f1.endswith(".png") or f1.endswith(".jpeg")):
            
            orgImg = Image.open(f1).resize(IMAGE_SIZE)
            mirImg = ImageOps.mirror(Image.open(f1).resize(IMAGE_SIZE))
            
            AMD_w_others.append(np.array(orgImg)) 
            AMD_w_others.append(np.array(mirImg))
            
            
    
    for f2 in normal_files:
        if(f2.endswith(".jpg") or f2.endswith(".png") or f1.endswith(".jpeg")):
            
            orgImg = Image.open(f2).resize(IMAGE_SIZE)
            mirImg = ImageOps.mirror(Image.open(f1).resize(IMAGE_SIZE))
            
            normal.append(np.array(orgImg)) 
            normal.append(np.array(mirImg))
            
            
    
    AMD_w_others_np = np.array(AMD_w_others)
    normal_np = np.array(normal)
    
    AMD_w_others_np = np.asarray(AMD_w_others_np , dtype=np.float32)/255
    AMD_w_others_np = AMD_w_others_np.reshape(AMD_w_others_np.shape[0],
                                                    AMD_w_others_np.shape[1],
                                                    AMD_w_others_np.shape[2],
                                                    1)
    
    normal_np = np.asarray(normal_np , dtype=np.float32)/255
    normal_np = normal_np.reshape(normal_np.shape[0],
                                     normal_np.shape[1],
                                     normal_np.shape[2],
                                     1)
    
    return AMD_w_others_np, normal_np

def getRFMIDData(IMAGE_SIZE):
    
    pure_AMD = []
    AMD_w_others = []
    normal = []
    
    img_dir_pure_AMD = '../data/preprocessing/CLAHE/RFMiD/Pure AMD/'
    img_dir_AMDwOthers = '../data/preprocessing/CLAHE/RFMiD/AMD w others/'
    img_dir_Normal = '../data/preprocessing/CLAHE/RFMiD/Normal/'
    
    data_img_AMD_path = os.path.join(img_dir_pure_AMD,'*g')
    data_img_AMD_w_others_path = os.path.join(img_dir_AMDwOthers,'*g')
    data_img_Normal_path = os.path.join(img_dir_Normal,'*g')
    
    amd_files = glob.glob(data_img_AMD_path) #error @ rgb_img when removing glob.glob dunno y
    amd_w_others_files = glob.glob(data_img_AMD_w_others_path)
    normal_files = glob.glob(data_img_Normal_path)
    
    for f1 in amd_files:
        if(f1.endswith(".jpg") or f1.endswith(".png") or f1.endswith(".jpeg")):
            
            orgImg = Image.open(f1).resize(IMAGE_SIZE)
            mirImg = ImageOps.mirror(Image.open(f1).resize(IMAGE_SIZE))
            
            pure_AMD.append(np.array(orgImg)) 
            pure_AMD.append(np.array(mirImg))
            
            
    
    for f2 in normal_files:
        if(f2.endswith(".jpg") or f2.endswith(".png") or f2.endswith(".jpeg")):
            
            orgImg = Image.open(f2).resize(IMAGE_SIZE)
            mirImg = ImageOps.mirror(Image.open(f1).resize(IMAGE_SIZE))
            
            normal.append(np.array(orgImg)) 
            normal.append(np.array(mirImg))
            
            
    
    for f3 in amd_w_others_files:
        if(f3.endswith(".jpg") or f3.endswith(".png") or f3.endswith(".jpeg")):
            
            orgImg = Image.open(f3).resize(IMAGE_SIZE)
            mirImg = ImageOps.mirror(Image.open(f1).resize(IMAGE_SIZE))
            
            AMD_w_others.append(np.array(orgImg)) 
            AMD_w_others.append(np.array(mirImg))
            
            
            
    pure_amd_np = np.array(pure_AMD)
    AMD_w_others_np = np.array(AMD_w_others)
    normal_np = np.array(normal)
    
    pure_amd_np = np.asarray(pure_amd_np , dtype=np.float32)/255
    pure_amd_np = pure_amd_np.reshape(pure_amd_np.shape[0],
                                      pure_amd_np.shape[1],
                                      pure_amd_np.shape[2],
                                      1)
    
    AMD_w_others_np = np.asarray(AMD_w_others_np , dtype=np.float32)/255
    AMD_w_others_np = AMD_w_others_np.reshape(AMD_w_others_np.shape[0],
                                                    AMD_w_others_np.shape[1],
                                                    AMD_w_others_np.shape[2],
                                                    1)
    
    normal_np = np.asarray(normal_np , dtype=np.float32)/255
    normal_np = normal_np.reshape(normal_np.shape[0],
                                     normal_np.shape[1],
                                     normal_np.shape[2],
                                     1)
    
    return pure_amd_np, AMD_w_others_np, normal_np

def getODIRData(IMAGE_SIZE):
    
    pure_AMD = []
    AMD_w_others = []
    normal = []
    
    img_dir_pure_AMD = '../data/preprocessing/CLAHE/ODIR/Pure AMD/'
    img_dir_AMDwOthers = '../data/preprocessing/CLAHE/ODIR/AMD w others/'
    img_dir_Normal = '../data/preprocessing/CLAHE/ODIR/Normal/'
    
    data_img_AMD_path = os.path.join(img_dir_pure_AMD,'*g')
    data_img_AMD_w_others_path = os.path.join(img_dir_AMDwOthers,'*g')
    data_img_Normal_path = os.path.join(img_dir_Normal,'*g')
    
    amd_files = glob.glob(data_img_AMD_path) #error @ rgb_img when removing glob.glob dunno y
    amd_w_others_files = glob.glob(data_img_AMD_w_others_path)
    normal_files = glob.glob(data_img_Normal_path)
    
    for f1 in amd_files:
        if(f1.endswith(".jpg") or f1.endswith(".png") or f1.endswith(".jpeg")):
            
            orgImg = Image.open(f1).resize(IMAGE_SIZE)
            mirImg = ImageOps.mirror(Image.open(f1).resize(IMAGE_SIZE))
            
            pure_AMD.append(np.array(orgImg)) 
            pure_AMD.append(np.array(mirImg))
            
            
            
    
    for f2 in normal_files:
        if(f2.endswith(".jpg") or f2.endswith(".png") or f2.endswith(".jpeg")):
            
            orgImg = Image.open(f2).resize(IMAGE_SIZE)
            mirImg = ImageOps.mirror(Image.open(f1).resize(IMAGE_SIZE))
            
            normal.append(np.array(orgImg)) 
            normal.append(np.array(mirImg))
            
            
    
    for f3 in amd_w_others_files:
        if(f3.endswith(".jpg") or f3.endswith(".png") or f3.endswith(".jpeg")):
            
            orgImg = Image.open(f3).resize(IMAGE_SIZE)
            mirImg = ImageOps.mirror(Image.open(f1).resize(IMAGE_SIZE))
            
            AMD_w_others.append(np.array(orgImg)) 
            AMD_w_others.append(np.array(mirImg))
            
            
            
    pure_amd_np = np.array(pure_AMD)
    AMD_w_others_np = np.array(AMD_w_others)
    normal_np = np.array(normal)
    
    pure_amd_np = np.asarray(pure_amd_np , dtype=np.float32)/255
    pure_amd_np = pure_amd_np.reshape(pure_amd_np.shape[0],
                                      pure_amd_np.shape[1],
                                      pure_amd_np.shape[2],
                                      1)
    
    AMD_w_others_np = np.asarray(AMD_w_others_np , dtype=np.float32)/255
    AMD_w_others_np = AMD_w_others_np.reshape(AMD_w_others_np.shape[0],
                                                    AMD_w_others_np.shape[1],
                                                    AMD_w_others_np.shape[2],
                                                    1)
    
    normal_np = np.asarray(normal_np , dtype=np.float32)/255
    normal_np = normal_np.reshape(normal_np.shape[0],
                                     normal_np.shape[1],
                                     normal_np.shape[2],
                                     1)
    
    return pure_amd_np, AMD_w_others_np, normal_np
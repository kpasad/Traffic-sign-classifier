from sklearn.utils import shuffle
import cv2
import pickle
import numpy as np

import matplotlib.pyplot as plt
import functools
import my_cv_lib as utl
import random

import os



def img_shift(img,x_pixels,y_pixels):    
   M = np.float32([[1,0,x_pixels],[0,1,y_pixels]])
   return(cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))) 

def img_rotate(img,deg):   
   M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),deg,1)
   return(cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))) 

def img_shear_x(img,x_shear):   
   M = np.array([[1,x_shear,1],[0,1,0]],float   )
   return(cv2.warpAffine(img,M,(img.shape[1],img.shape[0])))    

def img_shear_y(img,y_shear):   
   M = np.array([[1,0,1],[y_shear,1,0]],float   )
   return(cv2.warpAffine(img,M,(img.shape[1],img.shape[0])))  
                
def generate_replica(img,params):
     
    x_shift = random.randint(params['x_shift_min'],params['x_shift_max'])
    y_shift = random.randint(params['y_shift_min'],params['y_shift_max'])
    degrees = random.randint(params['degrees_min'],params['degrees_max'])
    shear_x = random.uniform(params['shear_min'],params['shear_max'])
    replica=np.expand_dims(img,0)
    img_s =img_shift(img,x_shift,y_shift)
    img_r = img_rotate(img_s,degrees)
    img_shear = img_shear_x(img_r,shear_x)
    img_shear = img_shear_y(img_r,shear_x)
    replica=np.vstack((replica,np.expand_dims(img_shear,0)))   
    return(replica)
    

def generate_aug_data(params):
    #training_file = r'C:\Users\kpasad\Box Sync\ML\Projects\sdc\CarND-Traffic-Sign-Classifier-Project\traffic-signs-data\train.p'    
    #training_file = training_file.replace('\\','/')

    
    training_file = '/home/kalpendu/ML/CarND-Traffic-Sign-Classifier-Project_v2/traffic-signs-data/train.p'
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    X_train, y_train = train['features'], train['labels']
    
    
    
    
    sign_id, sign_id_counts = np.unique(y_train, return_counts=True)
    num_imgs =3000
    num_replica = num_imgs- sign_id_counts
    num_augments = 2 #self,(x_shift, rotate, x_shear)
    
    y =[]
    replica=np.zeros_like(np.expand_dims(X_train[0],0))
    
    #for sign in sign_id:
    total_replica_cnt=0    
    for sign in range(0,43):   
        print('processing sign ',sign)    
        #num_replica = num_imgs- sign_id_counts[sign]    
        num_replica=int(num_imgs/num_augments) #How many replicas
        base_img_indices = np.where(train['labels']==sign)[0] # This last zero is the index. Np where will also return values in idx 1
        img_cnt=0
        base_img_cnt=0
        while(img_cnt < num_replica):
            img = X_train[base_img_indices[np.mod(img_cnt,sign_id_counts[sign])]]                                       
            replica=np.vstack((replica,(generate_replica(img,params))))
            y.extend([sign]*4)
            img_cnt+=1
        print('... num original images',len(base_img_indices),'num images after aug:',len(replica) )     
    X_train = np.delete(replica,0,0)        
    y_train=y
    pickle.dump([X_train,y_train],open('aug_data_p2.p','wb'))        



params={}
params['x_shift_max']=5
params['x_shift_min']=0
params['y_shift_max']=5
params['y_shift_min']=0
params['degrees_max']=20
params['degrees_min']=-20
params['shear_min']=0
params['shear_max']=0.3

generate_aug_data(params)     

#X_train, y_train = utl.generate_replica(X_train,y_train,params)
'''
rn=[]
for iter in range(200): 
    rn.append(random.randint(-15,15))
'''














from torch.utils.data import Dataset
import torch
import pandas as pd
import os
from os import path
from copy import copy
import nibabel as nib
import numpy as np
from nilearn import plotting
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter

import torchvision

sigma = 0
minimum_size = np.array([145, 230, 200])
maximum_size = np.array([235, 280, 280])

def sum1forline(filename):
    with open(filename) as f:
        return sum(1 for line in f)
def transform_bids_image(reading_img, transform='crop'):
    """
    Transformation of BIDS image: transposition of coordinates, flipping coordinages, rescaling voxel size,
    rescaling global size

    """

    header = reading_img.header
    img = reading_img.get_data()

    if len(np.shape(img)) == 4:
        img = img[:, :, :, 0]

    # Transposition
    loc_x = np.argmax(np.abs(header['srow_x'][:-1:]))
    loc_y = np.argmax(np.abs(header['srow_y'][:-1:]))
    loc_z = np.argmax(np.abs(header['srow_z'][:-1:]))
    transposed_image = img.transpose(loc_x, loc_y, loc_z)

    # Directions
    flips = [False, False, False]
    flips[0] = (np.sign(header['srow_x'][loc_x]) == -1)
    flips[1] = (np.sign(header['srow_y'][loc_y]) == -1)
    flips[2] = (np.sign(header['srow_z'][loc_z]) == -1)
    for coord, flip in enumerate(flips):
        if flip:
            transposed_image = np.flip(transposed_image, coord)

    # Resizing voxels
    coeff_x = np.max(np.abs(header['srow_x'][:-1:]))
    coeff_y = np.max(np.abs(header['srow_y'][:-1:]))
    coeff_z = np.max(np.abs(header['srow_z'][:-1:]))
    transposed_size = np.shape(transposed_image)
    transposed_image = transposed_image / np.max(transposed_image)
    new_size = np.rint(np.array(transposed_size) * np.array([coeff_x, coeff_y, coeff_z]))
    resized_image = resize(transposed_image, new_size, mode='constant')

    # Adaptation before rescale
    if transform == 'crop':
        image = crop(resized_image)
    elif transform == 'pad':
        image = pad(resized_image)
    else:
        raise ValueError("The transformations allowed are cropping (transform='crop') or padding (transform='pad')")

    # Final rescale
    rescale_image = resize(image, (121, 145, 121), mode='constant')
    #(rescale_image)
    return rescale_image
def crop(image):
    size = np.array(np.shape(image))
    crop_idx = np.rint((size - minimum_size) / 2).astype(int)
    first_crop = copy(crop_idx)
    second_crop = copy(crop_idx)
    for i in range(3):
        if minimum_size[i] + first_crop[i] * 2 != size[i]:
            first_crop[i] -= 1

    cropped_image = image[first_crop[0]:size[0]-second_crop[0],
                          first_crop[1]:size[1]-second_crop[1],
                          first_crop[2]:size[2]-second_crop[2]]

    return cropped_image


def pad(image):
    size = np.array(np.shape(image))
    pad_idx = np.rint((maximum_size - size) / 2).astype(int)
    first_pad = copy(pad_idx)
    second_pad = copy(pad_idx)
    for i in range(3):
        if size[i] + first_pad[i] * 2 != maximum_size[i]:
            first_pad[i] -= 1

    padded_image = np.pad(image, np.array([first_pad, second_pad]).T, mode='constant')

    return padded_image


class GaussianSmoothing(object):

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        image = sample['image']
        np.nan_to_num(image, copy=False)
        smoothed_image = gaussian_filter(image, sigma=self.sigma)
        sample['image'] = smoothed_image

        return sample 
    
composed = torchvision.transforms.Compose([GaussianSmoothing(sigma),])
  
    
caps_dir = '/scratch/di2078/shared/MLH/data/AGAIN/'
subjects_df = pd.read_csv('/scratch/di2078/shared/MLH/data/AGAIN/participants.tsv', sep='\t')
row_count = sum1forline('/scratch/di2078/shared/MLH/data/AGAIN/participants.tsv')
print(row_count)
index1=[]
index2=[]
index3=[]
for i in range(200):
    index1.append(i)
    index2.append(i+200)
for i in range(400,1000):
    index3.append(i)
import csv

with open('/scratch/di2078/shared/MLH/data/AGAIN/participants.tsv', 'r') as file:
    rows = [[row[0].strip()] for row in csv.reader(file)]
    
    
# open the file in the write mode
with open('/scratch/di2078/shared/MLH/data/test.csv', 'w') as f:
    writer = csv.writer(f)
    l =[]  
    l.append(str(rows[0][0]+ '\tses'))
    writer.writerow(l)
    for i in index1:
        subj_name = subjects_df.loc[i, 'participant_id']
        sessions_df = pd.read_csv(path.join(caps_dir,subj_name,subj_name+'_sessions.tsv'), sep='\t')
        for x in os.listdir(path.join(caps_dir,subj_name)):
            if x.startswith('ses'):
                #print(rows[i+1])
                img = '\t'+path.join(caps_dir,subj_name,x,'anat',subj_name+'_'+x+'_T1w.nii.gz')
                            
                #reading_image = nib.load(image_path)
                image_path = path.join(caps_dir,subj_name,x,'anat',subj_name+'_'+x+'_T1w.nii.gz')
                session = image_path[-18:-13]
                session_time = int(image_path[-13:-11]) + 12
                session = session + str(session_time)
                if (sessions_df['session_id'] == session).any() :
                    diagnosis_after = sessions_df[(sessions_df['session_id'] == session)].diagnosis.item()
                else:
                    diagnosis_after = 'no_diagnosis'
                if diagnosis_after == 'AD' or diagnosis_after == 'CN' or diagnosis_after == 'LMCI' or diagnosis_after == 'MCI':
                    l =[]  
                    l.append(str(rows[i+1][0]+ img))                
                    writer.writerow(l)
with open('/scratch/di2078/shared/MLH/data/valid.csv', 'w') as f:
    writer = csv.writer(f)
    l =[]  
    l.append(str(rows[0][0]+ '\tses'))
    writer.writerow(l)
    for i in index2:
        
        subj_name = subjects_df.loc[i, 'participant_id']
        sessions_df = pd.read_csv(path.join(caps_dir,subj_name,subj_name+'_sessions.tsv'), sep='\t')
        for x in os.listdir(path.join(caps_dir,subj_name)):
            if x.startswith('ses'):
                #sessions_df = pd.read_csv(path.join(caps_dir,subj_name,subj_name+'_sessions.tsv'), sep='\t')            
                #reading_image = nib.load(image_path)
                image_path = path.join(caps_dir,subj_name,x,'anat',subj_name+'_'+x+'_T1w.nii.gz')
                session = image_path[-18:-13]
                session_time = int(image_path[-13:-11]) + 12
                session = session + str(session_time)
                if (sessions_df['session_id'] == session).any() :
                    diagnosis_after = sessions_df[(sessions_df['session_id'] == session)].diagnosis.item()
                else:
                    diagnosis_after = 'no_diagnosis'
                if diagnosis_after == 'AD' or diagnosis_after == 'CN' or diagnosis_after == 'LMCI' or diagnosis_after == 'MCI':
                #print(rows[i+1])
                    img = '\t'+path.join(caps_dir,subj_name,x,'anat',subj_name+'_'+x+'_T1w.nii.gz')
                    l =[]  
                    l.append(str(rows[i+1][0]+ img))                
                    writer.writerow(l)
with open('/scratch/di2078/shared/MLH/data/train.csv', 'w') as f:
    writer = csv.writer(f)
    l =[]  
    l.append(str(rows[0][0]+ '\tses'))
    writer.writerow(l)
    for i in index3:
        subj_name = subjects_df.loc[i, 'participant_id']
        sessions_df = pd.read_csv(path.join(caps_dir,subj_name,subj_name+'_sessions.tsv'), sep='\t')
        for x in os.listdir(path.join(caps_dir,subj_name)):
            if x.startswith('ses'):
                #sessions_df = pd.read_csv(path.join(caps_dir,subj_name,subj_name+'_sessions.tsv'), sep='\t')            
                #reading_image = nib.load(image_path)
                image_path = path.join(caps_dir,subj_name,x,'anat',subj_name+'_'+x+'_T1w.nii.gz')
                session = image_path[-18:-13]
                session_time = int(image_path[-13:-11]) + 12
                session = session + str(session_time)
                if (sessions_df['session_id'] == session).any() :
                    diagnosis_after = sessions_df[(sessions_df['session_id'] == session)].diagnosis.item()
                else:
                    diagnosis_after = 'no_diagnosis'
                if diagnosis_after == 'AD' or diagnosis_after == 'CN' or diagnosis_after == 'LMCI' or diagnosis_after == 'MCI':
                #print(rows[i+1])
                    img = '\t'+path.join(caps_dir,subj_name,x,'anat',subj_name+'_'+x+'_T1w.nii.gz')
                    l =[]  
                    l.append(str(rows[i+1][0]+ img))                
                    writer.writerow(l)
print("done")

            
                
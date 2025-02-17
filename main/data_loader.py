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



minimum_size = np.array([145, 230, 200])
maximum_size = np.array([235, 280, 280])


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
    #print(rescale_image)
    return rescale_image


class BidsMriBrainDataset(Dataset):
    """Dataset of subjects of CLINICA (baseline only) from BIDS"""

    def __init__(self, subjects_df_path, caps_dir, transform=None, classes=2, rescale='crop'):
        """

        :param subjects_df_path: Path to a TSV file with the list of the subjects in the dataset
        :param caps_dir: The BIDS directory where the images are stored
        :param transform: Optional transform to be applied to a sample
        :param classes: Number of classes to consider for classification
            if 2 --> ['CN', 'AD']
            if 3 --> ['CN', 'MCI', 'AD']
        """
        #if type(subjects_df_path) is str:
         #   self.subjects_df = pd.read_csv(subjects_df_path, sep='\t')
        #elif type(subjects_df_path) is pd.DataFrame:
         #   self.subjects_df = subjects_df_path
        #else:
         #   raise ValueError('Please enter a path or a Dataframe as first argument')
        self.subjects_df = subjects_df_path
        self.caps_dir = caps_dir
        self.transform = transform

        if classes == 2:
            self.diagnosis_code = {'CN': 0, 'AD': 1}
        elif classes == 4:
            self.diagnosis_code = {'CN': 0, 'MCI': 1, 'LMCI': 1, 'AD': 3}

        #self.extension = '_ses-M00_T1w.nii.gz'
        #self.folder_path = path.join('ses-M00', 'anat')
        self.rescale = rescale

    def __len__(self):
        return len(self.subjects_df)

    def __getitem__(self, subj_idx):
        subj_name = self.subjects_df.loc[subj_idx, 'participant_id']
        diagnosis = self.subjects_df.loc[subj_idx, 'diagnosis_sc']
        paths = []
        for x in os.listdir(path.join(self.caps_dir,subj_name)):
            if x.startswith('ses'):
                paths.append(path.join(self.caps_dir,subj_name,x,'anat',subj_name+'_'+x+'_T1w.nii.gz'))
        #cohort = self.subjects_df.loc[subj_idx, 'cohort']
        #img_name = subj_name + self.extension

        #data_path = path.join(self.caps_dir, bids_cohort_dict[cohort])
        #data_path = path.join(self.caps_dir)
        #img_path = path.join(data_path, subj_name, self.folder_path, img_name)
        samples=[]
        #print(paths)
        sessions_df = pd.read_csv(path.join(self.caps_dir,subj_name,subj_name+'_sessions.tsv'), sep='\t')
        #sessions_df = pd.read_csv(path.join('/scratch/yx2105/shared/MLH/data/clinical_bids',subj_name,subj_name+'_sessions.tsv'), sep='\t')
        for x in paths:
            
            reading_image = nib.load(x)
            session = x[-18:-13]
            session_time = int(x[-13:-11]) + 12
            session = session + str(session_time)
            if (sessions_df['session_id'] == session).any() :
                index = sessions_df[(sessions_df['session_id'] == session)].diagnosis.item()
            else:
                index = 'no_diagnosis'
            #print(session,index)
            #dia_12_month = sessions_df.loc[index, 'diagnosis']
            #print(reading_image)
            image = transform_bids_image(reading_image, self.rescale)
            
        # Convert diagnosis to int
        #if type(diagnosis) is str:
        #    diagnosis = self.diagnosis_code[diagnosis]

            sample = {'image': image, 'orig_diagnosis': diagnosis,'diagnosis_after_12_months':index, 'name': subj_name}
            #print(sample)
            if self.transform:
                sample = self.transform(sample)
            samples.append(sample)

        return samples
    def subjects_list(self):

        return self.subjects_df['participant_id'].values.tolist()

    def diagnosis_list(self):
        diagnosis_list = self.subjects_df['diagnosis_sc'].values.tolist()
        diagnosis_code = [self.diagnosis_code[diagnosis] for diagnosis in diagnosis_list]
        return diagnosis_code


class GaussianSmoothing(object):

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        image = sample['image']
        np.nan_to_num(image, copy=False)
        smoothed_image = gaussian_filter(image, sigma=self.sigma)
        sample['image'] = smoothed_image

        return sample


class ToTensor(object):
    """Convert image type to Tensor and diagnosis to diagnosis code"""

    def __init__(self, gpu=False):
        self.gpu = gpu

    def __call__(self, sample):
        image, diagnosis, name = sample['image'], sample['diagnosis_after_12_months'], sample['name']
        np.nan_to_num(image, copy=False)

        if self.gpu:
            return {'image': torch.from_numpy(image[np.newaxis, :]).float(),
                    'diagnosis': torch.from_numpy(np.array(diagnosis)),
                    'name': name}
        else:
            return {'image': torch.from_numpy(image[np.newaxis, :]).float(),
                    'diagnosis': diagnosis,
                    'name': name}








if __name__ == '__main__':
    import torchvision

    #subjects_tsv_path = '/Volumes/aramis-projects/elina.thibeausutre/data/2-classes/dataset-ADNI+AIBL+corrOASIS.tsv'
    #caps_path = '/Volumes/aramis-projects/CLINICA/CLINICA_datasets/BIDS'
    #subjects_tsv_path='/vast/di2078/AGAIN/participants.tsv'    
    #caps_path='/vast/di2078/AGAIN'
    #subjects_tsv_path='/scratch/yx2105/shared/MLH/data/bids/participants.tsv'
    #caps_path='/scratch/yx2105/shared/MLH/data/bids'
    sigma = 0
    composed = torchvision.transforms.Compose([GaussianSmoothing(sigma),])
           
        # ToTensor()
    
    
    #dataset = BidsMriBrainDataset(subjects_tsv_path, caps_path, transform=composed)
    dataset = BidsMriBrainDataset(subjects_1, caps_path, transform=composed)
    for x in range(len(dataset)):
        print(dataset[x])
           


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
def collate_func_img(batch):
    """
    Collate functions used to collapse a mini-batch for single modal dataset.
    :param batch:
    :return:
    """
    img_list = []
    label_list = []
    name_list = []
    tab_data_list = []
   
    for dict_item in batch:
        img_list.append(dict_item['image'].float().unsqueeze(0))
        label_list.append(dict_item['diagnosis_after_12_months'])
        name_list.append(dict_item['name'])
        tab_data_list.append(dict_item['tab_data'].unsqueeze(0))

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
    #Dataset of subjects of CLINICA (baseline only) from BIDS

    def __init__(self, subjects_df_path, transform=None, classes=3, rescale='crop'):
        
        """
        :param subjects_df_path: Path to a TSV file with the list of the subjects in the dataset
        :param caps_dir: The BIDS directory where the images are stored
        :param transform: Optional transform to be applied to a sample
        :param classes: Number of classes to consider for classification
            if 2 --> ['CN', 'AD']
            if 3 --> ['CN', 'MCI', 'AD']
        """
        if type(subjects_df_path) is str:
            self.subjects_df = pd.read_csv(subjects_df_path, index_col=0, header=0)
        elif type(subjects_df_path) is pd.DataFrame:
            self.subjects_df = subjects_df_path
        else:
            raise ValueError('Please enter a path or a Dataframe as first argument')

        remove_list = ['participant_id','session_id','alternative_id_1','diagnosis_sc','diagnosis_12month', 'data_dir','img_dir']
        self.tab_columns = [i for i in self.subjects_df.columns if i not in remove_list]

        self.transform = transform

        if classes == 2:
            self.diagnosis_code = {'CN': 0, 'AD': 1}
        elif classes == 3:
            self.diagnosis_code = {'CN': 0, 'AD': 1, 'MCI': 2, 'LMCI': 2,'EMCI':2}

        self.rescale = rescale

    def __len__(self):
        return len(self.subjects_df)

    def __getitem__(self, subj_idx):
        subj_name = self.subjects_df.loc[subj_idx, 'participant_id']
        image_path = self.subjects_df.loc[subj_idx, 'img_dir']
        reading_image = nib.load(image_path)
        image = transform_bids_image(reading_image, self.rescale)

        diagnosis_after = self.subjects_df.loc[subj_idx, 'diagnosis_12month']
        if type(diagnosis_after) is str:
            diagnosis = self.diagnosis_code[diagnosis_after]
        #image=image.unsqueeze(0)

        tab_data = np.array(self.subjects_df.loc[subj_idx, self.tab_columns]).astype(np.float32)
        sample = {'image': image, 'diagnosis_after_12_months':diagnosis, 'name': subj_name, 'tab_data': tab_data}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def subjects_list(self):
        return self.subjects_df['participant_id'].values.tolist()

    def diagnosis_list(self):
        diagnosis_list = self.subjects_df['diagnosis_sc'].values.tolist()
        diagnosis_code = [self.diagnosis_code[diagnosis] for diagnosis in diagnosis_list]
        return diagnosis_code


def get_dict(path,transform=None,rescale='crop'):
    subjects_df = pd.read_csv(path, index_col=0, header=0)
    subjects_list = subjects_df['participant_id'].values.tolist() 
    samples=[]
    print("starting dict")
    for s in range(len(subjects_list)):
    #for s in range(3):
        #if s == 2:print("finished loading 1 image in dict")
        if s % 30 == 0: print("finished loading ",s," images in dict") 
        subj_name = subjects_df.loc[s, 'participant_id']
        image_path = subjects_df.loc[s, 'img_dir']
        reading_image = nib.load(image_path)
        image = transform_bids_image(reading_image, rescale)

        diagnosis_after = subjects_df.loc[s, 'diagnosis_sc']
        diagnosis_code = {'CN': 0, 'AD': 1, 'MCI': 2, 'LMCI': 2,'EMCI':2}
        remove_list = ['participant_id','session_id','alternative_id_1','diagnosis_sc','diagnosis_12month', 'data_dir','img_dir']
        if type(diagnosis_after) is str:
            diagnosis = diagnosis_code[diagnosis_after]
        tab_columns = [i for i in subjects_df.columns if i not in remove_list]
        tab_data = np.array(subjects_df.loc[s, tab_columns]).astype(np.float32)
        
        sample = {'image': image, 'diagnosis_after_12_months':diagnosis, 'name': subj_name, 'tab_data': tab_data}
        sample = transform(sample)
        #print(sample['image'],"printing in get_dict")
        samples.append(sample)
    return samples
        
class MriDataset(Dataset):
    def __init__(self,samples, transform=None):
        self.samples = samples
        self.transform = transform
    #def subjects_list(self):
     #   return self.samples['subj_name']
    def __getitem__(self, subj_idx):
        img = self.samples[subj_idx]['image']
        if self.transform is not None:
            img = self.transform(img)
        img = self.standardizer(img)
        return {'image': img, 'diagnosis_after_12_months':self.samples[subj_idx]['diagnosis_after_12_months'], 'name': self.samples[subj_idx]['name'], 'tab_data': self.samples[subj_idx]['tab_data']}
    
    def __len__(self):
        return len(self.samples)
    
    def standardizer(self, img):
        img = (img - img.mean()) / np.maximum(img.std(), 10 ** (-5))
        return img
    

class Standardizer(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        img = sample["img"]
        img = (img - img.mean()) / np.maximum(img.std(), 10 ** (-5))
        sample["img"] = img
        return sample

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
            image= torch.from_numpy(image[np.newaxis, :]).float()
            #image = image.unsqueeze(0)
            return {'image': image,
                    'diagnosis_after_12_months': torch.from_numpy(np.array(diagnosis)),
                    'name': name,
                    'tab_data': torch.from_numpy(sample['tab_data']).float()}
        else:
            image= torch.from_numpy(image[np.newaxis, :]).float()
            #image = image.unsqueeze(0)
            return {'image': image,
                    'diagnosis_after_12_months': diagnosis,
                    'name': name,
                    'tab_data': torch.from_numpy(sample['tab_data']).float()}


class MeanNormalization(object):
    """Normalize images using a .nii file with the mean values of all the subjets"""

    def __init__(self, mean_path):
        assert path.isfile(mean_path)
        self.mean_path = mean_path

    def __call__(self, sample):
        reading_mean = nib.load(self.mean_path)
        mean_img = reading_mean.get_data()
        return {'image': sample['image'] - mean_img,
                'diagnosis': sample['diagnosis'],
                'name': sample['name']}


class LeftHippocampusSegmentation(object):

    def __init__(self):
        self.x_min = 68
        self.x_max = 88
        self.y_min = 60
        self.y_max = 80
        self.z_min = 28
        self.z_max = 48

    def __call__(self, sample):
        image, diagnosis = sample['image'], sample['diagnosis']
        hippocampus = image[self.x_min:self.x_max:, self.y_min:self.y_max:, self.z_min:self.z_max:]
        return {'image': hippocampus,
                'diagnosis': sample['diagnosis'],
                'name': sample['name']}


if __name__ == '__main__':
    import torchvision
    sigma = 0
    train_path='/scratch/yx2105/shared/MLH/data/train.csv'
    test_path='/scratch/yx2105/shared/MLH/data/test.csv'
    valid_path='/scratch/yx2105/shared/MLH/data/val.csv'
    sigma = 0
    composed = torchvision.transforms.Compose([GaussianSmoothing(0), ToTensor(True)])
    
    s = get_dict(train_path,transform=composed)
    trainset = Dataset(s)
    
    
    s1 = get_dict(test_path,transform=composed)
    testset = Dataset(s1)
    
    
    s2 = get_dict(valid_path,transform=composed)
    validset = Dataset(s2)

    
    import pickle
    with open("valid_pickle", "wb") as f:
         pickle.dump(s2, f)
    with open("test_pickle", "wb") as f:
         pickle.dump(s1, f)
    with open("train_pickle", "wb") as f:
         pickle.dump(s, f)
    #with open("try", "rb") as f:
     #    a = pickle.load(f)
    

    #if ( torch.equal(a[0]['image'] ,validset[0]['image'])): print("valid istrue")
    #else:print("FALSE")

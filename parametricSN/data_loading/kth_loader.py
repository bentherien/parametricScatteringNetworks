"""Contains classes and functions for downloading and sampling the KTH-TIPS2 dataset

Author: Benjamin Therien, Shanel Gauthier

Dataset folder organization
├── a    <- Folder contains 11 folders (one per material)   
├── b    <- Folder contains 11 folders (one per material) 
├── c    <- Folder contains 11 folders (one per material) 
├── d    <- Folder contains 11 folders (one per material)    

Functions: 
    kth_augmentationFactory -- factory of KTH-TIPS2 augmentations 
    kth_getDataloaders      -- returns dataloaders for KTH-TIPS2
    download_from_url       -- Download Dataset from URL
    extract_tar             -- Extract files from tar
    create_dataset          -- Create dataset in the target folder
    downloadKTH_TIPS2       -- Download KTH-TIPS2 dataset to the data folder

class:
    KTHLoader -- loads KTH-TIPS2 from disk and creates dataloaders
"""

import shutil
import os
import requests
import tarfile
import sys
import torch
import time
import shutil

from pathlib import Path
from tqdm import tqdm
from torchvision import datasets, transforms
from glob import glob

from parametricSN.data_loading.auto_augment import AutoAugment, Cutout


def kth_augmentationFactory(augmentation, height, width):
    """Factory for different augmentation choices for KTH-TIPS2"""

    if augmentation == 'autoaugment':
        transform = [
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            Cutout()
        ]

    elif augmentation == 'original-cifar':
        transform = [
            transforms.Resize((200,200)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
        ]

    elif augmentation == 'noaugment':
        transform = [
            transforms.Resize((200,200)),
            transforms.CenterCrop((height, width))
        ]

    elif augmentation == 'glico':
        NotImplemented(f"augment parameter {augmentation} not implemented")

    else: 
        NotImplemented(f"augment parameter {augmentation} not implemented")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose(transform + [transforms.ToTensor(), normalize])



def kth_getDataloaders(trainBatchSize, valBatchSize, trainAugmentation,
                       height, width, sample, dataDir="."):
    """Samples a specified class balanced number of samples form the KTH-TIPS2 dataset
    
    returns:
        loader
    """
    datasetPath = Path(os.path.realpath(__file__)).parent.parent.parent/'data'/'KTH'
    print(datasetPath)
    
    if not os.path.isdir(datasetPath):
        downloadKTH_TIPS2()

    transform_train = kth_augmentationFactory(trainAugmentation, height, width)
    transform_val = kth_augmentationFactory('noaugment', height, width)

    loader = KTHLoader(data_dir=dataDir, train_batch_size=trainBatchSize, 
                       val_batch_size=valBatchSize, transform_train=transform_train, 
                       transform_val=transform_val, sample=sample)

    return loader

class KTHLoader():
    """Class for loading the KTH texture dataset"""
    def __init__(self, data_dir, train_batch_size, val_batch_size, 
                 transform_train, transform_val, sample='a'):

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.transform_train =  transform_train
        self.transform_val = transform_val
        self.sample = sample

    def generateNewSet(self, device, workers=5, seed=None, load=False):
        """ 
        Generates train and test loader for KTH dataset
        KTH-TIPS2 is a dataset that has 4 different samples (a, b c and d)
        See dataset details here: https://www.csc.kth.se/cvap/databases/kth-tips/credits.html
        Parameters:
                device  -- cuda or cupu
                workers -- number of workers
                seed    -- seed
                load    -- boolean to indicates if we want to load the dataset      

        returns:
            train_loader -- train_loader
            test_loader  -- test_loader
            seed         -- seed
        """
        datasets_val = []
        for s in ['a', 'b', 'c', 'd']:
            if self.sample == s:
                dataset = datasets.ImageFolder(#load train dataset
                    root=os.path.join(self.data_dir,s), 
                    transform=self.transform_train
                )
                dataset_train = dataset
            else:
                dataset = datasets.ImageFolder(#load train dataset
                    root=os.path.join(self.data_dir,s), 
                    transform=self.transform_val
                )

                datasets_val.append(dataset)

        dataset_val = torch.utils.data.ConcatDataset(datasets_val)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.train_batch_size, 
                                                   shuffle=True, num_workers=workers,
                                                   pin_memory=True)

        test_loader = torch.utils.data.DataLoader(dataset_val, batch_size=self.val_batch_size, 
                                                  shuffle=True, num_workers=workers, 
                                                  pin_memory=True)

        self.trainSampleCount, self.valSampleCount = sum([len(x) for x in train_loader]), sum([len(x) for x in test_loader])

        if load:
            for batch,target in train_loader:
                batch.cuda()
                target.cuda()

            for batch,target in test_loader:
                batch.cuda()
                target.cuda()    

        if seed == None:
            seed = int(time.time()) #generate random seed
        else:
            seed = seed

        return train_loader, test_loader, seed





def download_from_url(link, file_name):
    """Download Dataset from URL
    FROM: https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    Parameters:
        link -- url to dataset
        file_name -- file name of the dataset
    """
    with open(file_name, "wb") as f:
        print("Downloading %s" % file_name)
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None: # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                sys.stdout.flush()

def extract_tar(file_name, target_path):
    """Extract files from tar
    Parameters:
        file_name   -- file name of the dataset
        target_path -- path to the target dataset folder
    """
    print("Extracting %s" % file_name)
    with tarfile.open(name=file_name) as tar:
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            tar.extract(path= target_path, member=member)
    os.remove(file_name)

def create_dataset(target_path):
    """Create KTH dataset in the target folder
    Parameters:
        target_path -- path to the new dataset folder
    """
    folders = glob(f'{target_path}/KTH-TIPS2-b/*/*')
    print("Creating new dataset folder")
    for folder in tqdm(folders):
        new_folder = os.path.join(target_path, "KTH")
        sample = folder.split('/')[-1][-1]
        label = folder.split('/')[-2]
        destination_path = os.path.join(new_folder, f'{sample}/{label}')
        print(destination_path)
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)   
        pattern = f'{folder}*/*' 
        for img in glob(pattern):
            shutil.copy(img, destination_path)


def downloadKTH_TIPS2():    
    target_path = Path(os.path.realpath(__file__)).parent.parent.parent/'data'
    target_path.mkdir(parents=True, exist_ok= True)

    link = 'https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b_col_200x200.tar'
    file_name ='kth-tips2-b_col_200x200.tar'
    download_from_url(link, file_name)
    extract_tar(file_name, target_path)
    create_dataset(target_path)
    
    # remove extracted folder (not necessary)
    mydir = os.path.join(target_path,'KTH-TIPS2-b')
    try:
        shutil.rmtree( mydir)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
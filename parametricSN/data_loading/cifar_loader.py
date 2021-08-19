"""Subsamples cifar-10

Author: Benjamin Therien

Functions:
    cifar_getDataloaders -- samples from the cifar-10 dataset based on input
    cifar_augmentationFactory -- returns different augmentations for cifar-10

"""

from torchvision import datasets, transforms

from parametricSN.data_loading.auto_augment import AutoAugment, Cutout
from parametricSN.data_loading.SmallSampleController import SmallSampleController




def cifar_augmentationFactory(augmentation):
    """Factory for different augmentation choices"""

    if augmentation == 'autoaugment':
        transform = [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            Cutout()
        ]
    elif augmentation == 'original-cifar':
        transform = [
            transforms.RandomCrop(32, 4),
            transforms.RandomHorizontalFlip(),
        ]
    elif augmentation == 'noaugment':
        transform = []
    elif augmentation == 'glico':
        NotImplemented(f"augment parameter {augmentation} not implemented")
    else: 
        NotImplemented(f"augment parameter {augmentation} not implemented")

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    # more precise cifar normalization thanks to:
    # https://github.com/tomgoldstein/loss-landscape/blob/master/cifar10/dataloader.py#L16

    return transforms.Compose(transform + [transforms.ToTensor(), normalize])

def cifar_getDataloaders(trainSampleNum, valSampleNum, trainBatchSize, 
                         valBatchSize, trainAugmentation, dataDir="."):
    """Samples a specified class balanced number of samples form the Cifar-10 dataset
    
    returns:
        train_loader, test_loader, seed, glico_dataset
    """
    
    transform_train = cifar_augmentationFactory(trainAugmentation)
    transform_val = cifar_augmentationFactory("noaugment")

    dataset_train = datasets.CIFAR10(#load train dataset
        root=dataDir, train=True, 
        transform=transform_train, download=True
    )

    dataset_val = datasets.CIFAR10(#load test dataset
        root=dataDir, train=False, 
        transform=transform_val, download=True
    )

    ssc = SmallSampleController(
        trainSampleNum=trainSampleNum, valSampleNum=valSampleNum, 
        trainBatchSize=trainBatchSize, valBatchSize=valBatchSize,  
        trainDataset=dataset_train, valDataset=dataset_val 
    ) 

    return ssc












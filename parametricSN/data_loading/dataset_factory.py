"""Factory module for the datasets used

Functions:
    datasetFactory -- Factory for Cifar-10, kth-tips2, and COVID-CRX2 datasets 
"""

from parametricSN.data_loading.cifar_loader import cifar_getDataloaders
from parametricSN.data_loading.kth_loader import kth_getDataloaders
from parametricSN.data_loading.xray_loader import xray_getDataloaders


def datasetFactory(params, dataDir, use_cuda):
    """ Factory for Cifar-10, kth-tips2, and COVID-CRX2 datasets

    Creates and returns different dataloaders and datasets based on input

    parameters: 
        params -- dict of input parameters
        dataDir -- path to the dataset

    returns:
        train_loader, test_loader, seed
    """

    if params['dataset']['name'].lower() == "cifar":
        return cifar_getDataloaders(
                    trainSampleNum=params['dataset']['train_sample_num'], valSampleNum=params['dataset']['test_sample_num'], 
                    trainBatchSize=params['dataset']['train_batch_size'], valBatchSize=params['dataset']['test_batch_size'], 
                    trainAugmentation=params['dataset']['augment'], dataDir=dataDir
                )
    elif params['dataset']['name'].lower() == "kth":
        return kth_getDataloaders(
                    trainBatchSize=params['dataset']['train_batch_size'], valBatchSize=params['dataset']['test_batch_size'], 
                    trainAugmentation=params['dataset']['augment'], height=params['dataset']['height'], 
                    width=params['dataset']['width'], sample=params['dataset']['sample'], 
                    dataDir=dataDir
                )
    elif params['dataset']['name'].lower() == "x-ray":
        return xray_getDataloaders(
            trainSampleNum=params['dataset']['train_sample_num'], valSampleNum=params['dataset']['test_sample_num'], 
            trainBatchSize=params['dataset']['train_batch_size'], valBatchSize=params['dataset']['test_batch_size'], 
            trainAugmentation=params['dataset']['augment'], height=params['dataset']['height'], 
            width=params['dataset']['width'], dataDir=dataDir
        )
    else:
        raise NotImplemented(f"Dataset {params['dataset']['name']} not implemented")
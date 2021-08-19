"""Main module for Learnable Scattering Networks

Authors: Benjamin Therien, Shanel Gauthier

Functions: 
    run_train -- callable functions for the program
    main -- parses arguments an calls specified callable

"""
import sys
import os
from pathlib import Path 
sys.path.append(str(Path.cwd()))

import os
import time
import argparse
import torch
import math
import kymatio.datasets as scattering_datasets
import numpy as np

from parametricSN.utils.helpers import get_context, visualize_loss
from parametricSN.utils.helpers import visualize_learning_rates
from parametricSN.utils.helpers import  log_mlflow, getSimplePlot
from parametricSN.utils.helpers import  override_params, setAllSeeds
from parametricSN.utils.helpers import estimateRemainingTime
from parametricSN.utils.optimizer_factory import optimizerFactory
from parametricSN.utils.scheduler_factory import schedulerFactory
from parametricSN.data_loading.dataset_factory import datasetFactory

from parametricSN.models.models_factory import topModelFactory, baseModelFactory
from parametricSN.models.sn_hybrid_models import sn_HybridModel
from parametricSN.training.training_factory import train_test_factory


def get_data_root(dataset_name, data_root, data_folder):
    """ Get the path to the dataset.
        If the path is None, we assume the dataset is in the data folder
        that was generated automatically using the scripts (in parametricSN/datasets)

    parameters:
        dataset_name -- the name of the dataset (cifar, x-ray or KTH)
        data_root    -- path to the dataset folder
        data_folder  -- dataset folder name
    """
    if  data_root != None:
        DATA_DIR = Path( data_root)/data_folder
    elif dataset_name=='cifar':
        DATA_DIR = scattering_datasets.get_dataset_dir('CIFAR')
    elif dataset_name=='x-ray':
        DATA_DIR = Path(os.path.realpath(__file__)).parent.parent/'data'/'xray_preprocess'
    elif dataset_name=='KTH':
        DATA_DIR = Path(os.path.realpath(__file__)).parent.parent/'data'/'KTH' 
    return DATA_DIR

def run_train(args):
    """Launches the training script 

    parameters:
        args -- namespace of arguments passed from CLI
    """
    torch.backends.cudnn.deterministic = True #Enable deterministic behaviour
    torch.backends.cudnn.benchmark = False #Enable deterministic behaviour

    params = get_context(args.param_file) #parse params
    params = override_params(args,params) #override from CLI

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    DATA_DIR = get_data_root(params['dataset']['name'], params['dataset']['data_root'], params['dataset']['data_folder'])

    ssc = datasetFactory(params,DATA_DIR,use_cuda) #load Dataset

    train_loader, test_loader, params['general']['seed'] = ssc.generateNewSet(#Sample from datasets
        device, workers=params['general']['cores'],
        seed=params['general']['seed'],
        load=False
    ) 

    setAllSeeds(seed=params['general']['seed'])

    scatteringBase = baseModelFactory( #creat scattering base model
        architecture=params['scattering']['architecture'],
        J=params['scattering']['J'],
        N=params['dataset']['height'],
        M=params['dataset']['width'],
        second_order=params['scattering']['second_order'],
        initialization=params['scattering']['init_params'],
        seed=params['general']['seed'],
        learnable=params['scattering']['learnable'],
        lr_orientation=params['scattering']['lr_orientation'],
        lr_scattering=params['scattering']['lr_scattering'],
        filter_video=params['scattering']['filter_video'],
        device=device,
        use_cuda=use_cuda
    )

    setAllSeeds(seed=params['general']['seed'])
    
    top = topModelFactory( #create cnn, mlp, linearlayer, or other
        base=scatteringBase,
        architecture=params['model']['name'],
        num_classes=params['dataset']['num_classes'], 
        width= params['model']['width'], 
        use_cuda=use_cuda
    )

    hybridModel = sn_HybridModel(scatteringBase=scatteringBase, top=top, use_cuda=use_cuda) #creat hybrid model

    optimizer = optimizerFactory(hybridModel=hybridModel, params=params)

    #use gradient accumulation if VRAM is constrained
    if params['model']['loss'] == 'cross-entropy-accum':
        if params['dataset']['accum_step_multiple'] % params['dataset']['train_batch_size'] != 0:
            print("Incompatible batch size and accum step multiple")
            raise Exception
        else:
            steppingSize = int(params['dataset']['accum_step_multiple']/params['dataset']['train_batch_size'])

        params['dataset']['accum_step_multiple']
        scheduler = schedulerFactory(
            optimizer=optimizer, params=params, 
            steps_per_epoch=math.ceil(ssc.trainSampleCount/params['dataset']['accum_step_multiple'])
        )
    else:
        scheduler = schedulerFactory(optimizer=optimizer, params=params, steps_per_epoch=len(train_loader))
        steppingSize = None

    test_acc = []
    start_time = time.time()
    train_losses, test_losses , train_accuracies = [], [], []
    lrs, lrs_scattering, lrs_orientation = [], [], []
    param_distance = []

    trainTime = []
    testTime = []

    if params['scattering']['param_distance']: 
        param_distance.append(hybridModel.scatteringBase.checkParamDistance())
    
    params['model']['trainable_parameters'] = '%fM' % (hybridModel.countLearnableParams() / 1000000.0)
    print("Starting train for hybridModel with {} parameters".format(params['model']['trainable_parameters']))

    train, test = train_test_factory(params['model']['loss'])

    for epoch in  range(0, params['model']['epoch']):
        t1 = time.time()
        hybridModel.scatteringBase.setEpoch(epoch)

        try:
            lrs.append(optimizer.param_groups[0]['lr'])
            if params['scattering']['learnable']:
                lrs_orientation.append(optimizer.param_groups[1]['lr'])
                lrs_scattering.append(optimizer.param_groups[2]['lr'])
        except Exception:
            pass
        
        train_loss, train_accuracy = train(hybridModel, device, train_loader, scheduler, optimizer, 
                                           epoch+1, accum_step_multiple=steppingSize)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        if params['scattering']['param_distance']: 
            param_distance.append(hybridModel.scatteringBase.checkParamDistance())

        trainTime.append(time.time()-t1)
        if epoch % params['model']['step_test'] == 0 or epoch == params['model']['epoch'] -1: #check test accuracy
            t1 = time.time()
            accuracy, test_loss = test(hybridModel, device, test_loader)
            test_losses.append(test_loss)
            test_acc.append(accuracy)

            testTime.append(time.time()-t1)
            estimateRemainingTime(trainTime=trainTime,testTime=testTime,epochs=params['model']['epoch'],currentEpoch=epoch,testStep=params['model']['step_test'])

    if params['scattering']['filter_video']:
        hybridModel.scatteringBase.releaseVideoWriters()

    if params['scattering']['param_distance']:
        compareParamsVisualization = hybridModel.scatteringBase.compareParamsVisualization()
        torch.save(hybridModel.scatteringBase.params_history,
                   os.path.join('/tmp',"{}_{}.pt".format(params['scattering']['init_params'],params['mlflow']['experiment_name'])))


    #MLFLOW logging below
    f_loss = visualize_loss(
        train_losses, test_losses, step_test=params['model']['step_test'], 
        y_label='loss'
    )
                             
    f_accuracy = visualize_loss(
        train_accuracies ,test_acc, step_test=params['model']['step_test'], 
        y_label='accuracy'
    )
                             
    f_accuracy_benchmark = visualize_loss(
        train_accuracies, test_acc, step_test=params['model']['step_test'], 
        y_label='accuracy'
    )

    #visualize learning rates
    f_lr = visualize_learning_rates(lrs, lrs_orientation, lrs_scattering)

    paramDistancePlot = getSimplePlot(xlab='Epochs', ylab='Min Distance to TF params',
        title='Learnable parameters progress towards the TF initialization parameters', label='Dist to TF params',
        xvalues=[x+1 for x in range(len(param_distance))], yvalues=param_distance)


    if params['scattering']['architecture']  == 'scattering':
        #visualize filters
        filters_plots_before = hybridModel.scatteringBase.filters_plots_before
        hybridModel.scatteringBase.updateFilters() #update the filters based on the latest param update
        filters_plots_after = hybridModel.scatteringBase.getFilterViz() #get filter plots
        filters_values = hybridModel.scatteringBase.plotFilterValues()
        filters_grad = hybridModel.scatteringBase.plotFilterGrads()
        filters_parameters = hybridModel.scatteringBase.plotParameterValues()
    else:
        filters_plots_before = None
        filters_plots_after = None
        filters_values = None
        filters_grad = None
        filters_parameters = None

    log_mlflow(
        params=params, model=hybridModel, test_acc=np.array(test_acc).round(2), 
        test_loss=np.array(test_losses).round(2), train_acc=np.array(train_accuracies).round(2), 
        train_loss=np.array(train_losses).round(2), start_time=start_time, 
        filters_plots_before=filters_plots_before, filters_plots_after=filters_plots_after,
        misc_plots=[f_loss, f_accuracy, f_accuracy_benchmark, filters_grad, 
        filters_values, filters_parameters, f_lr, paramDistancePlot,compareParamsVisualization]
    )
    


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("run-train")
    subparser.set_defaults(callback=run_train)
    #general
    subparser.add_argument("--general-cores", "-gc", type=int)
    subparser.add_argument("--general-seed", "-gseed", type=int)
    subparser.add_argument("--general-save-metric", "-gsm", type=int)
    #mlflow 
    subparser.add_argument("--mlflow-tracking-uri", "-turi", type=str)
    subparser.add_argument("--mlflow-experiment-name", "-en", type=str)
    #dataset
    subparser.add_argument("--dataset-name", "-dname", type=str, choices=['cifar', 'kth', 'x-ray'])
    subparser.add_argument("--dataset-num-classes", "-dnc", type=int)
    subparser.add_argument("--dataset-train-batch-size", "-dtbs", type=int)
    subparser.add_argument("--dataset-test-batch-size", "-dtstbs", type=int)
    subparser.add_argument("--dataset-train-sample-num", "-dtsn", type=int)
    subparser.add_argument("--dataset-test-sample-num", "-dtstsn", type=int)
    subparser.add_argument("--dataset-accum-step-multiple", "-dasm", type=int)
    subparser.add_argument("--dataset-data-root", "-ddr", type=str)
    subparser.add_argument("--dataset-data-folder", "-ddf", type=str)
    subparser.add_argument("--dataset-height", "-dh", type=int)
    subparser.add_argument("--dataset-width", "-dw", type=int)
    subparser.add_argument("--dataset-augment", "-daug", type=str, choices=['autoaugment','original-cifar','noaugment'])
    subparser.add_argument("--dataset-sample", "-dsam", type=str, choices=['a','b','c','d'])
    #scattering
    subparser.add_argument("--scattering-J", "-sj", type=int)
    subparser.add_argument("--scattering-max-order", "-smo", type=int)
    subparser.add_argument("--scattering-lr-scattering", "-slrs", type=float)
    subparser.add_argument("--scattering-lr-orientation", "-slro", type=float)
    subparser.add_argument("--scattering-init-params", "-sip", type=str,choices=['Tight-Frame','Random'])
    subparser.add_argument("--scattering-learnable", "-sl", type=int, choices=[0,1])
    subparser.add_argument("--scattering-second-order", "-sso", type=int, choices=[0,1])
    subparser.add_argument("--scattering-max-lr", "-smaxlr", type=float)
    subparser.add_argument("--scattering-div-factor", "-sdivf", type=int)
    subparser.add_argument("--scattering-architecture", "-sa", type=str, choices=['scattering','identity'])
    subparser.add_argument("--scattering-three-phase", "-stp", type=int, choices=[0,1])
    subparser.add_argument("--scattering-filter-video", "-sfv", type=int, choices=[0,1])
    subparser.add_argument("--scattering-param-distance", "-spd", type=int, choices=[0,1])


    #optim
    subparser.add_argument("--optim-name", "-oname", type=str,choices=['adam', 'sgd'])
    subparser.add_argument("--optim-lr", "-olr", type=float)
    subparser.add_argument("--optim-weight-decay", "-owd", type=float)
    subparser.add_argument("--optim-momentum", "-omo", type=float)
    subparser.add_argument("--optim-max-lr", "-omaxlr", type=float)
    subparser.add_argument("--optim-div-factor", "-odivf", type=int)
    subparser.add_argument("--optim-three-phase", "-otp", type=int, choices=[0,1])
    subparser.add_argument("--optim-scheduler", "-os", type=str, choices=['CosineAnnealingLR','OneCycleLR','LambdaLR','StepLR','NoScheduler'])    
    subparser.add_argument("--optim-phase-num", "-opn", type=int)
    subparser.add_argument("--optim-phase-ends", "-ope", nargs="+", default=None)
    subparser.add_argument("--optim-T-max", "-otmax", type=int)

    #model 
    subparser.add_argument("--model-name", "-mname", type=str, choices=['cnn', 'mlp', 'linear_layer', 'resnet50'])
    subparser.add_argument("--model-width", "-mw", type=int)
    subparser.add_argument("--model-epoch", "-me", type=int)
    subparser.add_argument("--model-step-test", "-mst", type=int)
    subparser.add_argument("--model-loss", "-mloss", type=str, choices=['cosine', 'cross-entropy','cross-entropy-accum'])
    subparser.add_argument("--model-save", "-msave", type=int, choices=[0,1])

    subparser.add_argument('--param_file', "-pf", type=str, default='parameters.yml',
                        help="YML Parameter File Name")

    args = parser.parse_args()

    for key in ['optim_three_phase','scattering_learnable',
                'scattering_second_order','scattering_three_phase',
                'scattering_filter_video','scattering_param_distance']:
        if args.__dict__[key] != None:
            args.__dict__[key] = bool(args.__dict__[key]) #make 0 and 1 arguments booleans

    args.callback(args)


if __name__ == '__main__':
    main()

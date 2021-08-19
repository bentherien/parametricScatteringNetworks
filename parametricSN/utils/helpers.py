"""Helpers for the main script

Authors: Benjamin Therien, Shanel Gauthier

Functions:
    get_context              -- Create dictionnaries from yaml files
    visualize_loss           -- Plot Loss/accuracy   
    visualize_learning_rates -- Plot learning rates
    getSimplePlot            -- Generic function to generate simple plots  
    log_csv_file             -- Save dictionnary into a csv file using mlflow
    rename_params            -- Rename the name of the keys of a dictionnary by 
                                adding a prefix to the existing name
    log_mlflow               -- Log statistics in mlflow
    override_params          -- Override passed params dict with any CLI arguments
    setAllSeeds              -- Helper for setting seeds
    estimateRemainingTime    -- Estimates the remaining training time based on imput
"""

import random
import torch
import math
import time
import mlflow
import os
import yaml
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Process
from pathlib import Path 

sys.path.append(str(Path.cwd()))

from parametricSN.models.models_utils import compareParams

def get_context(parameters_file, full_path = False):
    """ Read yaml file that contains experiment parameters.
        Create dictionnaries from the yaml file.         
        
        Parameters:
            parameters_file -- the name of yaml file that contains the parameters
                                or the full path to the yaml file
            full_path       -- boolean that indicates if parameters_file is the name 
                                of the yaml file or the full path  
        Returns:
            params          -- dictionnary that contains experiment parameters
    """
    current_dir = Path.cwd()
    proj_path = current_dir
    sys.path.append(os.path.join(proj_path, 'kymatio'))
    if full_path:
        params_path = parameters_file
    else:
        params_path = os.path.join(proj_path, f'conf/{parameters_file}')
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)     
    return params


def visualize_loss(loss_train ,loss_test, step_test = 10, y_label='loss'):
    """ Plot Loss/accuracy       
        
        Parameters:
            loss_train -- list that contains all the train loss for each epoch
            loss_test  -- list that contains test loss for every 'step_test' epoch
            step_test  -- epoch interval that was used to save the test loss
            y_label    -- label to be displayed on the y axis

        Returns:
            f         -- figure (loss over epoch)
    """
    f = plt.figure (figsize=(7,7))
    plt.plot(np.arange(len(loss_test)*step_test, step=step_test), loss_test, label=f'Test {y_label}') 
    plt.plot(np.arange(len(loss_train)), loss_train, label= f'Train {y_label}')
    plt.ylabel(y_label)
    plt.xlabel('Epoch')
    plt.legend() 
    return f  

def visualize_learning_rates(lrs, lrs_orientation, lrs_scattering):
    """ Plot learning rates     
        
        Parameters:
            lrs              -- list that contains the learning rates used per epoch for the top model
            lrs_orientation  -- list that contains the learning rate used per epoch for the wavelet
                                orientation parameter (scattering model)
            lrs_scattering   -- list that contains the learning rates used per epoch for the wavelet
                                xis, sigmas and slants parameters (scattering model)

        Returns:
            f -- figure (learning rate over epoch)
    """
    f = plt.figure (figsize=(7,7))
    epochs = np.arange(len(lrs))
    plt.plot(epochs, lrs, label='Linear Layer LR') 

    if len(lrs_orientation) > 0:
        plt.plot(epochs, lrs_orientation, label='Orientation LR')

    if len(lrs_scattering) > 0:
        plt.plot(epochs, lrs_scattering, label='Scattering LR')

    plt.ylabel('LR')
    plt.xlabel('Epoch')
    plt.legend() 
    return f  

def getSimplePlot(xlab,ylab,title,label,xvalues,yvalues,figsize=(7,7)):
    """ Generic function to generate simple plots    
        
        Parameters:
            xlab    -- label for x axis
            ylab    -- label for y axis
            title   -- title of the plot
            xvalue  -- list or numpy array that contains the data points for x axis
            yvalues -- list or numpy array that contains the data points for x axis
            figsize -- figure size
        Returns:
            plot -- figure
    """
    plot = plt.figure(figsize=figsize)
    plt.title(title)
    plt.plot(xvalues, yvalues, label=label) 
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.legend() 
    return plot

def log_csv_file(name, file):
    """ Save dictionnary into a csv file and log the csv file using mlflow.
        Once the file is saved in mlflow, the file is deleted. 

        Parameters:
            name    -- name of the file
            file    -- dictionnary 
    """
    np.savetxt(name,  file, delimiter=",")
    mlflow.log_artifact(name, 'metrics')
    os.remove(name)

def rename_params(prefix, params):
    """ Rename the name of the keys of a dictionnary by adding a 
        prefix to the existing name
        Parameters:
            prefix   -- prefix
            file    -- dictionnary 
    """
    return {f'{prefix}-' + str(key): val for key, val in  params.items()}

def log_mlflow(params, model, test_acc, test_loss, train_acc, 
               train_loss, start_time, filters_plots_before, 
               filters_plots_after, misc_plots):
    """Log stats in mlflow
    
    parameters: 
        params               -- the parameters passed to the program
        model                -- the hybrid model used during training 
        test_acc             -- list of test accuracies over epochs
        test_loss            -- list of test losses over epochs
        train_acc            -- list of train accuracies over epochs
        train_loss           -- list of train losses over epochs
        start_time           -- the time at which the current run was started 
        filters_plots_before -- plots of scattering filter values before training 
        filters_plots_after  -- plots of scattering filter values after training 
        misc_plots           -- a list of miscelaneous plots to log in mlflow
    """

    duration = (time.time() - start_time)
    if params['mlflow']['tracking_uri'] is None:
        tracking_uri_folder = Path(os.path.realpath(__file__)).parent.parent.parent/'mlruns'
        try:
            tracking_uri_folder.mkdir(parents=True, exist_ok= True)
        except:
            pass
        params['mlflow']['tracking_uri'] = 'sqlite:///'+ str(tracking_uri_folder/'store.db')
    
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment(params['mlflow']['experiment_name'])

    with mlflow.start_run():
        mlflow.log_params(rename_params('model', params['model']))   
        mlflow.log_params(rename_params('scattering', params['scattering']))
        mlflow.log_params(rename_params('dataset', params['dataset']))
        mlflow.log_params(rename_params('optim', params['optim']))
        mlflow.log_params(params['general'])
        mlflow.log_param('Duration', duration)
        mlflow.log_metric('Final Accuracy', test_acc[-1])
        if params['model']['save']:
            mlflow.pytorch.log_model(model, artifact_path = 'model')
            mlflow.log_dict(params, "model/parameters.yml")
        #save filters 
        try:
            for key in filters_plots_before:
                
                    mlflow.log_figure(filters_plots_before[key], f'filters_before/{key}.pdf')
                    mlflow.log_figure(filters_plots_after[key], f'filters_after/{key}.pdf')
        except:
            pass

        mlflow.log_figure(misc_plots[0], f'plot/train_test_loss.pdf')
        mlflow.log_figure(misc_plots[1], f'plot/train_test_accuracy.pdf')
        mlflow.log_figure(misc_plots[2], f'plot/train_test_accuracy_2.pdf')
        
        try:
            mlflow.log_figure(misc_plots[3], f'learnable_parameters/filters_grad.pdf')
            mlflow.log_figure(misc_plots[4], f'learnable_parameters/filter_values.pdf')
            mlflow.log_figure(misc_plots[5], f'learnable_parameters/filter_parameters.pdf')
        except:
            pass

        mlflow.log_figure(misc_plots[6], f'plot/lr.pdf')
        mlflow.log_figure(misc_plots[7], f'learnable_parameters/param_distance.pdf')

        if params['scattering']['param_distance']: 
            mlflow.log_figure(misc_plots[8], f'learnable_parameters/param_match_visualization.pdf')


        # saving all accuracies
        log_csv_file('test_acc.csv', test_acc)
        log_csv_file('train_acc.csv', train_acc)
        log_csv_file('test_loss.csv', test_loss)
        log_csv_file('train_loss.csv', train_loss)
        print(f"finished logging to {params['mlflow']['tracking_uri']}")


def override_params(args, params):
    """override passed params dict with any CLI arguments
    
    parameters: 
        args -- namespace of arguments passed from CLI
        params -- dict of default arguments list    
    """
    for k,v in args.__dict__.items():
        if v != None and k != "param_file":
            tempSplit = k.split('_')
            prefix = tempSplit[0]
            key = "_".join(tempSplit[1:])
            try:
                params[prefix][key] = v
            except KeyError:
                pass

    return params


def setAllSeeds(seed):
    """Helper for setting seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def estimateRemainingTime(trainTime, testTime, epochs, currentEpoch, testStep):
    """Estimates the remaining training time based on imput
    
    Estimates remaining training time by using averages of the 
    each training and test epoch computed. Displays a message 
    indicating averages expected remaining time.

    parameters:
        trainTime -- list of time elapsed for each training epoch
        testTime -- list of time elapsed for each testing epoch
        epochs -- the total number of epochs specified
        currentEpoch -- the current epoch 
        testStep -- epoch multiple for validation set verfification 
    """
    meanTrain = np.mean(trainTime)
    meanTest = np.mean(testTime)

    remainingEpochs = epochs - currentEpoch

    remainingTrain = (meanTrain *  remainingEpochs) / 60
    remainingTest = (meanTest * (int(remainingEpochs / testStep) + 1)) / 60
    remainingTotal = remainingTest + remainingTrain

    print("[INFO] ~{:.2f} m remaining. Mean train epoch duration: {:.2f} s. Mean test epoch duration: {:.2f} s.".format(
        remainingTotal, meanTrain, meanTest
    ))

    return remainingTotal




def experiments_cli():
    """CLI arguments for experiments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-dr", type=str)
    parser.add_argument("--data-folder", "-df", type=str)
    parser.add_argument("--python", "-p", type=str)

    args = parser.parse_args()

    if args.data_root != None and args.data_folder != None:
        DATA_ARG = "-ddr {} -ddf {}".format(args.data_root,args.data_folder)
    else:
        DATA_ARG = ""

    return sys.executable, DATA_ARG


def experiments_runCommand(cmd):
    """runs one command"""
    print("[Running] {}".format(cmd))
    os.system(cmd)


def experiments_mpCommands(processBatchSize, commands):
    """runs commands in parallel"""
    processes = [Process(target=experiments_runCommand,args=(commands[i],)) for i,cmd in enumerate(commands)]
    processBatches = [processes[i*processBatchSize:(i+1)*processBatchSize] for i in range(math.ceil(len(processes)/processBatchSize))]

    for i,batch in enumerate(processBatches):
        print("Running process batch {}".format(i))
        startTime = time.time()

        for process in batch:
            process.start()
            time.sleep(5)

        for process in batch:
            process.join()

        print("\n\nRunning Took {} seconds".format(time.time() - startTime))
        time.sleep(1)


def logComparison(mlflow_exp_name):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paramsTF = torch.load(os.path.join('/tmp',"{}_{}.pt".format('Tight-Frame',mlflow_exp_name.strip("\""))))
    paramsRand = torch.load(os.path.join('/tmp',"{}_{}.pt".format('Random',mlflow_exp_name.strip("\""))))

    param_distance = []
    for i in range(len(paramsTF)):
        param_distance.append(
            compareParams(
            params1=paramsTF[i]['params'],
            angles1=paramsTF[i]['angle'], 
            params2=paramsRand[i]['params'],
            angles2=paramsRand[i]['angle'],
            device=device
            )
        )

    paramDistancePlot = getSimplePlot(xlab='Epochs', ylab='Distance',
        title='TF and Randomly intialized parameters distances from one another as they are optimized', label='Distance',
        xvalues=[x+1 for x in range(len(param_distance))], yvalues=param_distance)


    temp = str(os.path.join(os.getcwd(),'mlruns'))
    if not os.path.isdir(temp):
        os.mkdir(temp)

    mlflow.set_tracking_uri('sqlite:///' + os.path.join(temp,'store.db'))
    mlflow.set_experiment(mlflow_exp_name.strip("\""))


    with mlflow.start_run(run_name='filter comparison'):
        mlflow.log_figure(paramDistancePlot, 'learnable_parameters/param_distance.pdf')
        print(f"finished logging to {'sqlite:///' + os.path.join(temp,'store.db')}")

    os.system('rm {}'.format(os.path.join('/tmp',"{}_{}.pt".format('Tight-Frame',mlflow_exp_name))))
    os.system('rm {}'.format(os.path.join('/tmp',"{}_{}.pt".format('Random',mlflow_exp_name))))




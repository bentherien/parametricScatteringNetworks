"""Evaluate representation stability of deformed images

Authors: Laurent Alsene-Racicot, Shanel Gauthier

This script computes distances between scattering representations from image and transformed versions
of the original image. The figures generated from this script are saved in mlflow.

To run the script, you to pass as command line arguments one or multiple paths to the model folder such as:
"/NOBACKUP/gauthiers/kymatio_mod/mlruns/5/03f1f015288f47dc81d0529b23c25bf1/artifacts/model" 

Functions: 
    append_to_list         -- Append element to lists
    apply_transformation   -- Defines the transformation levels and calls
                              compute_l2norm function
    get_l2norm_deformation -- Call apply_transformation for various deformations and get the 
                              baseline for a given model
    load_models_weights    -- Loads model weigths
    get_loaders            -- Get test and train loaders
    get_baseline           -- Create baselines by computing the distances between representations from the
                              original image and random images.
    compute_l2norm         -- Computed distances between representations from image and transformed versions
                              of the original image      
    visualize_distances    -- Creates figures for each transformation
    diffeo                 -- Apply a custom transformation of the form x(u-tau(u)) to the image
    deformation_size       -- Compute the deformation size of a given transformation : sup |J_{tau}(u)| 
                              over u for J the jacobian
    log_mlflow             -- Logs parameters, metrics and figures to MLFLOW
    main                   -- Computes the distance between scattering representations of an image and transformed
                              versions of the same image
"""
import sys
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import ToPILImage
sys.path.append(str(Path.cwd()))

import torch
import mlflow
import torchvision

import torch.nn.functional as F
import kymatio.datasets as scattering_datasets
import numpy as np
import os

from parametricSN.utils.helpers import get_context
from parametricSN.main import datasetFactory
from parametricSN.utils.helpers import rename_params

def append_to_list(list1, list2, element1, element2):
    """ Append element to lists
        
        Parameters:
            list1 -- list 
            list2 -- list
            element1 -- element to append to list1
            element2 -- element to append to list2
    """
    list1.append(element1)
    list2.append(element2)

def apply_transformation(max_value, name, hybridModel, img, l2_norms, deformations, y, titles,  
                        imgs_deformed, transform = None, device = None, num_points = 20):
    """ Defines the transformation levels and calls compute_l2norm function 
            
        Parameters:
            max_value -- maximum transformation level
            name -- name of the transformation
            hybridModel -- scattering model 
            img -- image without any transformation
            l2_norms -- list of euclidien distance between scattering representations
            deformations -- list of deformation levels
            titles -- list of string (use to add the title for each plot)
            imgs_deformed -- list of deformed images (with maximal transformation level)
            transform -- torchvision transforms
            device -- device cuda or cpu
            num_points -- number of transformation levels
        
    """
    if max_value > num_points:
        deformation_levels = torch.arange(0,max_value, max_value/ num_points,dtype = int)
    else:
        deformation_levels = torch.arange(0,max_value, max_value/ num_points )
    
    l2_norm, deformation, img_deformed = compute_l2norm(hybridModel, name,img, deformation_levels,
                                                        transform, device)
    titles.append(f'Transformation: {name}')
    imgs_deformed.append(img_deformed)
    append_to_list(l2_norms, deformations, l2_norm, deformation)


def get_l2norm_deformation( model_path,  test_loader, img, device = None, num_data = 15):
    """ Call apply_transformation for various deformations and get the baseline for a given model

        Parameters:
            model_path -- path of the model
            test_loader -- test loader
            img -- image without any transformation
            device -- device cuda or cpu
            num_data -- number of transformation levels
            
        Returns:
            values -- dictionnary of all the informations required to create the plots

    """
    hybridModel= load_models_weights(model_path,device)
    _, params = get_context(os.path.join(model_path,'parameters.yml'), True) 
    
    print("Starting evaluate representationfor hybridModel".format(params['model']['trainable_parameters']))

    x, y, titles, transformation_names, imgs_deformed = [], [], [], [], []
    
    # rotation
    transform = torchvision.transforms.RandomAffine(degrees=[0,0])
    apply_transformation(max_value = 10, name = "rotation", hybridModel = hybridModel, img = img,
                         x=x,y=y,titles=titles,transformation_names=transformation_names, 
                         imgs_deformed=imgs_deformed,transform = transform, device = device,
                         num_data = 15)
    transformation_names.append("rotation")

    # distortion
    transform = torchvision.transforms.RandomPerspective(distortion_scale=0, p=1)
    apply_transformation(max_value = 0.2, name = "distortion", hybridModel = hybridModel, img = img,
                         x=x,y=y,titles=titles,transformation_names=transformation_names,
                         imgs_deformed=imgs_deformed,transform = transform, device = device,
                         num_data = 15)
    transformation_names.append("distortion")

    # shear
    transform = torchvision.transforms.RandomAffine(degrees = 0, shear= [0, 0])
    apply_transformation(max_value = 10, name = "shear", hybridModel = hybridModel, img = img,
                         x=x,y=y,titles=titles,transformation_names=transformation_names,
                         imgs_deformed=imgs_deformed,transform = transform, device = device,
                         num_data = 15)
    transformation_names.append("shear")


    # Sharpness
    transform = torchvision.transforms.RandomAdjustSharpness(sharpness_factor=100, p=1)
    apply_transformation(max_value = 50, name = "sharpness", hybridModel = hybridModel, img = img,
                         x=x,y=y,titles=titles,transformation_names=transformation_names,
                         imgs_deformed=imgs_deformed,transform = transform, device = device,
                         num_data = 15)
    transformation_names.append("sharpness")
    
    # Horizontal translation
    height = params['dataset']['height'] 
    max_translate = int(height* 0.1)
    transform = torchvision.transforms.RandomAffine(degrees = 0, translate=[0,0])
    apply_transformation(max_value = max_translate, name = "translation", hybridModel = hybridModel, 
                         img = img,x=x,y=y,titles=titles,transformation_names=transformation_names,
                         imgs_deformed=imgs_deformed,transform = transform, device = device,
                         num_data = 15)
    transformation_names.append("translation")

    # Mallat1
    apply_transformation(max_value = 1, name = "Mallat1", hybridModel = hybridModel, img = img,
                         x=x,y=y,titles=titles,transformation_names=transformation_names,
                         imgs_deformed=imgs_deformed,transform = None, device = device,
                         num_data = 15)

    distance  = get_baseline(img, (iter(test_loader)), hybridModel,  device)

    print("Done evaluating representationfor hybridModel: {}".format(model_path))
    
    values = {"model_path": model_path,"distance": distance,  "x":x, "y": y, "titles":titles,
              "params": params, "transformation_names":transformation_names,"image":imgs_deformed}
    return values    


def load_models_weights(model_path, device ):
    """Loads model weigths
        
        Parameters:
            model_path -- path to the model
            device: device cuda or cpu

        Returns:
            hybridModel -- the hybrid model loaded
    """
    hybridModel = mlflow.pytorch.load_model(model_path)
    hybridModel.to(device)
    hybridModel.eval()
    return hybridModel

def get_loaders(params, use_cuda):
    """ Get test and train loaders
        
        Parameters:
            params: dictionnary of all the parameters used for the experiments
            use_cuda: true if cuda is available, false if not

        Returns:
            train_loader -- train data loader
            test_loader -- test data loader

    """

    if params['dataset']['data_root'] != None:
        DATA_DIR = Path(params['dataset']['data_root'])/params['dataset']['data_folder'] 
    else:
        DATA_DIR = scattering_datasets.get_dataset_dir('CIFAR')
    train_loader, test_loader, params['general']['seed'], _ = datasetFactory(params,DATA_DIR,use_cuda) 
    return train_loader, test_loader

def get_baseline(img, it, hybridModel,  device, num_images=50):
    """ Create baselines by computing the distances between representations 
        from the original image and random images
        
        Parameters:
            img -- image without any transformation
            it -- images iterator
            hybridModel -- scattering model
            device -- device cuda or cpu
            num_images -- number of random images

        Returns:
            average distance between representations from the original image and random images
    """
    distances= []
    first_transformation = True
    with torch.no_grad():
        for i in range(num_images):
            if first_transformation:
                representation_0 = hybridModel.scatteringBase(img.to(device))
                first_transformation= False
            else:
                img2, _ = next(it)  
                representation = hybridModel.scatteringBase(img2.to(device) )
                distances.append(torch.dist(representation_0, representation ).item()/
                                torch.linalg.norm(representation_0).item())
    return np.array(distances).mean()


def compute_l2norm(hybridModel, deformation, img, deformation_list, transforms, device = None):
    """ Computed distances between representations from image and transformed versions of the
        original image
        
        Parameters:
            hybridModel -- scattering model
            deformation -- defomration name (for example rotation) 
            img -- image without any transformation
            deformation_list -- list of levels of transformations
            transforms -- torchvision transforms
            device -- cuda or cpu
        Returns:
            l2_norm -- list of distances between representations
            deformations -- list of deformation levels
            img_deformed -- transformed image using the maximum level of transformation

    """
    deformations= []
    l2_norm = []
    first_transformation = True
    with torch.no_grad():
        for v in deformation_list:
            if first_transformation:
                representation = hybridModel.scatteringBase(img.to(device))
                representation_0 = representation
                first_transformation= False
            else:
                if deformation == 'rotation':
                    transforms.degrees = [v.item(), v.item()]
                    img_deformed = transforms(img).to(device)
                elif deformation == 'shear':
                    transforms.shear = [v.item(), v.item()]
                    img_deformed = transforms(img).to(device)
                elif deformation == 'distortion':                    
                    transforms.distortion_scale = v.item()
                    img_deformed = transforms(img).to(device)
                elif deformation == 'sharpness':
                    transforms.sharpness_factor = v.item()
                    img_deformed = transforms(img).to(device)
                elif deformation == 'translation':
                    ret = transforms.get_params(transforms.degrees, transforms.translate,
                                                transforms.scale, transforms.shear, img.shape)
                    ret = list(ret)
                    ret[1] = (v.item(),0)
                    ret= tuple(ret)
                    img_deformed  = torchvision.transforms.functional.affine(img.to(device), *ret,
                                                                            interpolation=transforms.interpolation,
                                                                            fill=False)
                elif deformation == "Mallat1":
                    tau = lambda u : (v.item() *(0.5*u[0]+0.3*u[1]**2),v.item() *(0.3*u[1]) )      
                    img_deformed = diffeo(img.to(device),tau,device)

                
                representation = hybridModel.scatteringBase(img_deformed)
                if deformation == "Mallat1":
                    deformationSize = deformation_size(tau)
                    deformations.append(deformationSize)
                else:
                    deformations.append(v.item())
                l2_norm.append(torch.linalg.norm(representation_0 - representation).item()/
                              torch.linalg.norm(representation_0).item())
                
    l2_norm = np.array(l2_norm)
    deformations = np.array(deformations)
    return l2_norm, deformations, img_deformed

    
def visualize_distances(model_values, num_transformations = 4):
    """ Creates figures for each transformation
        The x axis of each figure is the transformation levels
        The y axis of each figure is the distance between scattering representations 
        of the original image and the transformed image
        
        Parameters:
            model_values-- dictionnary taht contains all the parameters used in the experiments
            num_transformations -- the number of tranformations which is equal tot he number of figures

        Returns:
            figures -- list of figures
                
    """
    plt.rcParams.update({'font.size': 20})
    colors = ['#ff0000','#0000ff', '#008000','#ffd700', '#800000', '#ff00ff' ]
    figures = []
    for idx in range(num_transformations):
        f = plt.figure(figsize=(10,10)) # create plots
        for c, model_value in enumerate(model_values):
            plt.scatter(x= model_value["x"][idx], y= model_value["y"][idx], 
                        label= f'{model_value["params"]["scattering"]["init_params"]} + {model_value["params"]["scattering"]["learnable"]}',
                        color =colors[c])
            plt.axhline(y= model_value['distance'], color=colors[c], linestyle='-')
            plt.xlabel("Deformation Size", fontsize=20)
            plt.title(model_value['titles'][idx],    fontsize=20)
            plt.ylabel('||S(x_tild) - S(x)||',   fontsize=20)
            plt.legend(fontsize = 20)
        figures.append(f)
    return figures

def diffeo(img,tau,device):
    """ Apply a custom transformation of the form x(u-tau(u)) to the image
        See section 2.1 and 3.1 of https://arxiv.org/pdf/1203.1513.pdf
        
        Parameters:
            img -- image without any transformation
            tau -- implicit fonction used to deform the image
            device -- device cuda or cpu

        Returns:
            img_transf -- image with the transformation applied
    """
    img = img
    # Number of pixels. Suppose square image.
    dim = img.shape[-1]
    # Create a (dim x dim) matrix of 2d vectors. Each vector represents the normalized position in the grid. 
    # Normalized means (-1,-1) is top left and (1,1) is bottom right.
    grid = torch.tensor([[[x,y] for x in torch.linspace(-1,1,dim)] for y in torch.linspace(-1,1,dim)])
    # Apply u-tau(u). 
    tau_mat = lambda grid : torch.tensor([[tau(grid[i,j,:]) for j in range(len(grid))] for i in range(len(grid))])
    grid_transf = (grid - tau_mat(grid)).unsqueeze(0).to(device)
    # Apply x(u-tau(u)) by interpolating the image at the index points given by grid_transf.
    img_transf = torch.nn.functional.grid_sample(img,grid_transf)
    return img_transf

def deformation_size(tau):
    """ Compute the deformation size of a given transformation : sup |J_{tau}(u)| over u for J the
        jacobian

        Parameters:
            tau -- implicit fonction used to deform the image

        Returns:
            the deformation size

    """
    # Set a precision. This is arbitrary.
    precision = 128
    # Create a (flatten) grid of points between (-1,-1) and (1,1). This is the same grid as in the 
    # previous function (but flatten), but it feels arbitrary also.
    points = [torch.tensor([x,y]) for x in torch.linspace(-1,1,precision) for y in torch.linspace(-1,1,precision)]
    # Evaluate the Jacobian of tau in each of those points. Returns a tensor of precision^2 x 2 x 2, 
    # i.e. for each point in points the 2 x 2 jacobian.
    jac = torch.stack(list(map(lambda point : torch.stack(torch.autograd.functional.jacobian(tau,point)), points)))
    # Find the norm of those jacobians.
    norm_jac = torch.linalg.matrix_norm(jac,ord=2,dim=(1, 2))
    # Return the Jacobian with the biggest norm.
    return torch.max(norm_jac)

def log_mlflow(params, model_values, figures, img):
    """ Logs parameters, metrics and figures to MLFLOW
        
        Parameters:
            params -- dictionnary taht contains all the parameters used in the experiments
            model_values -- list of dictionnaries. Each dictionnary contains the transformation details
            figures -- list of figures
            img -- image without any transformation
                
    """
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment('Transformation Experiment')
    with mlflow.start_run():
        ToPILImage = torchvision.transforms.ToPILImage()
        img = ToPILImage(img.squeeze(0)).convert("L")
        mlflow.log_params(rename_params('model', params['model']))   
        mlflow.log_params(rename_params('scattering', params['scattering']))
        mlflow.log_params(rename_params('dataset', params['dataset']))
        mlflow.log_params(rename_params('optim', params['optim']))
        mlflow.log_params(params['general'])
        mlflow.log_dict(params, "model/parameters.yml")
        for i,figure in enumerate(figures):
            mlflow.log_figure(figure, f'Deformation/{model_values[0]["transformation_names"][i]}/Deformation.pdf')        
            mlflow.log_image(img, f'Deformation/{model_values[0]["transformation_names"][i]}/Image_before.pdf')
            img_deformed = ToPILImage(model_values[0]["image"][i].squeeze(0)).convert("L")
            mlflow.log_image(img_deformed, f'Deformation/{model_values[0]["transformation_names"][i]}/Image_after.pdf')
        print(f"finish logging{params['mlflow']['tracking_uri']}")

def main(models):
    """Computes the distance between scattering representations of an image 
        and transformed versions of the same image
        
    """

    # we'll use the parameters.yml of the first model to generate our dataloader
    _, params = get_context(os.path.join(models[0],'parameters.yml'), True) 
    
    # set the batch sized to 1 since we only one image at the time
    params['dataset']['test_batch_size'] =1
    params['dataset']['train_batch_size'] =1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader= get_loaders(params, use_cuda)

    it = (iter(test_loader))  
    img, _ = next(it)
    img.to(device)  
    
    model_values = []
    for model in models:
        model_values.append(get_l2norm_deformation( model, train_loader, img, device, num_data = 20))
    
    figures = visualize_distances(model_values,len(model_values[0]["x"]))
    log_mlflow(params, model_values, figures, img)

if __name__ == '__main__':
    num_arguments = len(sys.argv)
    print("Total paths passed:", num_arguments-1)
    models = []
    for i in range(1, num_arguments):
        models.append(sys.argv[i])
    main(models)

"""Contains all the funcitons related to visualizing the different scattering filters

Functions:  
    get_filters_visualization -- Visualizes the scattering filters input for different modes
    getOneFilter              -- Function used to visualize one filter
    getAllFilters             -- Function used to concatenate filters, creating frames for a video

"""

import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def get_filters_visualization(psi, J, L, mode ='fourier'):
    """ Visualizes the scattering filters input for different modes

    parameters:
        psi  --  dictionnary that contains all the wavelet filters
        L    --  number of orientation
        J    --  scattering scale
        mode -- mode between fourier (in fourier space), real (real part) 
                or imag (imaginary part)
    returns:
        f -- figure/plot
    """
    n_filters =0
    for j in range(2, J+1):
        n_filters+=  j* L
    num_rows = int(n_filters/L) 
    num_col = L
    f, axarr = plt.subplots(num_rows, num_col, figsize=(20, 2*num_rows))
    start_row = 0
    for scale in range(J-1):
        count = L * scale
        end_row = (J-scale) + start_row 
        for i in range(start_row, end_row) :
            for j in range(0, L) :
                if mode =='fourier':
                    x = np.fft.fftshift(psi[count][scale].squeeze().cpu().detach().numpy()).real
                elif mode == 'real':
                    x = np.fft.fftshift(np.fft.ifft2(psi[count][scale].squeeze().cpu().detach().numpy())).real
                elif mode == 'imag':
                    x = np.fft.fftshift(np.fft.ifft2(psi[count][scale].squeeze().cpu().detach().numpy())).imag
                else:
                    raise NotImplemented(f"Model {params['name']} not implemented")
                
                a = np.abs(x).max()
                axarr[i,j].imshow(x, vmin=-a, vmax=a)
                axarr[i,j].set_title(f"J:{psi[count]['j']} L: {psi[count]['theta']}, S:{scale} ")
                axarr[i,j].axis('off')
                count = count +1
                axarr[i,j].set_xticklabels([])
                axarr[i,j].set_yticklabels([])
                axarr[i,j].set_aspect('equal')
        start_row = end_row

    f.subplots_adjust(wspace=0, hspace=0.2)
    return f


def getOneFilter(psi, count, scale, mode):
    """ Methdod used to visualize one filter

    parameters:
        psi   --  dictionnary that contains all the wavelet filters
        count --  key to identify one wavelet filter in the psi dictionnary
        scale --  scattering scale
        mode  --  mode between fourier (in fourier space), real (real part) 
                  or imag (imaginary part)
    returns:
        f -- figure/plot
    """
    if mode =='fourier':
        x = np.fft.fftshift(psi[count][scale].squeeze().cpu().detach().numpy()).real
    elif mode == 'real':
        x = np.fft.fftshift(np.fft.ifft2(psi[count][scale].squeeze().cpu().detach().numpy())).real
    elif mode == 'imag':
        x = np.fft.fftshift(np.fft.ifft2(psi[count][scale].squeeze().cpu().detach().numpy())).imag
    else:
        raise NotImplemented(f"Model {params['name']} not implemented")

    a = np.abs(x).max()
    temp = np.array((x+a)/(a/255), dtype=np.uint8)
    return np.stack([temp for x in range(3)],axis=2)


def getAllFilters(psi, totalCount, scale, mode):
    rows = []
    tempRow = None
    for count in range(totalCount):
        if count % 4 == 0:
            if type(tempRow) != np.ndarray:
                tempRow = getOneFilter(psi, count, scale, mode)
            else:
                rows.append(tempRow)
                tempRow = getOneFilter(psi, count, scale, mode)
        else:
            tempRow = np.concatenate([tempRow, getOneFilter(psi, count, scale, mode)], axis=1)

    rows.append(tempRow)


    temp = np.concatenate(rows, axis=0)
    return temp





def getAngleDistance(one,two):
    """returns the angle of arc between two points on the unit circle"""
    if one < 0 or (2 * np.pi) < one or two < 0 or (2 * np.pi) < two:
        raise Exception

    if one == two:
        return 0
    elif one < two:
        diff = min(
            two - one,
            one + (2 * np.pi) - two
        )
    elif two < one:
        diff = min(
            one - two,
            two + (2 * np.pi) - one
        )
    return diff


def compareParams(params1,angles1,params2,angles2,device):
    """Method to checking the minimal distance between initialized filters and learned ones
    
    Euclidean distances are calculated between each filter for parameters other than orientations
    for orientations, we calculate the arc between both points on the unit circle. Then, the sum of
    these two distances becomes the distance between two filters. Finally, we use munkre's assignment 
    algorithm to compute the optimal match (I.E. the one that minizes total distance)   

    parameters:
        params1 -- first set of parameters compared
        angles1 -- angles associated to the first set of parameters
        params2 -- second set of parameters compared
        angles2 -- angles associated to the second set of parameters
        device  -- torch device to use

    return: 
        minimal distance
    """
    with torch.no_grad():
        groupDistances = torch.cdist(params1,params2)
        angleDistances = torch.zeros(groupDistances.shape, device=device)
        avoidZero = torch.zeros(groupDistances.shape, device=device) + 0.0000000001

        for i in range(angleDistances.size(0)):
            for j in range(angleDistances.size(1)):
                angleDistances[i,j] = getAngleDistance(angles1[i],angles2[j])

        distances = groupDistances + angleDistances + avoidZero

        distNumpy = distances.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(distNumpy, maximize=False)


        return distNumpy[row_ind, col_ind].sum()


def vizMatches(params1,angles1,params2,angles2,row_ind,col_ind,device):
    """ Method used to visualize matches from the assignment algorithm"""
    def vizWavelet(wavelet,mode):
        if mode =='fourier':
            x = np.fft.fftshift(wavelet.squeeze().cpu().detach().numpy()).real
        elif mode == 'real':
            x = np.fft.fftshift(np.fft.ifft2(wavelet.squeeze().cpu().detach().numpy())).real
        elif mode == 'imag':
            x = np.fft.fftshift(np.fft.ifft2(wavelet.squeeze().cpu().detach().numpy())).imag
        else:
            raise NotImplemented(f"Model {params['name']} not implemented")

        return x


    mode = 'fourier'
    num_col = 8
    num_rows = int(len(angles1)/num_col) * 2

    # print(num_rows, num_col)
    
    f, axarr = plt.subplots(num_rows, num_col, figsize=(20, 2*num_rows))
    start_row = 0
    for indx in range(len(row_ind)):
        i = int(indx/num_col) * 2 
        j = indx % num_col

        # print("i:{}j:{}".format(i,j))

        tempWavelet = create_filter(
            theta=angles1[row_ind[indx]].cpu().item(),
            slant=params1[row_ind[indx],2].cpu().item(),
            xi=params1[row_ind[indx],0].cpu().item(),
            sigma=params1[row_ind[indx],1].cpu().item()
        )
        x = vizWavelet(mode=mode, wavelet=tempWavelet)
        a = np.abs(x).max()
        axarr[i,j].imshow(x, vmin=-a, vmax=a)
        axarr[i,j].set_title(f"Match {indx} Optimized")
        axarr[i,j].axis('off')
        axarr[i,j].set_xticklabels([])
        axarr[i,j].set_yticklabels([])
        axarr[i,j].set_aspect('equal')

        tempWavelet = create_filter(
            theta=angles2[col_ind[indx]].cpu().item(),
            slant=params2[col_ind[indx],2].cpu().item(),
            xi=params2[col_ind[indx],0].cpu().item(),
            sigma=params2[col_ind[indx],1].cpu().item()
        )
        x = vizWavelet(mode=mode, wavelet=tempWavelet)
        i+=1
        # print("i:{}j:{}".format(i,j))
        a = np.abs(x).max()
        axarr[i,j].imshow(x, vmin=-a, vmax=a)
        axarr[i,j].set_title(f"Match {indx} Fixed")
        axarr[i,j].axis('off')
        axarr[i,j].set_xticklabels([])
        axarr[i,j].set_yticklabels([])
        axarr[i,j].set_aspect('equal')

    f.subplots_adjust(wspace=0, hspace=0.2)
    return f




from parametricSN.models.create_filters import morlets

def create_filter(theta,slant,xi,sigma):
    n_filters = 1
    device = None

    orientations = np.array([theta]) 
    slants = np.array([slant])
    xis = np.array([xi])
    sigmas = np.array([sigma])

    xis = torch.tensor(xis, dtype=torch.float32, device=device)
    sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)
    slants = torch.tensor(slants, dtype=torch.float32, device=device)
    orientations = torch.tensor(orientations, dtype=torch.float32, device=device)  

    shape = (32, 32,)
    ranges = [torch.arange(-(s // 2), -(s // 2) + s, device=device, dtype=torch.float) for s in shape]
    grid = torch.stack(torch.meshgrid(*ranges), 0)
    wavelet = morlets(grid, orientations, xis, sigmas, slants, device=device)
    
    return wavelet




def compareParamsVisualization(params1,angles1,params2,angles2,device):
    """Method to checking the minimal distance between initialized filters and learned ones
    
    Euclidean distances are calculated between each filter for parameters other than orientations
    for orientations, we calculate the arc between both points on the unit circle. Then, the sum of
    these two distances becomes the distance between two filters. Finally, we use munkre's assignment 
    algorithm to compute the optimal match (I.E. the one that minizes total distance)   

    parameters:
        params1 -- first set of parameters compared
        angles1 -- angles associated to the first set of parameters
        params2 -- second set of parameters compared
        angles2 -- angles associated to the second set of parameters
        device  -- torch device to use

    return: 
        minimal distance
    """
    with torch.no_grad():
        groupDistances = torch.cdist(params1,params2)
        angleDistances = torch.zeros(groupDistances.shape, device=device)
        avoidZero = torch.zeros(groupDistances.shape, device=device) + 0.0000000001

        for i in range(angleDistances.size(0)):
            for j in range(angleDistances.size(1)):
                angleDistances[i,j] = getAngleDistance(angles1[i],angles2[j])

        distances = groupDistances + angleDistances + avoidZero

        distNumpy = distances.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(distNumpy, maximize=False)

        return vizMatches(params1,angles1,params2,angles2,row_ind,col_ind,device)
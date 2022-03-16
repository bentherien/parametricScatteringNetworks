import cv2
import matplotlib.pyplot as plt
import torch  #TODO REMOVE
import numpy as np  #TODO REMOVE
import cmapy

from parametricSN.models.create_filters import morlets, update_psi, create_filters_params
from .visualization_utils import get_filters_visualization, getOneFilter, getAllFilters, compareParams, compareParamsVisualization
#TODO move to utils
def toNumpy(x):
    return x.clone().cpu().numpy()

def getValue(x):
    return toNumpy(x.detach())

def getGrad(x):
    return toNumpy(x.grad)


class filterVisualizer(object):
    def __init__(self, scat, seed):
        super(filterVisualizer, self).__init__()
        self.epoch = 0


        def updateFiltersVideo_hook(scattering, ip):
            """if were using learnable scattering, update the filters to reflect 
            the new parameter values obtained from gradient descent"""
            if (scattering.training or scattering.scatteringTrain):
                if scattering.learnable:
                    if not scattering.pixelwise:
                        wavelets = morlets(scattering.grid, 
                                            scattering.scattering_params_0, 
                                            scattering.scattering_params_1,
                                            scattering.scattering_params_2, 
                                            scattering.scattering_params_3
                                            )
                    else:
                        wavelets = scattering.scattering_wavelets
                    _, psi = scattering.load_filters()
                    scattering.psi = update_psi(scattering.J, psi, wavelets)
                    scattering.register_filters()

                # define hooks
                    self.writeVideoFrame()
                # define hooks
                scattering.scatteringTrain = scattering.training
        scat.pres_hook = scat.register_forward_pre_hook(updateFiltersVideo_hook)

        def recordFilterValues_hook(scattering, ip):
            if scattering.training:
                self.saveFilterValues()
        scat.pre_filter_value_hook = scat.register_forward_pre_hook(recordFilterValues_hook)

        def print_hook(name):
            def recordFilterGrad(grad):
                self.filterGradTracker[name].append(toNumpy(grad))
            return recordFilterGrad
        if scat.scattering_params_0.requires_grad: 
            scat.scattering_params_0.register_hook(print_hook('angle'))
            scat.scattering_params_1.register_hook(print_hook('1'))
            scat.scattering_params_2.register_hook(print_hook('2'))
            scat.scattering_params_3.register_hook(print_hook('3'))

        compared_params = create_filters_params(scat.J, scat.L, scat.learnable,
                scat.equivariant) #kymatio init

        #TODO turn into util function
        self.compared_params_grouped = torch.cat([x.unsqueeze(1) for x in compared_params[1:]], dim=1)
        self.compared_params_angle = compared_params[0] % (2 * np.pi)

        self.params_history = []

        self.scattering = scat

        #TODO turn into util function
        self.videoWriters = {}
        self.videoWriters['real'] = cv2.VideoWriter('videos/scatteringFilterProgressionReal{}epochs.avi'.format("--"),
                                          cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)
        self.videoWriters['imag'] = cv2.VideoWriter('videos/scatteringFilterProgressionImag{}epochs.avi'.format("--"),
                                          cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)
        self.videoWriters['fourier'] = cv2.VideoWriter('videos/scatteringFilterProgressionFourier{}epochs.avi'.format("--"),
                                             cv2.VideoWriter_fourcc(*'DIVX'), 30, (160,160), isColor=True)
        self.videoWriter_lp = cv2.VideoWriter('videos/scattering_{}_LP{}epochs.avi'.format(seed, "--"),
                                          cv2.VideoWriter_fourcc(*'DIVX'), 30,
                                          (40,40), isColor=True)
      
        # visualization code
        self.filterTracker = {'1':[],'2':[],'3':[], 'scale':[], 'angle': []}
        self.filterGradTracker = {'angle': [],'1':[],'2':[],'3':[]}

        # take scattering as object
        self.filters_plots_before = self.getFilterViz()

    # dekha means visualize/look/see in Punjabi
    def littlewood_paley_dekha(self):
      wavelets = morlets(self.scattering.grid, 
                                        self.scattering.scattering_params_0, 
                                        self.scattering.scattering_params_1,
                                        self.scattering.scattering_params_2, 
                                        self.scattering.scattering_params_3).cpu().detach().numpy()
      lp = (np.abs(wavelets) ** 2).sum(0)
      fig = plt.figure()
      ax = plt.subplot()
      ax.imshow(np.fft.fftshift(lp))
      return fig

    def littlewood_paley_lollywood(self):
      wavelets = morlets(self.scattering.grid, self.scattering.scattering_params_0, 
                                        self.scattering.scattering_params_1,
                                        self.scattering.scattering_params_2, 
                                        self.scattering.scattering_params_3).cpu().detach().numpy()
      lp = (np.abs(wavelets) ** 2).sum(0)
      return np.fft.fftshift(lp)
                   
    # move to video recorder class
    def getOneFilter(self, count, scale, mode):
        phi, psi = self.scattering.load_filters()
        return getOneFilter(psi, count, scale, mode)

    # move to video recorder class
    def getAllFilters(self, totalCount, scale, mode):
        phi, psi = self.scattering.load_filters()
        return getAllFilters(psi, totalCount, scale, mode)

    # move to video recorder class
    def writeVideoFrame(self):
        """Writes frames to the appropriate video writer objects"""
        for vizType in self.videoWriters.keys():

            #TODO turn into util function
            temp = cv2.applyColorMap(np.array(self.getAllFilters(totalCount=16, scale=0, mode=vizType),dtype=np.uint8),cv2.COLORMAP_TURBO)
            temp = cv2.putText(temp, "Epoch {}".format(self.epoch),(2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.videoWriters[vizType].write(temp)
        lp = self.littlewood_paley_lollywood()
        lp = lp - lp.min()
        lp = lp / lp.max() * 255
        temp = cv2.applyColorMap(np.uint8(lp), cmapy.cmap('viridis'))
        #temp = cv2.putText(temp, "Epoch {}".format(self.epoch),(0, 4),
        #        cv2.FONT_HERSHEY_SIMPLEX, 0.10, (255, 255, 255), 1)
        self.videoWriter_lp.write(temp)

    # move to video recorder class
    def releaseVideoWriters(self):
        self.videoWriter_lp.release()
        for vizType in self.videoWriters.keys():
            self.videoWriters[vizType].release()

    # move to video recorder class
    def setEpoch(self, epoch):
        self.epoch = epoch

    def getFilterViz(self):
        """generates plots of the filters for ['fourier','real', 'imag' ] visualizations"""
        phi, psi = self.scattering.load_filters()
        filter_viz = {}
        for mode in ['fourier','real', 'imag' ]: # visualize wavlet filters before training
            f = get_filters_visualization(psi, self.scattering.J, 8, mode=mode) 
            filter_viz[mode] = f  
        return filter_viz

    def checkParamDistance(self):
        """Method to checking the minimal distance between initialized filters and learned ones
        
        Euclidean distances are calculated between each filter for parameters other than orientations
        for orientations, we calculate the arc between both points on the unit circle. Then, the sum of
        these two distances becomes the distance between two filters. Finally, we use munkre's assignment 
        algorithm to compute the optimal match (I.E. the one that minizes total distance)        

        return: 
            minimal distance
        """
        #TODO turn into util function
        tempParamsGrouped = torch.cat([x.unsqueeze(1) for x in
            [self.scattering.scattering_params_1, self.scattering.scattering_params_2, self.scattering.scattering_params_3]], dim=1).cpu()
        tempParamsAngle = (self.scattering.scattering_params_0 % (2 * np.pi)).cpu()
        self.params_history.append({'params':tempParamsGrouped, 'angle':tempParamsAngle})
        return compareParams(
            params1=tempParamsGrouped,
            angles1=tempParamsAngle, 
            params2=self.compared_params_grouped,
            angles2=self.compared_params_angle
        )

    def compareParamsVisualization(self):
        """visualize the matched filters"""
        #TODO turn into util function
        tempParamsGrouped = torch.cat([x.unsqueeze(1) for x in
            [self.scattering.scattering_params_1, self.scattering.scattering_params_2, self.scattering.scattering_params_3]], dim=1).cpu()
        tempParamsAngle = (self.scattering.scattering_params_0 % (2 * np.pi)).cpu()
        self.params_history.append({'params':tempParamsGrouped, 'angle':tempParamsAngle})
        return compareParamsVisualization(
            params1=tempParamsGrouped,
            angles1=tempParamsAngle, 
            params2=self.compared_params_grouped,
            angles2=self.compared_params_angle
        )

    def saveFilterValues(self):
        #TODO turn into util function
        self.filterTracker['angle'].append(getValue(self.scattering.scattering_params_0))
        self.filterTracker['1'].append(getValue(self.scattering.scattering_params_1))
        self.filterTracker['2'].append(getValue(self.scattering.scattering_params_2))
        self.filterTracker['3'].append(getValue(self.scattering.scattering_params_3))
        self.filterTracker['scale'].append(np.multiply(self.filterTracker['1'][-1], self.filterTracker['2'][-1]))

    def saveFilterGrads(self,scatteringActive):
        print("shooooooooo")
        self.filterGradTracker['angle'].append(getGrad(self.scattering.params_filters[0]))
        self.filterGradTracker['1'].append(getGrad(self.scattering.params_filters[1]))
        self.filterGradTracker['2'].append(getGrad(self.scattering.params_filters[2]))
        self.filterGradTracker['3'].append(getGrad(self.scattering.params_filters[3]))

    def get_param_grad_per_epoch(self, x):
        return {
                    'orientation1': [float(filters[x]) for filters in self.filterGradTracker['angle']],
                    'xis': [float(filters[x])  for filters in self.filterGradTracker['1']],
                    'sigmas': [float(filters[x])  for filters in self.filterGradTracker['2']],
                    'slant': [float(filters[x])  for filters in self.filterGradTracker['3']],
                }

    def get_param_per_epoch(self, x):
        return {
                    'orientation1': [float(filters[x]) for filters in self.filterTracker['angle']],
                    'xis': [float(filters[x])  for filters in self.filterTracker['1']],
                    'sigmas': [float(filters[x])  for filters in self.filterTracker['2']],
                    'slant': [float(filters[x])  for filters in self.filterTracker['3']],
                    'scale': [float(filters[x])  for filters in self.filterTracker['scale']],
                }

    def plotFilterGrads(self):
        """plots the graph of the filter gradients"""
        paramsNum = self.scattering.scattering_params_0.shape[0]
        if self.scattering.equivariant:
            col =  paramsNum
            row = 1
            size = (80, 10)
            f, axarr = plt.subplots(row, col, figsize=size) # create plots
            for x in range(paramsNum):
                temp= self.get_param_grad_per_epoch(x)
                axarr[x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='orientation1')
                axarr[x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
                axarr[x%col].plot([x  for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
                axarr[x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
                axarr[x%col].legend()
        
        else:
            col = 8
            row = int(paramsNum/col)
            size = (80, 10*row,)

            f, axarr = plt.subplots(row, col, figsize=size) # create plots

            for x in range(paramsNum):#iterate over all the filters
                temp= self.get_param_grad_per_epoch(x)
                axarr[int(x/col),x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='orientation1')
                axarr[int(x/col),x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
                axarr[int(x/col),x%col].plot([x  for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
                axarr[int(x/col),x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
                axarr[int(x/col),x%col].legend()
        return f
    
    def plotFilterValues(self):
        """plots the graph of the filter values"""
        paramsNum = self.scattering.scattering_params_0.shape[0]
        if self.scattering.equivariant:
            col = paramsNum
            row = 1
            size = (80, 10)
            f, axarr = plt.subplots(row, col, figsize=size) # create plots
            for x in range(paramsNum):
                temp = self.get_param_per_epoch(x)
                axarr[x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='theta')
                axarr[x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
                axarr[x%col].plot([x for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
                axarr[x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
                axarr[x%col].plot([x for x in range(len(temp['scale']))],temp['scale'],color='black', label='scale')
                axarr[x%col].legend()

        else:
            col = 8
            row = int(self.scattering.filterNum/col)
            size = (80, 10*row,)
            f, axarr = plt.subplots(row, col, figsize=size) # create plots

            for x in range(paramsNum):
                temp = self.get_param_per_epoch(x)
                axarr[int(x/col),x%col].plot([x for x in range(len(temp['orientation1']))],temp['orientation1'],color='red', label='theta')
                axarr[int(x/col),x%col].plot([x for x in range(len(temp['xis']))],temp['xis'],color='green', label='xis')
                axarr[int(x/col),x%col].plot([x for x in range(len(temp['sigmas']))],temp['sigmas'],color='yellow', label='sigma')
                axarr[int(x/col),x%col].plot([x for x in range(len(temp['slant']))],temp['slant'],color='orange', label='slant')
                axarr[int(x/col),x%col].plot([x for x in range(len(temp['scale']))],temp['scale'],color='black', label='scale')
                axarr[int(x/col),x%col].legend()
        return f

    def plotParameterValues(self):
        size = (10, 10)
        f, axarr = plt.subplots(2, 2, figsize=size) # create plots
        plt.subplots_adjust(hspace=0.35, wspace=0.35)
        label = ['theta', 'xis', 'sigma', 'slant']
        for idx,param in enumerate(['angle', "1", '2', '3']):#iterate over all the parameters
            for idx2, filter in enumerate(np.stack(self.filterTracker[param]).T):
                axarr[int(idx/2), idx%2].plot([x for x in range(len(filter))], filter)
            axarr[int(idx/2), idx%2].set_title(label[idx], fontsize=16)
            axarr[int(idx/2), idx%2].set_xlabel('Epoch', fontsize=12) # Or ITERATION to be more precise
            axarr[int(idx/2), idx%2].set_ylabel('Value', fontsize=12)
        return f

"""Contains all the base pytorch NN.modules for this project

Authors: Benjamin Therien, Shanel Gauthier

Functions: 
    create_scatteringExclusive -- creates scattering parameters

Exceptions:
    InvalidInitializationException -- Error thrown when an invalid initialization scheme is passed

Classes: 
    sn_Identity -- computes the identity function in forward pass
    sn_HybridModel -- combinations of a scattering and other nn.modules
    sn_ScatteringBase -- a scattering network
"""
import types

import torch
import torch.nn as nn
from kymatio.torch import Scattering2D

from .create_filters import morlets, update_psi, create_filters_params_random, create_filters_params, update_wavelets_psi


class InvalidInitializationException(Exception):
    """Error thrown when an invalid initialization scheme is passed"""
    pass

class InvalidParameterizationException(Exception):
    """Error thrown when an invalid parameterization scheme is passed"""
    pass





    
class sn_Identity(nn.Module):
    """Identity nn.Module for identity"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n_coefficients = 1

    def forward(self, x):
        return x
        

def _register_single_filter(self, v, n):
    self.register_buffer('tensor' + str(n), v)

class sn_ScatteringBase(Scattering2D):
    """A learnable scattering nn.module 

    parameters:
        learnable -- should the filters be learnable parameters of this model
        J -- scale of scattering (always 2 for now)
        N -- height of the input image
        M -- width of the input image
        initilization -- the type of init: ['Tight-Frame' or 'Random']
        seed -- the random seed used to initialize the parameters
    """

    def __str__(self):
        tempL = " L" if self.learnable else "NL"
        tempI = "TF" if self.initialization == 'Tight-Frame' else "R"
        return f"{tempI} {tempL}"


    def __init__(self, J, N, M, second_order, initialization, seed, 
                 learnable=True, lr_orientation=0.1, 
                 lr_scattering=0.1, monitor_filters=True,
                 filter_video=False, parameterization='canonical'):
        """Constructor for the leanable scattering nn.Module
        
        Creates scattering filters and adds them to the nn.parameters if learnable
        
        parameters: 
            J -- scale of scattering (always 2 for now)
            N -- height of the input image
            M -- width of the input image
            second_order -- 
            initilization -- the type of init: ['Tight-Frame' or 'Random']
            seed -- the random seed used to initialize the parameters
            learnable -- should the filters be learnable parameters of this model
            lr_orientation -- learning rate for the orientation of the scattering parameters
            lr_scattering -- learning rate for scattering parameters other than orientation
        """
        super(sn_ScatteringBase, self).__init__(J=J, shape=(M, N))

        self.second_order = second_order
        self.learnable = learnable
        self.initialization = initialization
        self.lr_scattering = lr_scattering
        self.lr_orientation = lr_orientation
        self.M_coefficient = self.M/(2**self.J)
        self.N_coefficient = self.N/(2**self.J)
        self.parameterization = parameterization
        self.scatteringTrain = True

        L = self.L

        if second_order:
            self.n_coefficients =  L*L*J*(J-1)//2
        else: 
            self.n_coefficients =  L*L*J*(J-1)//2 + 1 + L*J  
        
        self.set_parameterization()
        
        if initialization == "Tight-Frame":
            self.params_filters = create_filters_params(J, L, learnable, self.equivariant) #kymatio init
        elif initialization == "Random":
            self.params_filters = create_filters_params_random(J,L, learnable, self.equivariant) #random init
        else:
            raise InvalidInitializationException
            

        shape = (self.M_padded, self.N_padded,)
        ranges = [torch.arange(-(s // 2), -(s // 2) + s, dtype=torch.float) for s in shape]
        grid = torch.stack(torch.meshgrid(*ranges), 0)

        self.psi , wavelets =   update_wavelets_psi(J, L, self.psi, shape, self.params_filters, self.equivariant)
        self.filterNum = wavelets.shape[2]

        self.register_single_filter = types.MethodType(_register_single_filter, self)
        self.register_filters()
        if learnable:
            if self.pixelwise:
                self.register_parameter(name='scattering_wavelets',
                        param=nn.Parameter(wavelets))
                phi, psi = self.load_filters()

                self.psi = update_psi(self.J, self.psi, self.scattering_wavelets)
                
                for i in range(0, len(self.params_filters)):
                    self.register_buffer(name='scattering_params_'+str(i), tensor=self.params_filters[i])
            else:
                for i in range(0, len(self.params_filters)):
                    self.params_filters[i] = nn.Parameter(self.params_filters[i])
                    self.register_parameter(name='scattering_params_'+str(i), param=self.params_filters[i])
                self.params_filters = nn.ParameterList(self.params_filters)
        else:
            for i in range(0, len(self.params_filters)):
                self.register_buffer(name='scattering_params_'+str(i), tensor=self.params_filters[i])
            buffer_dict = dict(self.named_buffers())
            
            for i in range(0, len(self.params_filters)):
                self.params_filters[i] = buffer_dict['scattering_params_'+str(i)]


        self.register_buffer(name='grid', tensor=grid)


        
        def updateFilters_hook(self, ip):
            """if were using learnable scattering, update the filters to reflect 
            the new parameter values obtained from gradient descent"""
            if (self.training or self.scatteringTrain) and self.learnable:
                _, psi = self.load_filters()
                if not self.pixelwise:
                    wavelets = morlets(self.grid, 
                            self.scattering_params_0,
                            self.scattering_params_1,
                            self.scattering_params_2, 
                            self.scattering_params_3)
                else:
                    wavelets = self.scattering_wavelets

                self.psi = update_psi(self.J, psi, wavelets)
                self.register_filters()
                self.scatteringTrain = self.training
        self.pre_hook = self.register_forward_pre_hook(updateFilters_hook)

        def reshape_hook(self, x, S):
            S = S[:,:, -self.n_coefficients:,:,:]
            S = S.reshape(S.size(0), self.n_coefficients*3, S.size(3), S.size(4))
            return S

        self.register_forward_hook(reshape_hook)


    def set_parameterization(self):
        '''
        Set the parameterization of the scattering network
        '''
        if self.parameterization =='canonical':
            self.pixelwise = False
            self.equivariant = False
        elif self.parameterization =='pixelwise':
            self.pixelwise = True
            self.equivariant = False
        elif self.parameterization =='equivariant':
            self.pixelwise = False
            self.equivariant = True
        else:
            raise InvalidParameterizationException

"""Contains all the hybrid pytorch nn.modules for this project

Authors: Benjamin Therien, Shanel Gauthier

Classes: 
    sn_HybridModel -- combinations of a scattering and other nn.modules
"""

import torch.nn as nn

class sn_HybridModel(nn.Module):
    """An nn.Module combining two nn.Modules 
    
    This hybrid model was created to connect a scattering model to another
    nn.Module, but can also combine any other two modules. 
    """

    def __str__(self):
        return str(self.scatteringBase)

    def __init__(self, scatteringBase, top):
        """Constructor for a HybridModel

        scatteringBase -- the scattering nn.Module
        top -- the nn.Module used after scatteringBase
        """
        super(sn_HybridModel,self).__init__()
        self.scatteringBase = scatteringBase
        self.top = top

    def forward(self,inp):
        return self.top(self.scatteringBase(inp))

    def showParams(self):
        """prints shape of all parameters and is_leaf"""
        for x in self.parameters():
            if type(x['params']) == list:
                for tens in x['params']:
                    print(tens.shape,tens.is_leaf)
            else:
                print(x['params'].shape,x['params'].is_leaf)


import torch

def optimizerFactory(hybridModel, params):
    """Factory for adam, sgd

    parameters: 
        params -- dict of input parameters
        hybridModel -- the model used during training 
    """

    if params['optim']['name'] == 'adam':
        return torch.optim.Adam(
            hybridModel.parameters(),lr=params['optim']['lr'], 
            betas=(0.9, 0.999), eps=1e-08, 
            weight_decay=params['optim']['weight_decay'], amsgrad=False
        )
    elif params['optim']['name'] == 'sgd': 
        return torch.optim.SGD(
            hybridModel.parameters(), lr=params['optim']['lr'], 
            momentum=params['optim']['momentum'], weight_decay=params['optim']['weight_decay']
        )
        
    else:
        raise NotImplemented(f"Optimizer {params['optim']['name']} not implemented")


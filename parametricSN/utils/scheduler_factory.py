"""Factory for scheduler loading

Authors: Benjamin Therien, Shanel Gauthier

functions: 
    schedulerFactory -- Factory for OneCycle, CosineAnnealing, Lambda, Cyclic, and step schedulers
"""

import torch

def schedulerFactory(optimizer, params, steps_per_epoch):
    """Factory for OneCycle, CosineAnnealing, Lambda, Cyclic, and step schedulers

    parameters: 
        params -- dict of input parameters
        optimizer -- the optimizer paired with the scheduler
        steps_per_epoch -- number of steps the scheduler takes each epoch
    """

    if params['optim']['scheduler'] =='OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=params['optim']['max_lr'], 
            steps_per_epoch=steps_per_epoch, epochs=params['model']['epoch'], 
            three_phase=params['optim']['three_phase'],
            div_factor=params['optim']['div_factor']
        )

        for group in optimizer.param_groups:
            if 'maxi_lr' in group .keys():
                group['max_lr'] = group['maxi_lr']

    elif params['optim']['scheduler'] =='CosineAnnealingLR':
        scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = params['optim']['T_max'], eta_min = 1e-8)

    elif params['optim']['scheduler'] =='LambdaLR':
        lmbda = lambda epoch: 0.95
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

    elif params['optim']['scheduler'] =='CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, 
                                            step_size_up=params['optim']['T_max']*2,
                                             mode="triangular2")

    elif params['optim']['scheduler'] =='StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=steps_per_epoch * int(params['model']['epoch']/2), 
                                                    gamma=0.5)

    elif params['optim']['scheduler'] == 'NoScheduler':
        scheduler = None

    else:
        raise NotImplemented(f"Scheduler {params['optim']['scheduler']} not implemented")

    return scheduler
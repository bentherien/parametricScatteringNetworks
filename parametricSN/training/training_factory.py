""" Factory for all training methods"""

from . import cosine_training, cross_entropy_training, cross_entropy_training_accumulation


def train_test_factory(loss_name):
    """ Factory for different train and test Functions

    parameters: 
        loss_name -- the name of the loss function to use
    """
            
    if loss_name == 'cross-entropy':
        train = lambda *args, **kwargs : cross_entropy_training.train(*args,**kwargs)
        test = lambda *args : cross_entropy_training.test(*args)

    elif loss_name == 'cross-entropy-accum':
        train = lambda *args, **kwargs : cross_entropy_training_accumulation.train(*args,**kwargs)
        test = lambda *args : cross_entropy_training_accumulation.test(*args)    
    
    elif loss_name == 'cosine':
        train = lambda *args, **kwargs : cosine_training.train(*args, **kwargs)
        test = lambda *args : cosine_training.test(*args)

    else:
        raise NotImplemented(f"Loss {loss_name} not implemented")
    
    return train, test
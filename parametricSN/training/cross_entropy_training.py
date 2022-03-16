"""Standard Cross entropy training 

Functions: 
    train -- training function 
    test -- testing function
"""

import torch

import torch.nn.functional as F
from sklearn.metrics import classification_report

def test(model, device, test_loader):
    """test method"""
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.long)  
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: [Model -- {}] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        model, test_loss, correct, len(test_loader.dataset),accuracy ))
    return accuracy, test_loss

def train(model, device, train_loader, scheduler, optimizer, epoch, accum_step_multiple=None):
    """training method"""
    model.train()
    correct = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype=torch.long)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if scheduler != None:
            scheduler.step()

        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probabilityd
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
    
    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    
    print('[Model -- {}] Train Epoch: {:>6} Average Loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(
            model, epoch, train_loss, correct, 
            len(train_loader.dataset),train_accuracy
            )
        )

    return train_loss, train_accuracy

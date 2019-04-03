'''Implement the Mean Teacher algorithm.'''
import time
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler
import numpy as np
from utils import data as data_lib
from utils import utils, losses


def detach_model_params(model):
    '''Detach parameters from the model, so that pytorch won't accumulate computation history for them.
    This is used in creating the EMA teacher model.

    WARNING: when creating the EMA teacher model, the model should be a different instance from the student model,
    otherwise this function would also detach parameters from the student model as well.
    '''
    for param in model.parameters():
        param.detach_()
    return model

# def get_current_consistency_weight(epoch, alpha):
#     # Consistency ramp-up from https://arxiv.org/abs/1610.02242
#     return alpha * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    '''Update the EMA(Exponential Moving Average) model

    Args:
        model: current model.
        ema_model: the EMA model.
        alpha: EMA decay rate.
        global_step
    '''
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(alpha).add_(1 - alpha, param)

def train(args, model, ema_model, device, train_loader, optimizer, epoch, alpha, global_step):
    '''Train the model.'''
    # switch to train mode
    model.train()
    ema_model.train()

    for batch_idx, (data, target, weight) in enumerate(train_loader):
        start = time.time()
        # target = utils.one_hot_encoding(target, num_classes=args.num_classes)
        data, target, weight = data.to(device), target.to(device), weight.to(device)

        optimizer.zero_grad()
        model_out = model(data) # model output logits of the student model
        ema_model_out = ema_model(data) # output logits of the teacher model

        class_loss = F.cross_entropy(model_out[0:args.labeled_batch_size], target[0:args.labeled_batch_size])
        assert not (np.isnan(float(class_loss)) or float(class_loss) > 1e8), 'class_loss explosion: {}'.format(float(class_loss))

        # consistency_weight = get_current_consistency_weight(epoch)
        consistency_loss = alpha * losses.consistency_loss(model_out, ema_model_out, weight, consistency_type=args.consistency_type)
        loss = class_loss + consistency_loss
        assert not (np.isnan(float(loss)) or float(loss) > 1e8), 'Loss explosion: {}'.format(float(loss))
        
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        with torch.no_grad():
            output_softmax = F.softmax(model_out[0:args.labeled_batch_size], dim=1)
            pred = output_softmax.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct = pred.eq(target[0:args.labeled_batch_size].view_as(pred)).sum().item()
            accuracy = correct / len(target[0:args.labeled_batch_size])
            end = time.time()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)],  Loss: {:.6f},  Train_acc: {:.2f}%,  Time: {:.4f}'.format(
                    epoch, batch_idx * (len(data)-args.labeled_batch_size), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), 100. * accuracy, end-start))
    return global_step

def test(args, model, device, test_loader):
    '''Measure the performance of the model.'''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # target = utils.one_hot_encoding(target, num_classes=args.num_classes)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            # test_loss += F.nll_loss_one_hot(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return accuracy
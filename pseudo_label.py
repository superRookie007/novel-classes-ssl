'''Implement the Pseudo-Label algorithm.'''
import time

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler

from utils import data as data_lib
from utils import utils, losses



def train(args, model, device, train_loader, optimizer, epoch, alpha):
    '''Train the model.'''
    model.train()
    for batch_idx, (data, target, weight) in enumerate(train_loader):
        start = time.time()
        target = utils.one_hot_encoding(target, num_classes=args.num_classes)
        data, target, weight = data.to(device), target.to(device), weight.to(device)

        if args.exclude_unlabeled:
            optimizer.zero_grad()
            output = F.log_softmax(model(data), dim=1)
            loss = losses.nll_loss_one_hot(output, target)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct = pred.eq(target.max(1, keepdim=True)[1]).sum().item()
                accuracy = correct / len(target)

                end = time.time()
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)],  Loss: {:.6f},  Train_acc: {:.0f}%,  Time: {:.4f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(), 100. * accuracy, end-start))
        else:
            optimizer.zero_grad()
            output_labelled = F.log_softmax(model(data[0:args.labeled_batch_size]),dim=1)
            output_unlabelled = F.log_softmax(model(data[args.labeled_batch_size: len(data)]), dim=1)
            loss_labelled = losses.nll_loss_one_hot(output_labelled, target[0:args.labeled_batch_size])
            loss_unlabelled = losses.nll_loss_one_hot(output_unlabelled, target[args.labeled_batch_size:len(target)], weights=weight[args.labeled_batch_size:len(weight)])
            loss = loss_labelled + alpha * loss_unlabelled
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred = output_labelled.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct = pred.eq(target[0:args.labeled_batch_size].max(1, keepdim=True)[1]).sum().item()
                accuracy = correct / len(target[0:args.labeled_batch_size])
                end = time.time()
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)],  Loss: {:.6f},  Train_acc: {:.2f}%,  Time: {:.4f}'.format(
                        epoch, batch_idx * (len(data)-args.labeled_batch_size), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(), 100. * accuracy, end-start))


def test(args, model, device, test_loader):
    '''Measure the performance of the model.'''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = utils.one_hot_encoding(target, num_classes=args.num_classes)
            data, target = data.to(device), target.to(device)
            output = F.log_softmax(model(data), dim=1)
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            test_loss += losses.nll_loss_one_hot(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            correct += pred.eq(target.max(1, keepdim=True)[1]).sum().item()


    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return accuracy

def assign_labels(args, model, device, train_dataset, unlabeled_idxs):
    '''Assigning labels to unlabeled data based on current model predictions.'''
    print('Assigning labels...')
    start = time.time()
    sampler = data_lib.SequentialSampler(unlabeled_idxs)
    batch_sampler = BatchSampler(sampler, args.test_batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=1,
        pin_memory=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for data, target, _ in train_loader:
            target = utils.one_hot_encoding(target, num_classes=args.num_classes)
            data, target = data.to(device), target.to(device)
            # output = F.log_softmax(model(data), dim=1)
            pred = F.softmax(model(data), dim=1)
            pred = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
            preds.append(pred)

    preds = torch.cat(preds, 0)
    preds = preds.to('cpu')
    # print(preds)
    train_dataset.assign_labels(preds)
    end = time.time()
    print('\nAssigning labels took: {:.4f} seconds.\n'.format(end-start))
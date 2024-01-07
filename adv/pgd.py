import os
import time 
import torch
import random
import shutil
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from adv.utils import * 

__all__ = ['attack_pgd']

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

# https://github.com/locuslab/robust_overfitting/blob/master/train_cifar.py
def attack_pgd(model, X, y, epsilon, alpha, loss_fn, attack_iters, restarts = 1, early_stop=False):
    
    model.eval()
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_gamma = torch.zeros(y.shape[0]).cuda()
    
    
    #torchvision.transforms.functional.adjust_gamma()
    for _ in range(restarts):
        gamma = torch.zeros(y.shape[0]).cuda()
        
        gamma.uniform_(1-epsilon, 1+epsilon)

        #gamma = clamp(delta, lower_limit-X, upper_limit-X)
        gamma.requires_grad = True
        for _ in range(attack_iters):
            # print(X.size())
            # print(gamma.size())
            attack_img = torch.pow(X,gamma.view(-1,1,1,1))
            output = model(attack_img)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            loss =loss_fn(output, y)
            #print('attack_loss: ', loss)
            
            loss.backward()
            grad = gamma.grad.detach()
            d = gamma[index]
            g = grad[index]
            
            # gradient ascent
            d = torch.clamp(d + alpha * torch.sign(g), min=1-epsilon, max=1+epsilon)
            #print('updated_d: ', d)
            gamma.data = d
            gamma.grad.zero_()

        if restarts > 1:
            all_loss = torch.mean(F.mse_loss(model(torch.pow(X,gamma.view(-1,1,1,1))), y, reduction='none'),dim=[1,2,3])
            #print('all_loss: ', all_loss)
            max_gamma[all_loss >= max_loss] = gamma.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)
        else:
            max_gamma=gamma
        #max_gamma = torch.max(max_loss, all_loss)
        #max_gamma = gamma
        #print(max_gamma)
    return max_gamma.view(-1,1,1,1)

def train_epoch_adv(train_loader, model, criterion, optimizer, epoch, args):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        image = image.cuda()
        target = target.cuda()

        #adv samples
        model.eval() # https://arxiv.org/pdf/2010.00467.pdf
        delta = attack_pgd(model, image, target, args.train_eps, args.train_alpha, args.train_step, args.train_norm)
        delta.detach()
        image_adv = torch.clamp(image + delta[:image.size(0)], 0, 1)
        model.train()

        # compute output
        output_adv = model(image_adv)
        loss = criterion(output_adv, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_adv.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('adversarial train accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def test_adv(val_loader, model, criterion, args):
    """
    Run adversarial evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    start = time.time()
    for i, (image, target) in enumerate(val_loader):

        image = image.cuda()
        target = target.cuda()

        #adv samples
        delta = attack_pgd(model, image, target, args.test_eps, args.test_alpha, args.test_step, args.test_norm)
        delta.detach()
        image_adv = torch.clamp(image + delta[:image.size(0)], 0, 1)

        # compute output
        with torch.no_grad():
            output = model(image_adv)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            end = time.time()
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {2:.2f}'.format(
                    i, len(val_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('Robust Accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

import gc

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import GPUtil

from misc.utils import TruncatedLoss, GCELoss, MAELoss, SCELoss, ELRLoss, ELRLoss_100
import time


def train(model, train_loader, optim, args):
    model.train()
    with torch.autograd.set_detect_anomaly(True):
        data = next(iter(train_loader))
        inputs, labels, indexes = data[0].to(args.device), data[1].to(args.device), data[4].to(args.device)
        # print(inputs.shape, labels.shape)
        # inputs, labels = data[0], data[1]

        optim.zero_grad()

        outputs = model(inputs)
        # print(f'outputs: {outputs.shape}')
        # print(f'labels: {labels.shape}')
        
        # criterion = nn.CrossEntropyLoss(reduction='none')
        if args.loss_type == 'ce':
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = criterion(outputs, labels)
            loss = loss.mean() # explicit mean reduction
        elif args.loss_type == 'mae':
            criterion = nn.L1Loss(reduction='none')
            # labels_onehot = F.one_hot(labels, num_classes=args.num_classes)
            # loss = criterion(F.softmax(outputs, dim=1), labels_onehot)
            # loss = loss.mean() # explicit mean reduction
            criterion = MAELoss()
            loss = criterion(outputs, labels)
            loss = loss.mean()
        elif args.loss_type == 'gce':
            criterion = GCELoss(q=0.7)
            loss = criterion(outputs, labels)
            loss = loss.mean()
        elif args.loss_type == 'sce':
            if 'cifar10' in args.data:
                alpha = 0.1
                beta = 1.0
            elif 'cifar100' in args.data:
                alpha = 6.0
                beta = 0.1
            criterion = SCELoss(alpha=alpha, beta=beta, device=args.device, num_classes=args.num_classes)
            loss = criterion(outputs, labels)
            loss = loss.mean()
        elif args.loss_type == 'elr':
            if args.data == 'cifar10':
                lambda_ = 3
                beta = 0.7
            elif args.data == 'cifar100':
                lambda_ = 7
                beta = 0.9
            criterion = ELRLoss(num_examp=args.trainset_size, num_classes=args.num_classes, lambda_=lambda_, beta=beta)
            loss = criterion(indexes, outputs, labels)
            loss = loss.mean()
        elif args.loss_type == 'elr100':
            if args.data == 'cifar10':
                lambda_ = 3
                beta = 0.7
            elif args.data == 'cifar100':
                lambda_ = 7
                beta = 0.9
            criterion = ELRLoss_100(num_examp=args.trainset_size, num_classes=args.num_classes, lambda_=lambda_, beta=beta)
            loss = criterion(indexes, outputs, labels)
            loss = loss.mean()
            
        elif args.loss_type == 'tgce':
            criterion = TruncatedLoss(k=0, trainset_size=args.trainset_size).to(args.device)
            loss = criterion(outputs, labels, indexes) # implicit mean reduction
            # criterion = GCELoss()
            # loss = criterion(outputs, labels)
            # loss = loss.mean()
        # elif args.loss_type == 'tgce':
            # criterion = TruncatedLoss(k=0.5, trainset_size=args.trainset_size).to(args.device)
            # loss = criterion(outputs, labels, indexes) # implicit mean reduction
            # criterion = GCELoss()
            # loss = criterion(outputs, labels)
            # loss = loss.mean()

        # loss = criterion(outputs, labels)
        # loss = loss.mean() # explicit mean reduction
        loss.backward()
        optim.step()
        # print(labels)
        
        del data; gc.collect()
        torch.cuda.empty_cache()

    return loss.detach().cpu().numpy()

def train_(model, train_loader, optim, args):
    tstart = time.time()
    model.train()
    with torch.autograd.set_detect_anomaly(True):
        data = next(iter(train_loader))
        inputs, labels, indexes = data[0].to(args.device), data[1].to(args.device), data[4].to(args.device)
        # print(inputs.shape, labels.shape)
        # inputs, labels = data[0], data[1]

        optim.zero_grad()

        outputs = model(inputs)
        # print(f'outputs: {outputs.shape}')
        # print(f'labels: {labels.shape}')
        
        # criterion = nn.CrossEntropyLoss(reduction='none')
        if args.loss_type == 'ce':
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = criterion(outputs, labels)
            loss = loss.mean() # explicit mean reduction
        elif args.loss_type == 'mae':
            criterion = nn.L1Loss(reduction='none')
            # labels_onehot = F.one_hot(labels, num_classes=args.num_classes)
            # loss = criterion(F.softmax(outputs, dim=1), labels_onehot)
            # loss = loss.mean() # explicit mean reduction
            criterion = MAELoss()
            loss = criterion(outputs, labels)
            loss = loss.mean()
        elif args.loss_type == 'gce':
            criterion = GCELoss(q=0.7)
            loss = criterion(outputs, labels)
            loss = loss.mean()
        elif args.loss_type == 'sce':
            if args.data == 'cifar10':
                alpha = 0.1
                beta = 1.0
            elif args.data == 'cifar100':
                alpha = 6.0
                beta = 0.1
            criterion = SCELoss(alpha=alpha, beta=beta, device=args.device, num_classes=args.num_classes)
            loss = criterion(outputs, labels)
            loss = loss.mean()
        elif args.loss_type == 'elr':
            if args.data == 'cifar10':
                lambda_ = 3
                beta = 0.7
            elif args.data == 'cifar100':
                lambda_ = 7
                beta = 0.9
            criterion = ELRLoss(num_examp=args.trainset_size, num_classes=args.num_classes, lambda_=lambda_, beta=beta)
            loss = criterion(indexes, outputs, labels)
            loss = loss.mean()
        elif args.loss_type == 'elr100':
            if args.data == 'cifar10':
                lambda_ = 3
                beta = 0.7
            elif args.data == 'cifar100':
                lambda_ = 7
                beta = 0.9
            criterion = ELRLoss_100(num_examp=args.trainset_size, num_classes=args.num_classes, lambda_=lambda_, beta=beta)
            loss = criterion(indexes, outputs, labels)
            loss = loss.mean()
            
        elif args.loss_type == 'tgce':
            criterion = TruncatedLoss(k=0, trainset_size=args.trainset_size).to(args.device)
            loss = criterion(outputs, labels, indexes) # implicit mean reduction
            # criterion = GCELoss()
            # loss = criterion(outputs, labels)
            # loss = loss.mean()
        # elif args.loss_type == 'tgce':
            # criterion = TruncatedLoss(k=0.5, trainset_size=args.trainset_size).to(args.device)
            # loss = criterion(outputs, labels, indexes) # implicit mean reduction
            # criterion = GCELoss()
            # loss = criterion(outputs, labels)
            # loss = loss.mean()

        # loss = criterion(outputs, labels)
        # loss = loss.mean() # explicit mean reduction
        loss.backward()
        optim.step()
        # print(labels)
        
        del data; gc.collect(); torch.cuda.empty_cache()
    tend = time.time()
    print(f'train time: {tend-tstart}')

    return loss.detach().cpu().numpy()

def validate(model, valid_loader, args):
    model.eval()
    
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            inputs, labels, indexes = data[0].to(args.device), data[1].to(args.device), data[4].to(args.device)
            # print(labels.shape, indexes.shape)
            # inputs, labels = data[0], data[1]
            
            outputs = model(inputs)
            
            if args.loss_type == 'ce':
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss_ = criterion(outputs, labels)
                loss_ = loss_.mean() # explicit mean reduction
            elif args.loss_type == 'mae':
                # criterion = nn.L1Loss(reduction='none')
                # labels_onehot = F.one_hot(labels, num_classes=args.num_classes)
                # loss_ = criterion(F.softmax(outputs, dim=1), labels_onehot)
                # loss_ = loss_.mean() # explicit mean reduction
                criterion = MAELoss()
                loss_ = criterion(outputs, labels)
                loss_ = loss_.mean()
            elif args.loss_type == 'gce':
                criterion = GCELoss(q=0.7)
                loss_ = criterion(outputs, labels)
                loss_ = loss_.mean()
            elif args.loss_type == 'sce':
                if 'cifar10' in args.data:
                    alpha = 0.1
                    beta = 1.0
                elif 'cifar100' in args.data:
                    alpha = 6.0
                    beta = 0.1
                criterion = SCELoss(alpha=alpha, beta=beta, device=args.device, num_classes=args.num_classes)
                loss_ = criterion(outputs, labels)
                loss_ = loss_.mean()
            elif args.loss_type == 'elr':
                # if args.data == 'cifar10':
                #     lambda_ = 3
                #     beta = 0.7
                # elif args.data == 'cifar100':
                #     lambda_ = 7
                #     beta = 0.9
                # criterion = ELRLoss(num_examp=args.validset_size, num_classes=args.num_classes, lambda_=lambda_, beta=beta)
                # loss_ = criterion(indexes, outputs, labels)
                # loss_ = loss_.mean()
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss_ = criterion(outputs, labels)
                loss_ = loss_.mean() # explicit mean reduction
            elif args.loss_type == 'elr100':
                # if args.data == 'cifar10':
                #     lambda_ = 3
                #     beta = 0.7
                # elif args.data == 'cifar100':
                #     lambda_ = 7
                #     beta = 0.9
                # criterion = ELRLoss(num_examp=args.validset_size, num_classes=args.num_classes, lambda_=lambda_, beta=beta)
                # loss_ = criterion(indexes, outputs, labels)
                # loss_ = loss_.mean()
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss_ = criterion(outputs, labels)
                loss_ = loss_.mean() # explicit mean reduction
            
            loss += loss_.detach().cpu().numpy()
            
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum()
            
            del data; gc.collect()
            
    return loss/(i+1), 100*(correct/total)

def get_label_wave(model, fixed_loader, args): # only CE loss for now
    model.eval()
    
    loss = 0
    correct = 0
    total = 0
    preds = []
    with torch.no_grad():
        for i, data in enumerate(fixed_loader):
            # if i == 0:
            # print(data[1])
            inputs, labels, indexes = data[0].to(args.device), data[1].to(args.device), data[4].to(args.device)
            # print(labels.shape, indexes.shape)
            # inputs, labels = data[0], data[1]
            
            outputs = model(inputs)
            
            if args.loss_type == 'ce':
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss_ = criterion(outputs, labels)
                loss_ = loss_.mean() # explicit mean reduction

            loss += loss_.detach().cpu().numpy()
            
            _, pred = torch.max(outputs.data, 1)
            # print(pred)
            total += labels.size(0)
            correct += (pred == labels).sum()
            preds.append(pred.detach().cpu())
            del data; gc.collect()
        preds = torch.cat(preds)
        # print(preds.shape)
            
    return loss/(i+1), 100*(correct/total), preds

def validate_(model, valid_loader, args):
    model.eval()
    
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            vstart = time.time()
            print(f'valid: {i}')
            inputs, labels, indexes = data[0].to(args.device), data[1].to(args.device), data[4].to(args.device)
            # print(labels.shape, indexes.shape)
            # inputs, labels = data[0], data[1]
            
            outputs = model(inputs)
            
            if args.loss_type == 'ce':
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss_ = criterion(outputs, labels)
                loss_ = loss_.mean() # explicit mean reduction
            elif args.loss_type == 'mae':
                # criterion = nn.L1Loss(reduction='none')
                # labels_onehot = F.one_hot(labels, num_classes=args.num_classes)
                # loss_ = criterion(F.softmax(outputs, dim=1), labels_onehot)
                # loss_ = loss_.mean() # explicit mean reduction
                criterion = MAELoss()
                loss_ = criterion(outputs, labels)
                loss_ = loss_.mean()
            elif args.loss_type == 'gce':
                criterion = GCELoss(q=0.7)
                loss_ = criterion(outputs, labels)
                loss_ = loss_.mean()
            elif args.loss_type == 'sce':
                if args.data == 'cifar10':
                    alpha = 0.1
                    beta = 1.0
                elif args.data == 'cifar100':
                    alpha = 6.0
                    beta = 0.1
                criterion = SCELoss(alpha=alpha, beta=beta, device=args.device, num_classes=args.num_classes)
                loss_ = criterion(outputs, labels)
                loss_ = loss_.mean()
            elif args.loss_type == 'elr':
                # if args.data == 'cifar10':
                #     lambda_ = 3
                #     beta = 0.7
                # elif args.data == 'cifar100':
                #     lambda_ = 7
                #     beta = 0.9
                # criterion = ELRLoss(num_examp=args.validset_size, num_classes=args.num_classes, lambda_=lambda_, beta=beta)
                # loss_ = criterion(indexes, outputs, labels)
                # loss_ = loss_.mean()
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss_ = criterion(outputs, labels)
                loss_ = loss_.mean() # explicit mean reduction
            elif args.loss_type == 'elr100':
                # if args.data == 'cifar10':
                #     lambda_ = 3
                #     beta = 0.7
                # elif args.data == 'cifar100':
                #     lambda_ = 7
                #     beta = 0.9
                # criterion = ELRLoss(num_examp=args.validset_size, num_classes=args.num_classes, lambda_=lambda_, beta=beta)
                # loss_ = criterion(indexes, outputs, labels)
                # loss_ = loss_.mean()
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss_ = criterion(outputs, labels)
                loss_ = loss_.mean() # explicit mean reduction
            
            loss += loss_.detach().cpu().numpy()
            
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum()
            
            del data; gc.collect(); torch.cuda.empty_cache()
            vend = time.time()
            print(vend-vstart)
            
    return loss/(i+1), 100*(correct/total)


def test(model, test_loader, args):
    correct = 0
    total = 0
    loss = 0
    
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            # inputs, labels = data[0], data[1]
            
            outputs = model(inputs)
            
            # calculate acc.
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum()
            
            # calculate loss
            loss_ = criterion(outputs, labels)
            loss_ = loss_.mean() # explicit mean reduction
            loss += loss_.detach().cpu().numpy()
            
            del data; gc.collect()
    
    return loss/(i+1), (100*correct/total).detach().cpu().numpy()

def test_(model, test_loader, args):
    correct = 0
    total = 0
    loss = 0
    
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data[0].to(args.device), data[1].to(args.device)
            # inputs, labels = data[0], data[1]
            
            outputs = model(inputs)
            
            # calculate acc.
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum()
            
            # calculate loss
            loss_ = criterion(outputs, labels)
            loss_ = loss_.mean() # explicit mean reduction
            loss += loss_.detach().cpu().numpy()
            
            del data; gc.collect(); torch.cuda.empty_cache()
    
    return loss/(i+1), (100*correct/total).detach().cpu().numpy()

def valid_loss_decomp(model, train_loader, args): # warning: runs for trainloader not validloader
    # grad_norms_df = {}
    # grad_absavgs_df = {}
    grad_norms = []
    grad_normavgs = []
    grad_absavgs = []
    grad_coss = []
    
    model.train()
    train_loss = 0
    train_loss_c = 0
    train_loss_w = 0
    n_c = 0
    n_w = 0
    clean_correct = 0
    clean_incorrect = 0
    wrong_correct = 0
    wrong_memorized = 0
    wrong_incorrect = 0
    
    for i, data in enumerate(train_loader):
        # print(f'start {i}')
        # GPUtil.showUtilization()
        inputs, noisy_labels, original_labels, noise_flags = None, None, None, None
        inputs, noisy_labels, original_labels, noise_flags = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device), data[3].to(args.device)
        outputs = model(inputs)
        criterion = nn.CrossEntropyLoss(reduction='none')

        loss = criterion(outputs, noisy_labels)
        loss_total = loss.sum() # explicit reduction
        loss_correct = loss[~noise_flags].sum()
        loss_wrong = loss[noise_flags].sum()
        n_c_ = (noise_flags == 0).sum()
        n_w_ = (noise_flags == 1).sum()
        
        train_loss += loss_total
        train_loss_c += loss_correct
        train_loss_w += loss_wrong
        n_c += n_c_
        n_w += n_w_
        
        clean_correct_ = ((outputs.argmax(dim=1) == noisy_labels) & (~noise_flags)).sum()
        clean_incorrect_ = ((outputs.argmax(dim=1) != noisy_labels) & (~noise_flags)).sum()
        wrong_correct_ = ((outputs.argmax(dim=1) == original_labels) & (noise_flags)).sum()
        wrong_memorized_ = ((outputs.argmax(dim=1) == noisy_labels) & (noise_flags)).sum()
        wrong_incorrect_ = n_w_ - wrong_correct_ - wrong_memorized_
        clean_correct += clean_correct_
        clean_incorrect += clean_incorrect_
        wrong_correct += wrong_correct_
        wrong_memorized += wrong_memorized_
        wrong_incorrect += wrong_incorrect_
        
        del data; gc.collect()
        # torch.cuda.empty_cache()
        
    print('assert')
    assert n_c+n_w == len(train_loader.dataset)
    
    train_loss = train_loss/(n_c+n_w)
    train_loss_c = train_loss_c/n_c
    train_loss_w = train_loss_w/n_w
    clean_correct = clean_correct/n_c
    clean_incorrect = clean_incorrect/n_c
    wrong_correct = wrong_correct/n_w
    wrong_memorized = wrong_memorized/n_w
    wrong_incorrect = wrong_incorrect/n_w
    
    for i, layer in enumerate(model.network.children()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grad_weight_total = torch.autograd.grad(train_loss, layer.weight, retain_graph=True)[0]
            grad_bias_total = torch.autograd.grad(train_loss, layer.bias, retain_graph=True)[0]
            grad_total = torch.cat((grad_weight_total.flatten(), grad_bias_total.flatten()), dim=0)
            # grad_norm_total = torch.cat((grad_weight_total.flatten(), grad_bias_total.flatten()), dim=0).norm(p=2)
            # grad_absavg_total = torch.cat((grad_weight_total.flatten(), grad_bias_total.flatten()), dim=0).abs().mean()
            grad_norm_total = grad_total.norm(p=2)
            grad_normavg_total = grad_total.norm(p=2)/grad_total.size()[0]
            grad_absavg_total = grad_total.abs().mean()
            # print('grad_total shape: ', grad_total.shape, layer.weight.flatten().shape + layer.bias.flatten().shape)

            grad_weight_correct = torch.autograd.grad(train_loss_c, layer.weight, retain_graph=True)[0]
            grad_bias_correct = torch.autograd.grad(train_loss_c, layer.bias, retain_graph=True)[0]
            grad_correct = torch.cat((grad_weight_correct.flatten(), grad_bias_correct.flatten()), dim=0)
            grad_norm_correct = grad_correct.norm(p=2)
            grad_normavg_correct = grad_correct.norm(p=2)/grad_correct.size()[0]
            grad_absavg_correct = grad_correct.abs().mean()
            
            

            grad_weight_wrong = torch.autograd.grad(train_loss_w, layer.weight, retain_graph=True)[0]
            grad_bias_wrong = torch.autograd.grad(train_loss_w, layer.bias, retain_graph=True)[0]
            grad_wrong = torch.cat((grad_weight_wrong.flatten(), grad_bias_wrong.flatten()), dim=0)
            grad_norm_wrong = grad_wrong.norm(p=2)
            grad_normavg_wrong = grad_wrong.norm(p=2)/grad_wrong.size()[0]
            grad_absavg_wrong = grad_wrong.abs().mean()
            
            grad_cos_cw  = (grad_correct @ grad_wrong) / (grad_norm_correct*grad_norm_wrong)
            grad_cos_tw  = (grad_total @ grad_wrong) / (grad_norm_total*grad_norm_wrong)
            grad_cos_tc  = (grad_total @ grad_correct) / (grad_norm_total*grad_norm_correct)
            
            grad_coss.append([grad_cos_cw.detach().cpu().numpy(), grad_cos_tw.detach().cpu().numpy(), grad_cos_tc.detach().cpu().numpy()])
            grad_norms.append([grad_norm_total.detach().cpu().numpy(), grad_norm_correct.detach().cpu().numpy(), grad_norm_wrong.detach().cpu().numpy()])
            grad_normavgs.append([grad_normavg_total.detach().cpu().numpy(), grad_normavg_correct.detach().cpu().numpy(), grad_normavg_wrong.detach().cpu().numpy()])
            grad_absavgs.append([grad_absavg_total.detach().cpu().numpy(), grad_absavg_correct.detach().cpu().numpy(), grad_absavg_wrong.detach().cpu().numpy()])
            
            # grad_norms_df[f'{i}-{layer.__class__.__name__}'] = [grad_norm_total.detach().cpu().numpy(), grad_norm_correct.detach().cpu().numpy(), grad_norm_wrong.detach().cpu().numpy()]
            # grad_absavgs_df[f'{i}-{layer.__class__.__name__}'] = [grad_absavg_total.detach().cpu().numpy(), grad_absavg_correct.detach().cpu().numpy(), grad_absavg_wrong.detach().cpu().numpy()]
    
        # grad_norms = pd.DataFrame.from_dict(data=grad_norms, orient='index', columns=['total', 'correct', 'wrong'])
        # grad_absavgs = pd.DataFrame.from_dict(data=grad_absavgs, orient='index', columns=['total', 'correct', 'wrong'])
    
    loss_lst = [train_loss.detach().cpu().numpy(), train_loss_c.detach().cpu().numpy(), train_loss_w.detach().cpu().numpy()]
    metric_lst = [clean_correct.detach().cpu().numpy(), clean_incorrect.detach().cpu().numpy(), wrong_correct.detach().cpu().numpy(), wrong_memorized.detach().cpu().numpy(), wrong_incorrect.detach().cpu().numpy()]
    grad_lst = [grad_norms, grad_normavgs, grad_absavgs, grad_coss]
    
    return loss_lst, metric_lst, grad_lst

def train_grad_decomp_by_batch(model, train_loader, optim, args, curr_iter): # warning: runs for trainloader not validloader
    model.train()
    with torch.autograd.set_detect_anomaly(True):
        data = next(iter(train_loader))
        # inputs, labels, indexes = data[0].to(args.device), data[1].to(args.device), data[4].to(args.device)
        inputs, noisy_labels, original_labels, noise_flags = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device), data[3].to(args.device)
        

        optim.zero_grad()

        outputs = model(inputs)

        if args.loss_type == 'ce':
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = criterion(outputs, noisy_labels)
            if curr_iter%args.log_iter == 0:
                loss_to_decomp = loss.clone()
            loss_to_back = loss.mean() # explicit mean reduction
        elif args.loss_type == 'mae':
            criterion = nn.L1Loss(reduction='none')
            criterion = MAELoss()
            loss = criterion(outputs, noisy_labels)
            loss_to_back = loss.mean()
        elif args.loss_type == 'gce':
            criterion = GCELoss(q=0.7)
            loss = criterion(outputs, noisy_labels)
            loss_to_back = loss.mean()
        elif args.loss_type == 'sce':
            if 'cifar10' in args.data:
                alpha = 0.1
                beta = 1.0
            elif 'cifar100' in args.data:
                alpha = 6.0
                beta = 0.1
            criterion = SCELoss(alpha=alpha, beta=beta, device=args.device, num_classes=args.num_classes)
            loss = criterion(outputs, noisy_labels)
            loss_to_back = loss.mean() 

        loss_to_back.backward(retain_graph=True)
        # optim.step()
    
    if curr_iter%args.log_iter == 0:

        grad_norms = []
        grad_normavgs = []
        grad_absavgs = []
        grad_coss = []

        n_c = (noise_flags == 0).sum()
        n_w = (noise_flags == 1).sum()
        
        train_loss = loss_to_decomp.sum()/(n_c+n_w) # explicit reduction
        train_loss_c = loss_to_decomp[~noise_flags].sum()/n_c
        train_loss_w = loss_to_decomp[noise_flags].sum()/n_w
        
        print('assert')
        assert n_c+n_w == inputs.size(0)
        
        
        clean_correct = ((outputs.argmax(dim=1) == noisy_labels) & (~noise_flags)).sum()
        clean_incorrect = ((outputs.argmax(dim=1) != noisy_labels) & (~noise_flags)).sum()
        wrong_correct = ((outputs.argmax(dim=1) == original_labels) & (noise_flags)).sum()
        wrong_memorized = ((outputs.argmax(dim=1) == noisy_labels) & (noise_flags)).sum()
        wrong_incorrect = n_w - wrong_correct - wrong_memorized

        clean_correct = clean_correct/n_c
        clean_incorrect = clean_incorrect/n_c
        wrong_correct = wrong_correct/n_w
        wrong_memorized = wrong_memorized/n_w
        wrong_incorrect = wrong_incorrect/n_w
        
        for i, layer in enumerate(model.network.children()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grad_weight_total = torch.autograd.grad(train_loss, layer.weight, retain_graph=True)[0]
                grad_bias_total = torch.autograd.grad(train_loss, layer.bias, retain_graph=True)[0]
                grad_total = torch.cat((grad_weight_total.flatten(), grad_bias_total.flatten()), dim=0)
                # grad_norm_total = torch.cat((grad_weight_total.flatten(), grad_bias_total.flatten()), dim=0).norm(p=2)
                # grad_absavg_total = torch.cat((grad_weight_total.flatten(), grad_bias_total.flatten()), dim=0).abs().mean()
                grad_norm_total = grad_total.norm(p=2)
                grad_normavg_total = grad_total.norm(p=2)/grad_total.size()[0]
                grad_absavg_total = grad_total.abs().mean()
                # print('grad_total shape: ', grad_total.shape, layer.weight.flatten().shape + layer.bias.flatten().shape)

                grad_weight_correct = torch.autograd.grad(train_loss_c, layer.weight, retain_graph=True)[0]
                grad_bias_correct = torch.autograd.grad(train_loss_c, layer.bias, retain_graph=True)[0]
                grad_correct = torch.cat((grad_weight_correct.flatten(), grad_bias_correct.flatten()), dim=0)
                grad_norm_correct = grad_correct.norm(p=2)
                grad_normavg_correct = grad_correct.norm(p=2)/grad_correct.size()[0]
                grad_absavg_correct = grad_correct.abs().mean()

                grad_weight_wrong = torch.autograd.grad(train_loss_w, layer.weight, retain_graph=True)[0]
                grad_bias_wrong = torch.autograd.grad(train_loss_w, layer.bias, retain_graph=True)[0]
                grad_wrong = torch.cat((grad_weight_wrong.flatten(), grad_bias_wrong.flatten()), dim=0)
                grad_norm_wrong = grad_wrong.norm(p=2)
                grad_normavg_wrong = grad_wrong.norm(p=2)/grad_wrong.size()[0]
                grad_absavg_wrong = grad_wrong.abs().mean()
                
                grad_cos_cw  = (grad_correct @ grad_wrong) / (grad_norm_correct*grad_norm_wrong)
                grad_cos_tw  = (grad_total @ grad_wrong) / (grad_norm_total*grad_norm_wrong)    
                grad_cos_tc  = (grad_total @ grad_correct) / (grad_norm_total*grad_norm_correct)
                
                print(f'noise flags: {noise_flags}')
                print(f'check cos_tw, {(grad_total @ grad_wrong).detach().cpu().numpy()} {(grad_norm_total*grad_norm_wrong).detach().cpu().numpy()} {grad_cos_tw.detach().cpu().numpy()}')
                print(f'check cos_tc, {(grad_total @ grad_correct).detach().cpu().numpy()} {(grad_norm_total*grad_norm_correct).detach().cpu().numpy()} {grad_cos_tc.detach().cpu().numpy()}')
                
                grad_coss.append([grad_cos_cw.detach().cpu().numpy(), grad_cos_tw.detach().cpu().numpy(), grad_cos_tc.detach().cpu().numpy()])
                grad_norms.append([grad_norm_total.detach().cpu().numpy(), grad_norm_correct.detach().cpu().numpy(), grad_norm_wrong.detach().cpu().numpy()])
                grad_normavgs.append([grad_normavg_total.detach().cpu().numpy(), grad_normavg_correct.detach().cpu().numpy(), grad_normavg_wrong.detach().cpu().numpy()])
                grad_absavgs.append([grad_absavg_total.detach().cpu().numpy(), grad_absavg_correct.detach().cpu().numpy(), grad_absavg_wrong.detach().cpu().numpy()])
        
        loss_lst = [train_loss.detach().cpu().numpy(), train_loss_c.detach().cpu().numpy(), train_loss_w.detach().cpu().numpy()]
        metric_lst = [clean_correct.detach().cpu().numpy(), clean_incorrect.detach().cpu().numpy(), wrong_correct.detach().cpu().numpy(), wrong_memorized.detach().cpu().numpy(), wrong_incorrect.detach().cpu().numpy()]
        grad_lst = [grad_norms, grad_normavgs, grad_absavgs, grad_coss]
        
        optim.step()
        del data; gc.collect()
        torch.cuda.empty_cache()
        
        return loss.detach().cpu().numpy(), loss_lst, metric_lst, grad_lst

    optim.step()
    del data; gc.collect()
    torch.cuda.empty_cache()
    
    return loss.detach().cpu().numpy()

def train_grad_decomp_by_batch_correction(model, train_loader, optim, args, curr_iter): # warning: runs for trainloader not validloader
    model.train()
    with torch.autograd.set_detect_anomaly(True):
        data = next(iter(train_loader))
        # inputs, labels, indexes = data[0].to(args.device), data[1].to(args.device), data[4].to(args.device)
        inputs, noisy_labels, original_labels, noise_flags = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device), data[3].to(args.device)
        

        optim.zero_grad()

        outputs = model(inputs)

        if args.loss_type == 'ce':
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = criterion(outputs, noisy_labels)
            if curr_iter%args.log_iter == 0:
                loss_to_decomp = loss.clone()
            loss_to_back = loss.mean() # explicit mean reduction
        elif args.loss_type == 'mae':
            criterion = nn.L1Loss(reduction='none')
            criterion = MAELoss()
            loss = criterion(outputs, noisy_labels)
            loss_to_back = loss.mean()
        elif args.loss_type == 'gce':
            criterion = GCELoss(q=0.7)
            loss = criterion(outputs, noisy_labels)
            loss_to_back = loss.mean()
        elif args.loss_type == 'sce':
            if 'cifar10' in args.data:
                alpha = 0.1
                beta = 1.0
            elif 'cifar100' in args.data:
                alpha = 6.0
                beta = 0.1
            criterion = SCELoss(alpha=alpha, beta=beta, device=args.device, num_classes=args.num_classes)
            loss = criterion(outputs, noisy_labels)
            loss_to_back = loss.mean() 

        loss_to_back.backward(retain_graph=True)
        # optim.step()
    
    if curr_iter%args.log_iter == 0:

        grad_norms = []
        grad_normavgs = []
        grad_absavgs = []
        grad_coss = []

        n_c = (noise_flags == 0).sum()
        n_w = (noise_flags == 1).sum()
        
        train_loss = loss_to_decomp.sum()/(n_c+n_w) # explicit reduction
        train_loss_c = loss_to_decomp[~noise_flags].sum()/n_c
        train_loss_w = loss_to_decomp[noise_flags].sum()/n_w
        
        print('assert')
        assert n_c+n_w == inputs.size(0)
        
        clean_correct = ((outputs.argmax(dim=1) == noisy_labels) & (~noise_flags)).sum()
        clean_incorrect = ((outputs.argmax(dim=1) != noisy_labels) & (~noise_flags)).sum()
        wrong_correct = ((outputs.argmax(dim=1) == original_labels) & (noise_flags)).sum()
        wrong_memorized = ((outputs.argmax(dim=1) == noisy_labels) & (noise_flags)).sum()
        wrong_incorrect = n_w - wrong_correct - wrong_memorized

        clean_correct = clean_correct/n_c
        clean_incorrect = clean_incorrect/n_c
        wrong_correct = wrong_correct/n_w
        wrong_memorized = wrong_memorized/n_w
        wrong_incorrect = wrong_incorrect/n_w
        
        for i, layer in enumerate(model.network.children()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grad_weight_total = torch.autograd.grad(train_loss, layer.weight, retain_graph=True)[0]
                grad_bias_total = torch.autograd.grad(train_loss, layer.bias, retain_graph=True)[0]
                grad_total = torch.cat((grad_weight_total.flatten(), grad_bias_total.flatten()), dim=0)
                # grad_norm_total = torch.cat((grad_weight_total.flatten(), grad_bias_total.flatten()), dim=0).norm(p=2)
                # grad_absavg_total = torch.cat((grad_weight_total.flatten(), grad_bias_total.flatten()), dim=0).abs().mean()
                grad_norm_total = grad_total.norm(p=2)
                grad_normavg_total = grad_total.norm(p=2)/grad_total.size()[0]
                grad_absavg_total = grad_total.abs().mean()
                # print('grad_total shape: ', grad_total.shape, layer.weight.flatten().shape + layer.bias.flatten().shape)

                grad_weight_correct = torch.autograd.grad(train_loss_c, layer.weight, retain_graph=True)[0]
                grad_bias_correct = torch.autograd.grad(train_loss_c, layer.bias, retain_graph=True)[0]
                grad_correct = torch.cat((grad_weight_correct.flatten(), grad_bias_correct.flatten()), dim=0)
                # add correction term to the gradients
                grad_correct = grad_correct * (n_c/(n_c+n_w))
                
                grad_norm_correct = grad_correct.norm(p=2)
                grad_normavg_correct = grad_correct.norm(p=2)/grad_correct.size()[0]
                grad_absavg_correct = grad_correct.abs().mean()

                grad_weight_wrong = torch.autograd.grad(train_loss_w, layer.weight, retain_graph=True)[0]
                grad_bias_wrong = torch.autograd.grad(train_loss_w, layer.bias, retain_graph=True)[0]
                grad_wrong = torch.cat((grad_weight_wrong.flatten(), grad_bias_wrong.flatten()), dim=0)
                # add correction term to the gradients
                grad_wrong = grad_wrong * (n_w/(n_c+n_w))
                
                grad_norm_wrong = grad_wrong.norm(p=2)
                grad_normavg_wrong = grad_wrong.norm(p=2)/grad_wrong.size()[0]
                grad_absavg_wrong = grad_wrong.abs().mean()
                
                grad_cos_cw  = (grad_correct @ grad_wrong) / (grad_norm_correct*grad_norm_wrong)
                grad_cos_tw  = (grad_total @ grad_wrong) / (grad_norm_total*grad_norm_wrong)    
                grad_cos_tc  = (grad_total @ grad_correct) / (grad_norm_total*grad_norm_correct)
                
                print(f'noise flags: {noise_flags}')
                print(f'check cos_tw, {(grad_total @ grad_wrong).detach().cpu().numpy()} {(grad_norm_total*grad_norm_wrong).detach().cpu().numpy()} {grad_cos_tw.detach().cpu().numpy()}')
                print(f'check cos_tc, {(grad_total @ grad_correct).detach().cpu().numpy()} {(grad_norm_total*grad_norm_correct).detach().cpu().numpy()} {grad_cos_tc.detach().cpu().numpy()}')
                
                grad_coss.append([grad_cos_cw.detach().cpu().numpy(), grad_cos_tw.detach().cpu().numpy(), grad_cos_tc.detach().cpu().numpy()])
                grad_norms.append([grad_norm_total.detach().cpu().numpy(), grad_norm_correct.detach().cpu().numpy(), grad_norm_wrong.detach().cpu().numpy()])
                grad_normavgs.append([grad_normavg_total.detach().cpu().numpy(), grad_normavg_correct.detach().cpu().numpy(), grad_normavg_wrong.detach().cpu().numpy()])
                grad_absavgs.append([grad_absavg_total.detach().cpu().numpy(), grad_absavg_correct.detach().cpu().numpy(), grad_absavg_wrong.detach().cpu().numpy()])
        
        loss_lst = [train_loss.detach().cpu().numpy(), train_loss_c.detach().cpu().numpy(), train_loss_w.detach().cpu().numpy()]
        metric_lst = [clean_correct.detach().cpu().numpy(), clean_incorrect.detach().cpu().numpy(), wrong_correct.detach().cpu().numpy(), wrong_memorized.detach().cpu().numpy(), wrong_incorrect.detach().cpu().numpy()]
        grad_lst = [grad_norms, grad_normavgs, grad_absavgs, grad_coss]
        
        optim.step()
        del data; gc.collect()
        torch.cuda.empty_cache()
        
        return loss.detach().cpu().numpy(), loss_lst, metric_lst, grad_lst

    optim.step()
    del data; gc.collect()
    torch.cuda.empty_cache()
    
    return loss.detach().cpu().numpy()
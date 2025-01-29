import sys
sys.path.append('../data')
sys.path.append('../data/gem')
import os
import math
from copy import deepcopy
import random
import hashlib
import errno
import pickle

import torch
import numpy as np
from numpy.testing import assert_array_almost_equal
import os.path as path

from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Lambda
from torchvision.datasets.vision import VisionDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Subset, WeightedRandomSampler
from torch_geometric.datasets import ZINC

from gem.datasets.cifair import ciFAIR10

def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files

# basic function#
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    #print np.max(y), P.shape[0]
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    #print m
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)
    print(f'flip with random seed {random_state}')

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    #print P

    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    #print P

    return y_train, actual_noise

def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate

def noisify_instance(train_data,train_labels,noise_rate):
    if max(train_labels)>10:
        num_class = 100
    else:
        num_class = 10
    np.random.seed(0)

    q_ = np.random.normal(loc=noise_rate,scale=0.1,size=1000000)
    q = []
    for pro in q_:
        if 0 < pro < 1:
            q.append(pro)
        if len(q)==50000:
            break

    w = np.random.normal(loc=0,scale=1,size=(32*32*3,num_class))

    noisy_labels = []
    for i, sample in enumerate(train_data):
        sample = sample.flatten()
        p_all = np.matmul(sample,w)
        p_all[train_labels[i]] = -1000000
        p_all = q[i]* F.softmax(torch.tensor(p_all),dim=0).numpy()
        p_all[train_labels[i]] = 1 - q[i]
        noisy_labels.append(np.random.choice(np.arange(num_class),p=p_all/sum(p_all)))
    over_all_noise_rate = 1 - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum())/50000
    return noisy_labels, over_all_noise_rate

class TruncatedLoss(nn.Module):

    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)
             
    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        if self.k == 0:
            loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes]
        else:
            loss = ((1-(Yg**self.q))/self.q)*self.weight[indexes] - ((1-(self.k**self.q))/self.q)*self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1-(Yg**self.q))/self.q)
        Lqk = np.repeat(((1-(self.k**self.q))/self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)
        

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)

class GCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()

        self.q = q
        self.eps = 1e-7

    def forward(self, outputs, labels):
        outputs_softmax = F.softmax(outputs, dim=1)
        f_j = torch.gather(outputs_softmax, 1, torch.unsqueeze(labels, 1))+self.eps
        
        numerator = 1 - (f_j**self.q)
        loss = numerator/self.q
        
        # loss = loss.mean()

        return loss
    
class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        # self.eps = 1e-7

    def forward(self, outputs, labels):
        outputs_softmax = F.softmax(outputs, dim=1)
        # f_j = torch.gather(outputs_softmax, 1, torch.unsqueeze(labels, 1))+self.eps
        f_j = torch.gather(outputs_softmax, 1, torch.unsqueeze(labels, 1))
        loss = 2 - 2*f_j
        
        # loss = loss.mean()

        return loss

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, device, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = device
        
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class ELRLoss(nn.Module):
    def __init__(self, num_examp, num_classes, lambda_=3, beta=0.7):
        super(ELRLoss, self).__init__()
        # num_examp = len(data_loader.dataset)
        
        # self.config = ConfigParser.get_instance()        
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
        self.lambda_ = lambda_
        self.beta = beta
        

    def forward(self, index, output, label):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()
        # final_loss = ce_loss +  self.config['train_loss']['args']['lambda']*elr_reg
        final_loss = ce_loss + self.lambda_*elr_reg
        
        return  final_loss
    

class ELRLoss_100(nn.Module):
    def __init__(self, num_examp, num_classes, lambda_=3, beta=0.7):
        super(ELRLoss_100, self).__init__()
        # num_examp = len(data_loader.dataset)
        
        # self.config = ConfigParser.get_instance()        
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
        self.lambda_ = lambda_
        self.beta = beta
        

    def forward(self, index, output, label):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        if self.num_classes == 100:
            y_pred = self.target[index] * y_pred
            y_pred = y_pred/(y_pred).sum(dim=1,keepdim=True)
        
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()
        # final_loss = ce_loss +  self.config['train_loss']['args']['lambda']*elr_reg
        final_loss = ce_loss + self.lambda_*elr_reg
        
        return  final_loss


class ELRLoss_plus(nn.Module):
    def __init__(self, num_examp, num_classes, lambda_=3, beta=0.7):
        super(ELRLoss_plus, self).__init__()
        # self.config = config
        self.USE_CUDA = torch.cuda.is_available()
        self.pred_hist = (torch.zeros(num_examp, num_classes)).cuda() if self.USE_CUDA else torch.zeros(num_examp, num_classes)
        self.q = 0
        self.lambda_ = lambda_
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, iteration, output, y_labeled):
        y_pred = F.softmax(output,dim=1)

        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)

        if self.num_classes == 100:
            y_labeled = y_labeled*self.q
            y_labeled = y_labeled/(y_labeled).sum(dim=1,keepdim=True)

        ce_loss = torch.mean(-torch.sum(y_labeled * F.log_softmax(output, dim=1), dim = -1))
        reg = ((1-(self.q * y_pred).sum(dim=1)).log()).mean()
        # final_loss = ce_loss + sigmoid_rampup(iteration, self.config['coef_step'])*(self.lambda_*reg)
        final_loss = ce_loss + self.lambda_*reg
        
      
        return  final_loss, y_pred.cpu().detach()

    def update_hist(self, epoch, out, index= None, mix_index = ..., mixup_l = 1):
        y_pred_ = F.softmax(out,dim=1)
        self.pred_hist[index] = self.beta * self.pred_hist[index] +  (1-self.beta) *  y_pred_/(y_pred_).sum(dim=1,keepdim=True)
        self.q = mixup_l * self.pred_hist[index]  + (1-mixup_l) * self.pred_hist[index][mix_index]

def perturb_model_(m, eps, device):
    with torch.no_grad():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # if hasattr(m, 'weight') or hasattr(m, 'bias'):
            # print(f'weight device: {m.weight.get_device()}')
            # print(f'eps device: {eps.get_device()}')
            w_perturb = torch.randn(m.weight.size()).to(device)*(m.weight.norm(p=2)/np.sqrt(m.weight.flatten().size()[0]))*eps
            b_perturb = torch.randn(m.bias.size()).to(device)*(m.bias.norm(p=2)/np.sqrt(m.bias.flatten().size()[0]))*eps
            m.weight.add_(w_perturb)
            m.bias.add_(b_perturb)
            
class NoisyCIFAIR10(Dataset):
    def __init__(self, root, split, transform, noise_rate, seed):
        self.dataset = ciFAIR10(root=root, split=split, transform=transform)
        self.noise_rate = noise_rate
        self.rng = np.random.RandomState(seed)
        self.noise_flags = self.generate_noise_flags()
        self.noisy_labels = self.generate_noisy_labels()
        self.original_labels = np.array(self.dataset.targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        original_label = self.original_labels[idx]
        noisy_label = self.noisy_labels[idx]
        noise_flag = self.noise_flags[idx]
        return img, noisy_label, original_label, noise_flag, idx

    def generate_noise_flags(self):
        labels = np.array(self.dataset.targets)
        noise_flags = self.rng.rand(len(labels)) < self.noise_rate
        return noise_flags
    
    def generate_noisy_labels(self):
        labels = np.array(self.dataset.targets)
        noise_flags = self.noise_flags
        # noisy_labels = labels.copy()
        noisy_labels = deepcopy(labels)
        for idx, flag in enumerate(noise_flags):
            if flag:
                possible_labels = list(range(10))
                possible_labels.remove(labels[idx])
                noisy_labels[idx] = self.rng.choice(possible_labels)
        return noisy_labels

class NoisyCIFAR10(Dataset):
    def __init__(self, root, train, download, transform, noise_rate, seed):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
        self.noise_rate = noise_rate
        self.rng = np.random.RandomState(seed)
        self.noise_flags = self.generate_noise_flags()
        self.noisy_labels = self.generate_noisy_labels()
        self.original_labels = np.array(self.dataset.targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        original_label = self.original_labels[idx]
        noisy_label = self.noisy_labels[idx]
        noise_flag = self.noise_flags[idx]
        return img, noisy_label, original_label, noise_flag, idx

    def generate_noise_flags(self):
        labels = np.array(self.dataset.targets)
        noise_flags = self.rng.rand(len(labels)) < self.noise_rate
        return noise_flags
    
    def generate_noisy_labels(self):
        labels = np.array(self.dataset.targets)
        noise_flags = self.noise_flags
        # noisy_labels = labels.copy()   
        noisy_labels = deepcopy(labels)
        for idx, flag in enumerate(noise_flags):
            if flag:
                possible_labels = list(range(10))
                possible_labels.remove(labels[idx])
                noisy_labels[idx] = self.rng.choice(possible_labels)
        return noisy_labels

class NoisyCIFAR100(Dataset):
    def __init__(self, root, train, download, transform, noise_rate, seed):
        self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=download, transform=transform)
        self.noise_rate = noise_rate
        self.rng = np.random.RandomState(seed)
        self.noise_flags = self.generate_noise_flags()
        self.noisy_labels = self.generate_noisy_labels()
        self.original_labels = np.array(self.dataset.targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        original_label = self.original_labels[idx]
        noisy_label = self.noisy_labels[idx]
        noise_flag = self.noise_flags[idx]
        return img, noisy_label, original_label, noise_flag, idx

    def generate_noise_flags(self):
        labels = np.array(self.dataset.targets)
        noise_flags = self.rng.rand(len(labels)) < self.noise_rate
        return noise_flags
    
    def generate_noisy_labels(self):
        labels = np.array(self.dataset.targets)
        noise_flags = self.noise_flags
        # noisy_labels = labels.copy()
        noisy_labels = deepcopy(labels)
        for idx, flag in enumerate(noise_flags):
            if flag:
                possible_labels = list(range(100))
                possible_labels.remove(labels[idx])
                noisy_labels[idx] = self.rng.choice(possible_labels)
        return noisy_labels

coarse_fine_id = {  0: [4, 30, 55, 72, 95],
                    1: [1, 32, 67, 73, 91],
                    2: [54, 62, 70, 82, 92],
                    3: [9, 10, 16, 28, 61],
                    4: [0, 51, 53, 57, 83],
                    5: [22, 39, 40, 86, 87],
                    6: [5, 20, 25, 84, 94],
                    7: [6, 7, 14, 18, 24],
                    8: [3, 42, 43, 88, 97],
                    9: [12, 17, 37, 68, 76],
                    10: [23, 33, 49, 60, 71],
                    11: [15, 19, 21, 31, 38],
                    12: [34, 63, 64, 66, 75],
                    13: [26, 45, 77, 79, 99],
                    14: [2, 11, 35, 46, 98],
                    15: [27, 29, 44, 78, 93],
                    16: [36, 50, 65, 74, 80],
                    17: [47, 52, 56, 59, 96],
                    18: [8, 13, 48, 58, 90],
                    19: [41, 69, 81, 85, 89]}

mapping = np.array(list(coarse_fine_id.values())).flatten()

class AsymmCIFAR10(Dataset):
    # def __init__(self, root, cfg_trainer, indexs, train=True, transform=None, target_transform=None, download=False):
    def __init__(self, root, train, download, transform, noise_rate, seed):
        # super(CIFAR10_train, self).__init__(root, train=train,
        #                                     transform=transform, target_transform=target_transform,
        #                                     download=download)
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
        self.noise_rate = noise_rate
        self.rng = np.random.RandomState(seed)
        self.num_classes = 10
        # self.cfg_trainer = cfg_trainer
        self.train_data = self.dataset.data #self.train_data[indexs]
        self.train_labels = np.array(self.dataset.targets)#np.array(self.train_labels)[indexs]
        # self.indexs = indexs
        self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
        self.noise_indx = []
        
    def symmetric_noise(self):
        self.train_labels_gt = self.train_labels.copy()
        #np.random.seed(seed=888)
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.noise_rate * len(self.train_data):
                self.noise_indx.append(idx)
                self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def asymmetric_noise(self):
        self.train_labels_gt = self.train_labels.copy()
        for i in range(self.num_classes):
            indices = np.where(self.train_labels == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                if j < self.noise_rate * len(indices):
                    self.noise_indx.append(idx)
                    # truck -> automobile
                    if i == 9:
                        self.train_labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.train_labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.train_labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.train_labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.train_labels[idx] = 7
        
        self.noise_flags = self.train_labels_gt - self.train_labels
        self.noise_flags[self.noise_flags !=0 ] = 1
        
    def __getitem__(self, index):
        # img, target, target_gt = self.train_data[index], self.train_labels[index], self.train_labels_gt[index]
        img, _ = self.dataset[index]
        target = self.train_labels[index]
        target_gt = self.train_labels_gt[index]
        noise_flag = self.noise_flags[index]
        
        
        # doing this so that it is consistent with all other datasets to return a PIL Image
        # img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target, target_gt, noise_flag, index

    def __len__(self):
        return len(self.train_data)

class AsymmCIFAR100(Dataset):
    # def __init__(self, root, cfg_trainer, indexs, train=True, transform=None, target_transform=None, download=False):
    def __init__(self, root, train, download, transform, noise_rate, seed):
        self.dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=download, transform=transform)
        self.noise_rate = noise_rate
        self.rng = np.random.RandomState(seed)
        self.num_classes = 100
        # self.cfg_trainer = cfg_trainer
        self.train_data = self.dataset.data #self.train_data[indexs]
        self.train_labels = np.array(self.dataset.targets) #np.array(self.train_labels)[indexs]
        # self.indexs = indexs
        self.prediction = np.zeros((len(self.train_data), self.num_classes, self.num_classes), dtype=np.float32)
        self.noise_indx = []
        #self.all_refs_encoded = torch.zeros(self.num_classes,self.num_ref,1024, dtype=np.float32)

        self.count = 0

    def symmetric_noise(self):
        self.train_labels_gt = self.train_labels.copy()
        indices = np.random.permutation(len(self.train_data))
        for i, idx in enumerate(indices):
            if i < self.noise_rate  * len(self.train_data):
                self.noise_indx.append(idx)
                self.train_labels[idx] = np.random.randint(self.num_classes, dtype=np.int32)

    def multiclass_noisify(self, y, P):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """

        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        # print(f'm: {m}')
        new_y = y.copy()
        # flipper = np.random.RandomState(random_state)
        flipper = self.rng

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            # print(i, flipped)
            # break
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y

#     def build_for_cifar100(self, size, noise):
#         """ random flip between two random classes.
#         """
#         assert(noise >= 0.) and (noise <= 1.)

#         P = np.eye(size)
#         cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
#         P[cls1, cls2] = noise
#         P[cls2, cls1] = noise
#         P[cls1, cls1] = 1.0 - noise
#         P[cls2, cls2] = 1.0 - noise

#         assert_array_almost_equal(P.sum(axis=1), 1, 1)
#         return P
    def build_for_cifar100(self, size, noise):
        """ The noise matrix flips to the "next" class with probability 'noise'.
        """

        assert(noise >= 0.) and (noise <= 1.)

        P = (1. - noise) * np.eye(size)
        for i in np.arange(size - 1):
            P[i, i + 1] = noise

        # adjust last row
        P[size - 1, 0] = noise

        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P


    def asymmetric_noise(self, original=False):
        self.train_labels_gt = self.train_labels.copy()
        P = np.eye(self.num_classes)
        n = self.noise_rate
        nb_superclasses = 20
        nb_subclasses = 5

        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i+1) * nb_subclasses
                P[init:end, init:end] = self.build_for_cifar100(nb_subclasses, n)

            P_ = np.zeros_like(P)
            T = np.zeros_like(P)
            for i, row in enumerate(P):
                # P_[i][mapping] = row
                for j in range(100):
                    P_[i][mapping[j]] = row[j]
            idx = np.empty_like(mapping)
            
            idx[mapping] = np.arange(len(mapping))
            P_[:] = P_[idx, :]
            
            if original:
                T = P
            else:
                T = P_
        
            y_train_noisy = self.multiclass_noisify(self.train_labels, T)
            actual_noise = (y_train_noisy != self.train_labels).mean()
            assert actual_noise > 0.0
            self.train_labels = y_train_noisy
            
            self.noise_flags = self.train_labels_gt - self.train_labels
            self.noise_flags[self.noise_flags !=0 ] = 1
        
        # comment below after test
        # return T

    def __getitem__(self, index):
        # img, target, target_gt = self.train_data[index], self.train_labels[index],  self.train_labels_gt[index]
        img, _ = self.dataset[index]
        target = self.train_labels[index]
        target_gt = self.train_labels_gt[index]
        noise_flag = self.noise_flags[index]
        

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target, target_gt, noise_flag, index

    def __len__(self):
        return len(self.train_data)

# from https://github.com/UCSC-REAL/cifar-10-100n/tree/main
class CIFAR10N(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_path = None, is_human=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='cifar10'
        self.noise_type=noise_type
        self.nb_classes=10
        self.noise_path = noise_path
        idx_each_class_noisy = [[] for i in range(10)]
        if download:
           self.download()


        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            #if noise_type is not None:
            if noise_type !='clean':
                # Load human noisy labels
                train_noisy_labels = self.load_label()
                self.train_noisy_labels = train_noisy_labels.tolist()
                print(f'noisy labels loaded from {self.noise_path}')

                if not is_human:
                    T = np.zeros((self.nb_classes,self.nb_classes))
                    for i in range(len(self.train_noisy_labels)):
                        T[self.train_labels[i]][self.train_noisy_labels[i]] += 1
                    T = T/np.sum(T,axis=1)
                    print(f'Noise transition matrix is \n{T}')
                    train_noisy_labels = multiclass_noisify(y=np.array(self.train_labels), P=T,
                                        random_state=0) #np.random.randint(1,10086)
                    self.train_noisy_labels = train_noisy_labels.tolist()
                    T = np.zeros((self.nb_classes,self.nb_classes))
                    for i in range(len(self.train_noisy_labels)):
                        T[self.train_labels[i]][self.train_noisy_labels[i]] += 1
                    T = T/np.sum(T,axis=1)
                    print(f'New synthetic noise transition matrix is \n{T}')

                for i in range(len(self.train_noisy_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                print(f'The noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)
                self.actual_noise_rate = np.sum(self.noise_or_not)/50000
                print('over all noise rate is ', self.actual_noise_rate)
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def load_label(self):
        #NOTE only load manual training label
        noise_label = torch.load(self.noise_path)
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.train_labels) - clean_label) == 0  
                print(f'Loaded {self.noise_type} from {self.noise_path}.')
                print(f'The overall noise rate is {1-np.mean(clean_label.numpy() == noise_label[self.noise_type])}')
            return noise_label[self.noise_type].reshape(-1)  
        else:
            raise Exception('Input Error')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type !='clean':
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, index, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



class CIFAR100N(CIFAR10N):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
 

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False,
                 noise_type=None, noise_rate=0.2, random_state=0,noise_path = None, is_human = True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='cifar100'
        self.noise_type=noise_type
        self.nb_classes=100
        self.noise_path = noise_path
        idx_each_class_noisy = [[] for i in range(100)]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            if noise_type !='clean':
                # load noise label
                train_noisy_labels = self.load_label()
                self.train_noisy_labels = train_noisy_labels.tolist()
                print(f'noisy labels loaded from {self.noise_type}')
                if not is_human:
                    T = np.zeros((self.nb_classes,self.nb_classes))
                    for i in range(len(self.train_noisy_labels)):
                        T[self.train_labels[i]][self.train_noisy_labels[i]] += 1
                    T = T/np.sum(T,axis=1)
                    print(f'Noise transition matrix is \n{T}')
                    train_noisy_labels = multiclass_noisify(y=np.array(self.train_labels), P=T,
                                        random_state=0) #np.random.randint(1,10086)
                    self.train_noisy_labels = train_noisy_labels.tolist()
                    T = np.zeros((self.nb_classes,self.nb_classes))
                    for i in range(len(self.train_noisy_labels)):
                        T[self.train_labels[i]][self.train_noisy_labels[i]] += 1
                    T = T/np.sum(T,axis=1)
                    print(f'New synthetic noise transition matrix is \n{T}')
                for i in range(len(self.train_labels)):
                    idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(100)]
                self.noise_prior = np.array(class_size_noisy)/sum(class_size_noisy)
                print(f'The noisy data ratio in each class is {self.noise_prior}')
                self.noise_or_not = np.transpose(self.train_noisy_labels)!=np.transpose(self.train_labels)
                self.actual_noise_rate = np.sum(self.noise_or_not)/50000
                print('over all noise rate is ', self.actual_noise_rate)
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

class TransformDataset(Dataset):
  def __init__(self, base_dataset, transform):
    super(TransformDataset, self).__init__()
    self.base = base_dataset
    self.transform = transform

  def __len__(self):
    return len(self.base)

  def __getitem__(self, idx):
    x, y, i, i, i = self.base[idx]
    return self.transform(x), y, i, i, i

class Animal10N(Dataset):
  def __init__(self, base_dataset, transform):
    super(Animal10N, self).__init__()
    self.base = base_dataset
    self.transform = transform

  def __len__(self):
    return len(self.base)

  def __getitem__(self, idx):
    x, y = self.base[idx]
    return self.transform(x), y, idx, idx, idx

# from https://github.com/chenpf1025/RobustnessAccuracy
class DATASET_CUSTOM(VisionDataset):

    def __init__(self, root, data, targets, transform=None, target_transform=None):
        super(DATASET_CUSTOM, self).__init__(root, transform=transform, target_transform=target_transform)

        self.data, self.targets = data, targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index, index, index

    def __len__(self):
        return len(self.data)

class Clothing1M(VisionDataset):
    def __init__(self, root, mode='train', transform=None, target_transform=None, sample_train=True, use_noisy_val=False):

        super(Clothing1M, self).__init__(root, transform=transform, target_transform=target_transform)


        if not use_noisy_val: # benchmark setting
            if mode=='train':
                if sample_train:
                    flist = os.path.join(root, "clean_val_annotations/noisy_train_sampled.txt")
                else:
                    flist = os.path.join(root, "clean_val_annotations/noisy_train.txt")
            if mode=='val':
                flist = os.path.join(root, "clean_val_annotations/clean_val.txt")
            if mode=='test':
                flist = os.path.join(root, "clean_val_annotations/clean_test.txt")
        else: # using a nnoisy validation setting, saving clean labels for training
            if mode=='train':
                flist = os.path.join(root, "noisy_val_annotations/nv_noisy_train.txt")
            if mode=='val':
                flist = os.path.join(root, "noisy_val_annotations/nv_noisy_val.txt")
            if mode=='test':
                flist = os.path.join(root, "noisy_val_annotations/nv_clean_test.txt")

        self.impaths, self.targets = self.flist_reader(flist)

        # # for debug
        # if mode=='train':
        #     self.impaths, self.targets = self.impaths[:1000], self.targets[:1000]


    def __getitem__(self, index):
        impath = self.impaths[index]
        target = self.targets[index]

        img = Image.open(impath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index, index, index

    def __len__(self):
        return len(self.impaths)

    def flist_reader(self, flist):
        impaths = []
        targets = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(" ")
                impaths.append(self.root + '/' + row[0])
                targets.append(int(row[1]))
        return impaths, targets

class ConstantAfterCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(ConstantAfterCosineLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.T_max:
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                    for base_lr in self.base_lrs]
        else:
            return [self.eta_min for _ in self.base_lrs]

class WrappedDataLoader:
    def __init__(self, dataloader, func, shuffle=False): #func is a function used of preprocess and move the tensor to gpu
        self.dataloader = dataloader
        self.func = func
        self.shuffle=shuffle
        self.address=[]
        batches = iter(self.dataloader)
        for b in batches:
            self.address.append(self.func(*b))

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.address)
            return iter(self.address)
        return iter(self.address)

def calculate_euclidean_distance(model1, model2):
    tot_distance = 0
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if param1.requires_grad and param2.requires_grad and param1.shape == param2.shape:
            distance = torch.square(param1 - param2).sum().detach().cpu().numpy()
            tot_distance += distance
    return np.sqrt(tot_distance)

def calculate_euclidean_distance_vcnn(model1, model2):
    calc_layers = [0, 3, 7, 10, 14, 17, 23, 26] # conv; 0, 3, 7, 10, 14, 17, linear; 23, 26
    layerwise_distance = []
    tot_distance = 0
    
    for i, layer in enumerate(calc_layers):
        distance_w = torch.square(model1.network[layer].weight-model2.network[layer].weight).sum().detach().cpu().numpy()
        distance_b = torch.square(model1.network[layer].bias-model2.network[layer].bias).sum().detach().cpu().numpy()
        distance = distance_w + distance_b
        
        tot_distance += distance
        layerwise_distance.append(distance)
    
    return layerwise_distance, np.sqrt(tot_distance)

def act_fn(name):
    """
    adapted from github.com/greydanus/hamiltonian-nn
    """
    nl = None
    if name == 'tanh':
        nl = nn.Tanh()
    elif name == 'relu':
        nl = nn.ReLU()
    elif name == 'sigmoid':
        nl = nn.Sigmoid()
    elif name == 'softplus':
        nl = nn.Softplus()
    elif name == 'selu':
        nl = nn.SELU()
    elif name == 'elu':
        nl = nn.ELU()
    elif name == 'silu':
        nl = nn.SiLU()
    elif name == 'mish':
        nl = nn.Mish()
    else:
        raise Exception("Undefined nonlinearity function.")
    return nl

def load_transform(args):
    transform = None

    if args.data == 'cifar10':
        if ('vit' in args.model)&('p16' in args.model):
            transform_train = transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            
            transform_test = transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        elif ('vit' in args.model)&('p16' in args.model)&('384' in args.model):
            transform_train = transforms.Compose([
                transforms.Resize(384),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            
            transform_test = transforms.Compose([
                transforms.Resize(384),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        else:    
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    
    elif args.data == 'cifar100':
        if ('vit' in args.model)&('p16' in args.model):
            transform_train = transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))])
            
            transform_test = transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))])
        elif ('vit' in args.model)&('p16' in args.model)&('384' in args.model):
            transform_train = transforms.Compose([
                transforms.Resize(384),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))])
            
            transform_test = transforms.Compose([
                transforms.Resize(384),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))])
        else:    
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))])
    
    elif args.data == 'cifar10n':
        if ('vit' in args.model)&('p16' in args.model):
            transform_train = transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            
            transform_test = transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        
        # elif ('vit' in args.model)&('p16' in args.model)&('384' in args.model):
        #     transform_train = transforms.Compose([
        #         transforms.Resize(384),
        #         # transforms.RandomCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            
        #     transform_test = transforms.Compose([
        #         transforms.Resize(384),
        #         # transforms.RandomCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        
    elif args.data == 'cifar100n':
        if ('vit' in args.model)&('p16' in args.model):
            transform_train = transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))])
            
            transform_test = transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))])
        
        # elif ('vit' in args.model)&('p16' in args.model)&('384' in args.model):
        #     transform_train = transforms.Compose([
        #         transforms.Resize(384),
        #         # transforms.RandomCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))])
            
        #     transform_test = transforms.Compose([
        #         transforms.Resize(384),
        #         # transforms.RandomCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))])
        
        else:    
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))])
            
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2675, 0.2565, 0.2761))])
        
    elif args.data == 'clothing1m':
        transform_train = transforms.Compose([
            # transforms.Resize(256),
            # transforms.RandomCrop(224),
            transforms.Resize(150),
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),                
            transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ]) 

        transform_test = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(150),
            transforms.RandomCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])

    elif args.data == 'tiny-imagenet':
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    elif args.data == 'imagenet':
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    elif args.data == 'animal10n':
        transform_train = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5138, 0.4843, 0.4213), (0.2046, 0.2046, 0.2047))])
        
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5084, 0.4785, 0.4160), (0.2060, 0.2061, 0.2048))])
    
    elif args.data == 'cifair':
        if args.split == 0:
            trainval_mean, trainval_std = (0.48881868, 0.47871667, 0.4426884), (0.24615733, 0.24149285, 0.2571039)
            test_mean, test_std = (0.49374542, 0.4842227, 0.449154), (0.24656914, 0.2427784, 0.26127002)
        elif args.split == 1:
            trainval_mean, trainval_std = (0.4916925, 0.48580545, 0.44541037), (0.24411494, 0.2432127, 0.2652114)
            test_mean, test_std = (0.49374542, 0.4842227, 0.449154), (0.24656914, 0.2427784, 0.26127002)
        elif args.split == 2:
            trainval_mean, trainval_std = (0.48881868, 0.47871667, 0.4426884), (0.24615733, 0.24149285, 0.2571039)
            test_mean, test_std = (0.49530792, 0.48619178, 0.45144373), (0.24746875, 0.24375987, 0.26063156)
        
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(trainval_mean, trainval_std)
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize(test_mean, test_std)
        ])
    
    return [transform_train, transform_test]


def load_dataset(transform, args):
    # if args.small_dataset == 1:
    #     if args.data == 'cifair':
    #         train_dataset = NoisyCIFAIR10(root='./data', split=f'trainval{args.split}', transform=transform[0], noise_rate=args.noise_rate, seed=args.seed)
    #         valid_dataset = ciFAIR10(root='./data', split='test0', transform=transform[1])
    #         test_dataset  = ciFAIR10(root='./data', split='test0', transform=transform[1])
        
        # return train_dataset, valid_dataset, test_dataset
            
        # if args.data == 'cifar10':
        #     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform[0])
        #     test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform[1])
    if args.data == 'cifar10':
        # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform[0])
        if args.class_dependent == 0: # symmetric noise
            train_dataset = NoisyCIFAR10(root='./data', train=True, download=False, transform=transform[0], noise_rate=args.noise_rate, seed=args.seed)
            if args.noisy_valid == 0:
                valid_dataset = NoisyCIFAR10(root='./data', train=False, download=False, transform=transform[1], noise_rate=0, seed=args.seed)
            elif args.noisy_valid == 1:
                valid_dataset = NoisyCIFAR10(root='./data', train=False, download=False, transform=transform[1], noise_rate=args.noise_rate, seed=args.seed)
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform[1])
        else: # asymmetric noise (class-dependent noise)
            train_dataset = AsymmCIFAR10(root='./data', train=True, download=False, transform=transform[0], noise_rate=args.noise_rate, seed=args.seed)
            train_dataset.asymmetric_noise()
            if args.noisy_valid == 0:
                valid_dataset = AsymmCIFAR10(root='./data', train=False, download=False, transform=transform[1], noise_rate=0, seed=args.seed)
            elif args.noisy_valid == 1:
                valid_dataset = AsymmCIFAR10(root='./data', train=False, download=False, transform=transform[1], noise_rate=args.noise_rate, seed=args.seed)
            valid_dataset.asymmetric_noise()
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform[1])

    elif args.data == 'cifar100':
        # train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform[0])
        if args.class_dependent == 0: # symmetric noise
            train_dataset = NoisyCIFAR100(root='./data', train=True, download=False, transform=transform[0], noise_rate=args.noise_rate, seed=args.seed)
            if args.noisy_valid == 0:
                valid_dataset =  NoisyCIFAR100(root='./data', train=False, download=False, transform=transform[1], noise_rate=0, seed=args.seed)
            elif args.noisy_valid == 1:
                valid_dataset = NoisyCIFAR100(root='./data', train=False, download=False, transform=transform[1], noise_rate=args.noise_rate, seed=args.seed)
            test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform[1])
        else: # asymmetric noise (class-dependent noise)
            train_dataset = AsymmCIFAR100(root='./data', train=True, download=False, transform=transform[0], noise_rate=args.noise_rate, seed=args.seed)
            train_dataset.asymmetric_noise()
            if args.noisy_valid == 0:
                valid_dataset =  AsymmCIFAR100(root='./data', train=False, download=False, transform=transform[1], noise_rate=0, seed=args.seed)
            elif args.noisy_valid == 1:
                valid_dataset = AsymmCIFAR100(root='./data', train=False, download=False, transform=transform[1], noise_rate=args.noise_rate, seed=args.seed)
            valid_dataset.asymmetric_noise()
            test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform[1])
    
    elif args.data == 'clothing1m':
        root = './data/clothing1M'
        train_dataset = Clothing1M(root=root, mode='train', sample_train=True, use_noisy_val=False, transform=transform[0])
        valid_dataset = Clothing1M(root=root, mode='val', sample_train=True, use_noisy_val=False, transform=transform[1])
        test_dataset = Clothing1M(root=root, mode='test', sample_train=True, use_noisy_val=False, transform=transform[1])
    
    elif args.data == 'cifar10n':
        noise_path = '/home/ywssng/projects/neural-network-resetting/data/CIFAR-10_human.pt'
        noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
        noise_type = noise_type_map[args.cifarn_noise_type]
        
        if args.noisy_valid:
            train_dataset = CIFAR10N(root='./data', download=False, train=True, is_human=True, noise_type=noise_type, noise_path=noise_path, transform=None)
            size = len(train_dataset)
            valid_ratio = 0.1
            valid_size = int(valid_ratio*size)
            train_size = size - valid_size
            train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
            
            train_dataset = TransformDataset(train_dataset, transform[0])
            valid_dataset = TransformDataset(valid_dataset, transform[1])
            test_dataset = CIFAR10N(root='./data', download=False, train=False, is_human=True, noise_type=noise_type, noise_path=noise_path, transform=transform[1])
        else:
            train_dataset = CIFAR10N(root='./data', download=False, train=True, is_human=True, noise_type=noise_type, noise_path=noise_path, transform=transform[0])
            valid_dataset = CIFAR10N(root='./data', download=False, train=False, is_human=True, noise_type=noise_type, noise_path=noise_path, transform=transform[1])
            test_dataset = CIFAR10N(root='./data', download=False, train=False, is_human=True, noise_type=noise_type, noise_path=noise_path, transform=transform[1])


    elif args.data == 'cifar100n':
        noise_path = '/home/ywssng/projects/neural-network-resetting/data/CIFAR-100_human.pt'
        noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
        noise_type = noise_type_map[args.cifarn_noise_type]
        
        if args.noisy_valid:
            train_dataset = CIFAR100N(root='./data', download=False, train=True, is_human=True, noise_type=noise_type, noise_path=noise_path, transform=None)
            size = len(train_dataset)
            valid_ratio = 0.1
            valid_size = int(valid_ratio*size)
            train_size = size - valid_size
            train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
            
            train_dataset = TransformDataset(train_dataset, transform[0])
            valid_dataset = TransformDataset(valid_dataset, transform[1])
            test_dataset = CIFAR100N(root='./data', download=False, train=False, is_human=True, noise_type=noise_type, noise_path=noise_path, transform=transform[1])
        else:
            train_dataset = CIFAR100N(root='./data', download=False, train=True, is_human=True, noise_type=noise_type, noise_path=noise_path, transform=transform[0])
            valid_dataset = CIFAR100N(root='./data', download=False, train=False, is_human=True, noise_type=noise_type, noise_path=noise_path, transform=transform[1])
            test_dataset = CIFAR100N(root='./data', download=False, train=False, is_human=True, noise_type=noise_type, noise_path=noise_path, transform=transform[1])
    
    elif args.data == 'animal10n':
        train_path = '/home/ywssng/projects/neural-network-resetting/data/Animal10N/train'
        test_path = '/home/ywssng/projects/neural-network-resetting/data/Animal10N/test'
        
        if args.noisy_valid:
            train_dataset = Animal10N(ImageFolder(train_path), transform=transform[0])
            size = len(train_dataset)
            valid_ratio = 0.1
            valid_size = int(valid_ratio*size)
            train_size = size - valid_size
            train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
            test_dataset = Animal10N(ImageFolder(test_path), transform=transform[1])
        else:
            train_dataset = Animal10N(ImageFolder(train_path), transform=transform[0])
            valid_dataset = Animal10N(ImageFolder(test_path), transform=transform[1])
            test_dataset = Animal10N(ImageFolder(test_path), transform=transform[1])
    
    elif args.data == 'tiny-imagenet':
        train_dir = './data/tiny-imagenet/train'
        test_dir = './data/tiny-imagenet/test'
        train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform[0])
        test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform[1])
        
    elif args.data == 'imagenet':
        train_dir = './data/imagenet/train'
        test_dir = './data/imagenet/val'
        train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform[0])
        test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform[1])
    
    elif args.data == 'cifair':
        train_dataset = NoisyCIFAIR10(root='./data', split=f'trainval{args.split}', transform=transform[0], noise_rate=args.noise_rate, seed=args.seed)
        valid_dataset = NoisyCIFAIR10(root='./data', split=f'test0', transform=transform[1], noise_rate=0, seed=args.seed)
        test_dataset = NoisyCIFAIR10(root='./data', split=f'test0', transform=transform[1], noise_rate=0, seed=args.seed)
        # valid_dataset = ciFAIR10(root='./data', split='test0', transform=transform[1])
        # test_dataset  = ciFAIR10(root='./data', split='test0', transform=transform[1])
        
        # valid_size = int(args.valid_ratio * len(trainset))
        # train_size = len(trainset) - valid_size

        # lengths = [train_size, valid_size]

        # train_dataset_, valid_dataset = torch.utils.data.dataset.random_split(trainset, lengths, generator=torch.Generator().manual_seed(42))
        
        # sample_train_size = int(args.sample_train*len(train_dataset_))
        # discard_train_size = len(train_dataset_) - sample_train_size
        # train_dataset, _ = torch.utils.data.dataset.random_split(train_dataset_, [sample_train_size, discard_train_size])
    
    return train_dataset, valid_dataset, test_dataset

def load_zinc(args):
    data_dir = '/home/ywssng/projects/neural-network-resetting/data/zinc'
    
    train_dataset_ = ZINC(
        root=data_dir,
        subset=args.zinc_subset,
        split='train'
    )
    
    if args.bias_trainset:
        train_log = []
        for data in train_dataset_:
            train_log.append(data.y)

        logP_values = torch.as_tensor(train_log).numpy()

        # Assuming dataset is your pytorch dataset with an attribute logP_values
        # logP_values = train_log.numpy()
        # subset_size = 5000
        # Assign weights based on logP values
        left = -0.55
        right = 1.7
        weights = np.ones_like(logP_values)
        weights[(logP_values >= left) & (logP_values <= right)] = 5
        weights[(logP_values < left) | (logP_values > right)] = 0.1
        # Adjust weights for values close to the boundaries for a smoother transition (if desired)

        # Create a weighted random sampler to sample indices
        if np.sum(weights > 0) < args.subset_size:
            raise ValueError("There aren't enough data points with non-zero weights to create the subset without replacement.")

        sampler = WeightedRandomSampler(weights, num_samples=args.subset_size, replacement=False)

        # Get the sampled indices
        sampled_indices = list(sampler)

        # Create a new dataset using the sampled indices
        train_dataset = Subset(train_dataset_, sampled_indices)
    
    else:
        sample_train_size = int(args.sample_train*len(train_dataset_))
        discard_train_size = len(train_dataset_) - sample_train_size
        train_dataset, _ = torch.utils.data.dataset.random_split(train_dataset_, [sample_train_size, discard_train_size], generator=torch.Generator().manual_seed(args.subset_seed))
    
    val_dataset = ZINC(
        root=data_dir,
        subset=args.zinc_subset,
        split='val'
    )
    test_dataset = ZINC(
        root=data_dir,
        subset=args.zinc_subset,
        split='test'
    )
    
    
    return train_dataset, val_dataset, test_dataset


def save_checkpoint(state, best_state, save_path):
    print('start', save_path)
    torch.save(state, path.join(save_path, "checkpoint.pth.tar"))
    torch.save(best_state, path.join(save_path, "best_model.pth.tar"))
    print('end', save_path)
    

def save_resume_checkpoint(state, best_state, save_path):
    torch.save(state, path.join(save_path, "checkpoint_resume.pth.tar"))
    torch.save(best_state, path.join(save_path, "best_model_resume.pth.tar"))

# OS
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import gc
import argparse
import pickle
from copy import deepcopy

# mathematics
import random
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

# data handling
import pandas as pd

# Torch
import torch
import torch.optim as optim

# logging
import wandb
# import GPUtil

# my_lib
from model.net import *
from model.train import train, validate, test_
from misc.utils import load_transform, load_dataset, save_checkpoint, ConstantAfterCosineLR


# @profile
def main(origin_args):
    print(torch.__version__)
    args = deepcopy(origin_args)
    
    # @profile
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
    random.seed(args.seed)
    rs = RandomState(MT19937(SeedSequence(args.seed)))
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # doesn't ensure but show warning when non-deterministic ops. are used
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    args.reset_prob_ = np.format_float_positional(args.reset_prob, trim='-', precision=4) # precicsion could change
    args.lr_ = np.format_float_positional(args.lr, trim='-')
    args.save = args.save_path # + '/seed%d' %args.seed
    os.makedirs(args.save, exist_ok=True)
    
    ckpt_path = args.save.split('/')[1:]
    ckpt_path.insert(0, 'checkpoints')
    ckpt_path = os.path.join(*ckpt_path)
    os.makedirs(ckpt_path, exist_ok=True)
    os.environ['WANDB_API_KEY'] = args.wandb_api_key # or set your wandb api key here manually
    os.environ['WANDB_ENTITY'] = args.wandb_team
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
    run = wandb.init(project=args.wandb_project, tags=['restart-False'] if args.reset_prob==0 else ['restart-True'], notes=args.wandb_notes)
    wandb.config.update(args)
    wandb.define_metric("loss", summary="min")
    wandb.define_metric("acc", summary="max")
    
    for key, value in wandb.config.items():
        print(f'{key} : {value}')

    transforms = load_transform(args)
    train_dataset, valid_dataset, test_dataset = load_dataset(transforms, args)


    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(args.seed)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    del train_dataset, valid_dataset, test_dataset; gc.collect()
    
    curr_iter = 0
    results = {}
    results["loss_train"] = [[np.inf, 0]]
    results["loss_valid"] = [[np.inf, 0]]
    results["loss_test"] = [[np.inf, 0]]
    results["loss_valid_best"] = [[np.inf, 0]]
    results["acc_valid_best"] = [[0, 0]]
    results["acc_test"] = [[0, 0]]
    results["acc_valid"] = [[0, 0]]
    results["track_reset"] = [[0, 0]]
    results["track_norm"] = [[0, 0]]
    results["lr"] = [[0, 0]]
    results['layerwise_distance_curr'] = []
    results['layerwise_distance_best'] = []
    results['tot_distance_curr'] = []
    results['tot_distance_best'] = []
    results['distance_curr'] = []
    results['distance_best'] = []
    
    reset_flag = False
    best_valid_loss = np.inf
    best_valid_acc = 0
    best_iter = 0
    test_acc = 0
    
    curr_net, reset_net, best_net = load_model(args)
    print(f'model loaded: {args.model}')
    wandb.watch(models=curr_net, log='all', log_freq=1000)
    # print(curr_net)

    if args.opt == 'sgd':
        curr_optimizer = optim.SGD(curr_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.opt_reset:
            reset_optimizer = optim.SGD(curr_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            best_optimizer = optim.SGD(curr_net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    elif args.opt == 'adam':
        curr_optimizer = optim.Adam(curr_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.opt_reset:
            reset_optimizer = optim.Adam(curr_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_optimizer = optim.Adam(curr_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    elif args.opt == 'adamw':
        curr_optimizer = optim.AdamW(curr_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.opt_reset:
            reset_optimizer = optim.AdamW(curr_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_optimizer = optim.AdamW(curr_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # learning rate scheduling
    if args.lr_schedule != 'none':
        if args.lr_schedule == 'cosineannealingwarmrestart':
            main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                curr_optimizer, T_max=args.tot_iter-args.warmup_iter
                )
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                curr_optimizer, start_factor=args.warmup_decay, total_iters=args.warmup_iter
            )
            
            curr_lr_scheduler = optim.lr_scheduler.SequentialLR(
                curr_optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.warmup_iter]
            )
            best_lr_scheduler = optim.lr_scheduler.SequentialLR(
                curr_optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.warmup_iter]
            )
            # reset_lr_scheduler = optim.lr_scheduler.SequentialLR(
                # curr_optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.warmup_iter]
            # )
        
        elif args.lr_schedule == 'cosineannealing':
            curr_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(curr_optimizer, T_max=args.tot_iter)
            # reset_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(curr_optimizer, T_max=args.tot_iter)
            best_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(curr_optimizer, T_max=args.tot_iter)

        elif args.lr_schedule == 'linear':
            curr_lr_scheduler = optim.lr_scheduler.PolynomialLR(curr_optimizer, total_iters=args.tot_iter, power=1.0)
            # reset_lr_scheduler = optim.lr_scheduler.PolynomialLR(curr_optimizer, total_iters=args.tot_iter, power=1.0)
            best_lr_scheduler = optim.lr_scheduler.PolynomialLR(curr_optimizer, total_iters=args.tot_iter, power=1.0)
        
        elif args.lr_schedule == 'step':
            args.milestone = [25000, 40000]
            curr_lr_scheduler = optim.lr_scheduler.MultiStepLR(curr_optimizer, milestones=args.milestone, gamma=args.gamma)
            # reset_lr_scheduler = optim.lr_scheduler.MultiStepLR(curr_optimizer, milestones=args.milestone, gamma=args.gamma)
            best_lr_scheduler = optim.lr_scheduler.MultiStepLR(curr_optimizer, milestones=args.milestone, gamma=args.gamma)
        
        elif args.lr_schedule == 'constantaftercosine':
            curr_lr_scheduler = ConstantAfterCosineLR(curr_optimizer, T_max=args.T_warm, eta_min=args.eta_min)
            # reset_lr_scheduler = ConstantAfterCosineLR(curr_optimizer, T_max=args.T_warm, eta_min=args.eta_min)
            best_lr_scheduler = ConstantAfterCosineLR(curr_optimizer, T_max=args.T_warm, eta_min=args.eta_min)
        else:
            raise RuntimeError(f'Invalid lr scheuler {args.lr_schedule}')

    for curr_iter in range(args.tot_iter):
        # log norm of the weights, and the reset flag
        weights_norm = torch.norm(torch.cat([param.view(-1) for param in curr_net.parameters()]), p=2).detach().cpu().numpy()        
        results["track_norm"].append([weights_norm, curr_iter])
        results["track_reset"].append([1 if reset_flag else 0, curr_iter])
        
        # train
        train_loss = train(curr_net, trainloader, curr_optimizer, args)
        print(f'iter: {curr_iter}, train loss: {train_loss}')
        lr = deepcopy(curr_optimizer.param_groups[0]['lr'])
        results["loss_train"].append([train_loss, curr_iter])
        wandb.log({
                    'train_loss': train_loss,
                    'reset_flag': 1 if reset_flag else 0,
                    'weights_norm': weights_norm,
                    'lr': lr,
                    },
                step=curr_iter
                )
        
        # stochastic resetting process
        if reset_flag is True:
            prob = np.random.rand()
            if prob < args.reset_prob:
                curr_net.load_state_dict(reset_net.state_dict())
                if args.opt_reset:
                    curr_optimizer.load_state_dict(reset_optimizer.state_dict())
                    if args.scheduler_reset == 0:
                        curr_optimizer.param_groups[0]['lr'] = lr
                        print('scheduler no reset')
                    else:
                        print('scheduler reset')
            
            # uncomment below if needed
            # if args.reset_prob==1:
            #     break
    
        # validation
        if curr_iter % (args.log_iter) == 0:
            valid_loss, valid_acc = validate(curr_net, validloader, args)
            results["loss_valid"].append([valid_loss, curr_iter])
            results["acc_valid"].append([valid_acc, curr_iter])
            
            if args.best_metric == 'loss':
                is_best = valid_loss < best_valid_loss
            elif args.best_metric == 'acc':
                is_best = valid_acc > best_valid_acc
            else:
                raise RuntimeError(f'Invalid best metric {args.best_metric}')
            
            if is_best:
                best_iter = curr_iter

                best_valid_loss = valid_loss
                best_valid_acc = valid_acc

                best_net.load_state_dict(curr_net.state_dict())
                best_optimizer.load_state_dict(curr_optimizer.state_dict())
                if args.lr_schedule != 'none':
                    best_lr_scheduler.load_state_dict(curr_lr_scheduler.state_dict())
                
                # adaptive reset
                if args.adaptive and reset_flag:
                    reset_net.load_state_dict(best_net.state_dict())
                    reset_optimizer.load_state_dict(best_optimizer.state_dict())
                
                # run test if best model is updated
                test_loss, test_acc = test_(best_net, testloader, args)
                results["loss_test"].append([test_loss, curr_iter])
                results["acc_test"].append([test_acc, curr_iter])
            
            if curr_iter >= args.warmup_iter:
                results["loss_valid_best"].append([best_valid_loss, curr_iter])
                results["acc_valid_best"].append([best_valid_acc, curr_iter])
            
            wandb.log({
                        'valid_loss': valid_loss,
                        'valid_acc': valid_acc,
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                        'best_valid_loss': best_valid_loss,
                        'best_valid_acc': best_valid_acc,
                        },
                    step=curr_iter
                    )

            if reset_flag is False:
                if args.best_metric == 'acc':
                    metric_flag = len(results["acc_valid_best"]) > (args.threshold_iter//args.log_iter) and best_valid_acc == results["acc_valid_best"][-args.threshold_iter//args.log_iter][0]
                
                elif args.best_metric == 'loss':
                    metric_flag = len(results["loss_valid_best"]) > (args.threshold_iter//args.log_iter)  and best_valid_loss == results["loss_valid_best"][-args.threshold_iter//args.log_iter][0]

                if curr_iter > (args.warmup_iter) and metric_flag:
                    reset_flag = True
                    reset_net.load_state_dict(best_net.state_dict()) # reset to best network 
                    if args.opt_reset:
                        reset_optimizer.load_state_dict(best_optimizer.state_dict())

            print(f'iter: {curr_iter}')
            print(f'train_loss: {train_loss}, valid_loss: {valid_loss}, valid_acc: {valid_acc}')
            print(f'best_valid_loss: {best_valid_loss}, best_valid_acc: {best_valid_acc}')
            print(f'test_loss: {test_loss}, test_acc: {test_acc}')
            
            file_name = 'metric_results.pkl'
            with open(os.path.join(args.save, file_name), "wb") as f:
                pickle.dump(results, f)
            
        if curr_iter % 10000 == 0:
            file_name = f'metric_results_at_{curr_iter}.pkl'
            with open(os.path.join(args.save, file_name), "wb") as f:
                pickle.dump(results, f)
                
        # learning rate update
        if args.lr_schedule == 'none':
            pass
        else:
            curr_lr_scheduler.step()

    test_loss, test_acc = test_(best_net, testloader, args)
    results["loss_test"].append([test_loss, curr_iter])
    results["acc_test"].append([test_acc, curr_iter])

    # save results as .pkl with pickles
    file_name = 'metric_results.pkl'
    with open(os.path.join(args.save, file_name), "wb") as f:
        pickle.dump(results, f)

    save_checkpoint(
        {
            "curr_iter": curr_iter,
            "curr_state_dict": curr_net.state_dict(),
            "reset_net_state_dict": reset_net.state_dict(),
            "train_loss": train_loss,
            "optimizer": curr_optimizer.state_dict(),
            "lr_scheduler": curr_lr_scheduler.state_dict() if args.lr_schedule != 'none' else None,
            "wandb_run_id": os.environ["WANDB_RUN_ID"]
        },
        {
            "best_iter": best_iter,
            "state_dict": best_net.state_dict(),
            "best_valid_loss": best_valid_loss,
            "best_test_acc": test_acc,
            "wandb_run_id": os.environ["WANDB_RUN_ID"]
        },
        # args.save
        ckpt_path
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SGD with Stochastic Resetting"
    )
    parser.add_argument(
        "--save-path",
        default="./checkpoint",
        type=str,
        metavar="PATH",
        help="path to save result (default: none)",
    )
    parser.add_argument(
        "--wandb-api-key",
        default="wandbapikey",
        type=str,
        help="wandb api key",
    )
    parser.add_argument(
        "--wandb-project",
        default="nn-reset",
        type=str,
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb-team",
        default="nn-reset",
        type=str,
        help="wandb team name",
    )
    parser.add_argument(
        "--wandb-notes",
        default="none",
        type=str,
        help="wandb notes (default: none)",
    )
    parser.add_argument(
        "--job-name",
        default="none",
        type=str,
        help="job name with loaded number (at the cluster)",
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='cifair', 
        help="choice for dataset"
    )
    parser.add_argument(
        '--small-dataset', 
        type=str, 
        default=0, 
        help="choice for using small dataset; cifair, etc."
    )
    parser.add_argument(
        '--split', 
        type=int, 
        default=0, 
        help="dataset split for small datsets"
    )
    parser.add_argument(
        '--sample-train', 
        type=float, 
        default='0.1', 
        help="sampling size for training dataset (for fast convergence)"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='vcnn', 
        help="choice for model (default: vcnn)"
    )
    parser.add_argument(
        '--best-metric', 
        type=str, 
        default='loss', 
        help="which metric to track best model (default: loss, choose between loss or acc)"
    )
    parser.add_argument(
        '--pretrained', 
        type=int, 
        default=1, 
        help="choice for calling pretrained model or not"
    )
    parser.add_argument(
        '--reset-prob', 
        type=float, 
        default=0.01,
        help="Resetting probability when the DNN begins the stochastic resetting process. (default: 0.01)",
    )
    parser.add_argument(
        '--threshold-iter', 
        type=int, 
        default=10, 
        help="Threshold iter (TI). If the best validation loss does not change until TI, the stochastic resetting process starts. (default: 1000)"
    )
    parser.add_argument(
        '--tot-iter', 
        type=int, 
        default=30000
    )
    parser.add_argument(
        '--log-iter', 
        type=int, 
        default=100
    )
    parser.add_argument(
        '--warmup-iter', 
        type=int, 
        default=0
    )
    parser.add_argument(
        '--warmup-decay', 
        type=float, 
        default=0.01
    )
    parser.add_argument(
        '--opt', 
        type=str, 
        default='sgd', 
        help="choice for optimizer (default: sgd)"
    )
    parser.add_argument(
        '--opt-reset', 
        type=int, 
        default=0, 
        help="choice to whether reset also optimizer or not (default: 0)"
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=1e-2, 
        help="learning rate for optimizer (default: 1e-2)"
    )
    parser.add_argument(
        '--lr-schedule', 
        type=str, 
        default='cosineannealing', 
        help="choice for lr schedule (default: cosineannealing)"
    )
    parser.add_argument(
        '--gamma', 
        type=float, 
        default=0.1, 
        help="learning rate decay gamma"
    )
    parser.add_argument(
        '--momentum', 
        type=float, 
        default=0.9, 
        help="momentum for optimizer (default: 1e-2)"
    )
    parser.add_argument(
        '--weight-decay', 
        type=float, 
        default=1e-3, 
        help="weight decay for optimizer (default: 1e-4)"
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32, 
        help="batch size (default: 32)"
    )
    parser.add_argument(
        '--test-batch-size', 
        type=int, 
        default=4096
    )
    parser.add_argument(
        '--valid-ratio', 
        type=float, 
        default=0.2
    )
    parser.add_argument(
        '--num-workers', 
        type=int, 
        default=0, 
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=1
    )
    parser.add_argument(
        '--cuda', 
        type=int, 
        default=0
    )
    parser.add_argument(
        '--comment', 
        type=str
    )
    parser.add_argument(
        '--noise-rate', 
        type=float, 
        default=0, 
        help="add noise to labels with the amount of noise-rate"
    )
    parser.add_argument(
        '--adaptive', 
        type=float, 
        default=0, 
        help="adaptive reset option"
    )
    parser.add_argument(
        '--scheduler-reset', 
        type=int, 
        default=1, 
        help="whether to reset lr scheduler or not"
    )
    parser.add_argument(
        '--loss-type', 
        type=str, 
        default='ce', 
        help="loss type; currently ce, mae, gce, tgce implemented"
    )
    parser.add_argument(
        '--noisy-valid', 
        type=int, 
        default=1, 
        help="whether to currpt validation set or not"
    )
    parser.add_argument(
        '--class-dependent', 
        type=int, 
        default=0, 
        help="is dataset noise is class dependent or not"
    )
    parser.add_argument(
        '--cifarn-noise-type', 
        type=str, 
        default='none', 
        help="type of noise for cifar-noisy dataset"
    )
    parser.add_argument(
        '--milestone', 
        type=int, 
        default=10000, 
        help="first milestone for multisteplr, i.e. warmup epoch"
    )
    parser.add_argument(
        '--T-warm', 
        type=int, 
        default=10000, 
        help="T_max for cosine annealing part"
    )
    parser.add_argument(
        '--eta-min', 
        type=float, 
        default=1e-3, 
        help="eta_min, i.e. minimum learning rate when cosine ends"
    )
    
    
    args = parser.parse_args()
    if args.data == 'cifar10':
        args.num_classes = 10
        args.trainset_size = 50000
        args.validset_size = 10000
    elif args.data == 'cifar100':
        args.num_classes = 100
        args.trainset_size = 50000
        args.validset_size = 10000
    elif args.data == 'cifar10n':
        args.num_classes = 10
        args.trainset_size = 50000
        args.validset_size = 10000
    elif args.data == 'cifar100n':
        args.num_classes = 100
        args.trainset_size = 50000
        args.validset_size = 10000
    elif args.data == 'clothing1m':
        args.num_classes = 14
    elif args.data == 'tiny-imagenet':
        # args.num_classes = 1000
        args.num_classes = 200
    elif args.data == 'imagenet':
        args.num_classes = 1000
    elif args.data == 'cifair':
        args.num_classes = 10
        args.trainset_size = 500
        args.validset_size = 10000
    elif args.data == 'animal10n':
        args.num_classes = 10
        
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.pretrained = True if args.pretrained==1 else False
    args.lr_schedule = args.lr_schedule.lower()

    main(args)
    
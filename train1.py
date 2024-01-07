import os
import math
import argparse
import random
import logging
from tqdm import tqdm
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel
from data.data_sampler import DistIterSampler


import options.options as option
from utils import util
from utils.loss import loss_fun,VGGLoss1,CharbonnierLoss
from adv.pgd import attack_pgd
from adv.rand import attack_random
from data import create_dataloader, create_dataset

from models.enhance_net import LIE
import models.lr_scheduler as lr_scheduler
import numpy as np
import cv2

def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)



def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.', default='./options/train/LOLv1.yml')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch',
                        help='job launcher')
    parser.add_argument("--adv_type", type=str, default='pgd', choices=['random', 'pgd'], 
                        help='type of adv noises: random or pgd')
    parser.add_argument("--adv_lambda", type=float, default=0.5, help='lambda coefficient of adv loss')
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()

    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            t = time.localtime()
            current_time = time.strftime("%Y%m%d%H%M%S", t)
            tb_logger = SummaryWriter(log_dir='./tb_logger/LOL1_alph0.1_beta_0.5' + current_time)
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 50  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)

            # print(train_set[0])
            # import pdb; pdb.set_trace()

            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['mode'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = LIE()

    # define network and load pretrained models
    model = model.to('cuda')
    if opt['dist']:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    else:
        model = DataParallel(model)
    #### resume training
    # if resume_state:
    #     logger.info('Resuming training from epoch: {}, iter: {}.'.format(
    #         resume_state['epoch'], resume_state['iter']))

    #     start_epoch = resume_state['epoch']
    #     current_step = resume_state['iter']
    #     model.resume_training(resume_state)  # handle optimizers and schedulers
    #     del resume_state
    # else:
    #     current_step = 0
    #     start_epoch = 0


    train_opt = opt['train']
    wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
    optimizer_G = torch.optim.Adam(model.parameters(), lr=train_opt['lr_G'],
                                        weight_decay=wd_G,
                                        betas=(train_opt['beta1'], train_opt['beta2']))
    
    if train_opt['lr_scheme'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR_Restart(optimizer_G, train_opt['lr_steps'],
                                         restarts=train_opt['restarts'],
                                         weights=train_opt['restart_weights'],
                                         gamma=train_opt['lr_gamma'],
                                         clear_state=train_opt['clear_state'])

    else:
        scheduler = lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer_G, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights'])

    #### training
    
    ## Attack Parameters
    epsilons = 1e-1
    alphas = 1e-4
    pgd_iters = 1
    cri_pix = torch.nn.MSELoss(reduction='mean')
    cri_pix1 = CharbonnierLoss() #torch.nn.L1Loss(reduction='mean')
    cri_vgg = VGGLoss1()
    start_epoch=0
    current_step=0
    best_psnr = 0
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        epoch_gamma = []
        with tqdm(train_loader) as t:
            for _, train_data in enumerate(t):
                current_step += 1
                if current_step > total_iters:
                    break
                
                #### Feedforward
                ll_img = train_data['LQs'].to('cuda')
                gt = train_data['GT'].to('cuda')
                
                # 1. Obtain adversarial sample
                #model.eval()
                #Adv perturb on image gamma
                ## Gamaa transformation
                #model, X, y, epsilon, alpha, loss_fn, attack_iters
                if args.adv_type == 'pgd': #projected gradient descent
                    adv_perturb = attack_pgd(model, ll_img, gt, epsilons, 
                                             alphas, loss_fn=cri_pix, attack_iters=pgd_iters)
                elif args.adv_type == 'random':
                    adv_perturb = attack_random(model, ll_img, gt, epsilons)
                
                # 2. Inject adversarial sample
                model.train()
                #print(adv_perturb)
                #epoch_gamma.append(adv_perturb.detach().cpu().squeeze().numpy())
                #logger.info('ADV_PERTURB: {:d}'.format(adv_perturb.item()))
                
                ret_adv = model(torch.pow(ll_img,adv_perturb)) # with gamma adversarial
                ret_clean = model(ll_img) # without adversarial

                optimizer_G.zero_grad()
                
                #### loss and backward
                loss_clean  = cri_pix1(ret_clean, gt)+0.1*cri_vgg(ret_clean, gt)
                loss_adv = cri_pix1(ret_adv, gt)+0.1*cri_vgg(ret_adv, gt)
                l_final = (1.0 - args.adv_lambda) * loss_clean + args.adv_lambda * loss_adv
                #l_final = loss_fun(ret_adv,gt)
                l_final.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 10, norm_type=2)
                optimizer_G.step()
                
                #### update learning rate
                scheduler.step()
                #### log
                if current_step % opt['logger']['print_freq'] == 0:
                    
                    message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                    ## show learning rate
                    message += '{:.3e},'.format(optimizer_G.param_groups[0]['lr'])
                    message += ')] '
                    
                    message += '{:s}: {:.4e} '.format('all_loss', l_final.item())
                    message += '{:s}: {:.4e} '.format('clean_loss', loss_clean.item())
                    message += '{:s}: {:.4e} '.format('adv_loss', loss_adv.item())
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar('loss', l_final.item(), current_step)
                    if rank <= 0:
                        logger.info(message)

                #### validation
                if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                    model.eval()
                    if opt['dist']:
                        # multi-GPU testing
                        if rank == 0:
                            psnr_list=[]
                            for valid_data in val_loader:

                                ## random index test
                                #idx = random.randint(0, len(val_set)-1)
                                val_data = valid_data

                                # tmp = torch.zeros(max_idx, dtype=torch.float32, device='cuda')
                                LQ = val_data['LQs'].to('cuda')
                                GT = val_data['GT'].to('cuda')
                                
                                rec = model(LQ)
        
                                sou_img = util.tensor2img(LQ)
                                rlt_img = util.tensor2img(rec.detach())  # uint8
        
                                gt_img = util.tensor2img(GT)  # uint8
        
        
                                save_img = np.concatenate([sou_img, rlt_img, gt_img], axis=0)
                                im_path = os.path.join(opt['path']['val_images'], '%06d.png' % current_step)
                                cv2.imwrite(im_path, save_img.astype(np.uint8))
                                
        
                                # calculate PSNR
                                psnr = util.calculate_psnr(rlt_img, gt_img)
                                
                                psnr_list.append(psnr)

                            # # collect data
                            avg_psnr = sum(psnr_list)/len(psnr_list)
                            log_s = '# Validation # PSNR: {:.4e}:'.format(avg_psnr)
                            logger.info(log_s)
                            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                                tb_logger.add_scalar('psnr_avg', avg_psnr, current_step)

                            if avg_psnr > best_psnr:
                                best_psnr = avg_psnr
                                logger.info('Saving models and training states.')
                                path = opt['path']['experiments_root']+'/models/best.pth'
                                save_dict = {
                                    'global_step': current_step,
                                    'model': model.state_dict(),
                                    'optimizer': optimizer_G.state_dict()
                                }
                                torch.save(save_dict, path)

                    else:
                        pbar = util.ProgressBar(len(val_loader))
                        psnr_rlt = {}  # with border and center frames
                        psnr_rlt_avg = {}
                        psnr_total_avg = 0.
                        for val_data in val_loader:
                            folder = val_data['folder'][0]
                            idx_d = val_data['idx'].item()
                            # border = val_data['border'].item()
                            if psnr_rlt.get(folder, None) is None:
                                psnr_rlt[folder] = []

                            LQ = val_data['LQs'].to('cuda')
                            GT = val_data['GT'].to('cuda')
                            
                            rec = model(LQ)
                            rlt_img = util.tensor2img(rec)  # uint8
                            gt_img = util.tensor2img(GT)  # uint8

                            # calculate PSNR
                            psnr = util.calculate_psnr(rlt_img, gt_img)
                            psnr_rlt[folder].append(psnr)
                            pbar.update('Test {} - {}'.format(folder, idx_d))
                        for k, v in psnr_rlt.items():
                            psnr_rlt_avg[k] = sum(v) / len(v)
                            psnr_total_avg += psnr_rlt_avg[k]
                        psnr_total_avg /= len(psnr_rlt)
                        log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                        for k, v in psnr_rlt_avg.items():
                            log_s += ' {}: {:.4e}'.format(k, v)
                        logger.info(log_s)
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                            for k, v in psnr_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)
                        
                        if psnr_total_avg > best_psnr:
                            best_psnr = psnr_total_avg
                            logger.info('Saving models and training states.')
                            path = opt['path']['experiments_root']+'/models/best.pth'
                            save_dict = {
                                'global_step': current_step,
                                'model': model.state_dict(),
                                'optimizer': optimizer_G.state_dict()
                            }
                            torch.save(save_dict, path)
    
                #### save models and training states
                if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                    if rank <= 0:
                        logger.info('Saving models and training states.')
                        path = opt['path']['experiments_root']+'/models/'+str(current_step)+'.pth'
                        save_dict = {
                            'global_step': current_step,
                            'model': model.state_dict(),
                            'optimizer': optimizer_G.state_dict()
                        }
                        torch.save(save_dict, path)

                if current_step % 500 == 0:
                    name = "gamma_"+str(current_step)+'.txt'
                    #np.savetxt(name, np.array(epoch_gamma))
    if rank <= 0:
        logger.info('Saving the final model.')
        path = opt['path']['experiments_root']+'/models/final.pth'
        save_dict = {
            'global_step': current_step,
            'model': model.state_dict(),
            'optimizer': optimizer_G.state_dict()
        }
        torch.save(save_dict, path)
        logger.info('End of training.')
        tb_logger.close()

def save_network(self, network, network_label, iter_label):
    save_filename = '{}_{}.pth'.format(iter_label, network_label)
    save_path = os.path.join(self.opt['path']['models'], save_filename)
    if isinstance(network, torch.nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)


if __name__ == '__main__':
    main()

import os
import os.path as osp

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import grad 

import numpy as np
import random
import wandb
from tqdm import tqdm
from PIL import Image
from packaging import version
from datetime import datetime

from model.refinenetlw import rf_lw101
from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1
from utils.losses import CrossEntropy2d
from dataset.paired_cityscapes import Pairedcityscapes
from dataset.Foggy_Zurich import foggyzurichDataSet
from configs.train_config import get_arguments
from utils.optimisers import get_optimisers, get_lr_schedulers
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
RESTORE_FROM = 'without_pretraining'
RESTORE_FROM_fogpass = 'without_pretraining'

def loss_calc(pred, label, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)
    return criterion(pred, label)

def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def fft_log_mag(x):
    """Log-magnitude of FFT for frequency consistency."""
    ffted = torch.fft.rfft2(x, norm='ortho')
    mag = torch.sqrt(ffted.real ** 2 + ffted.imag ** 2 + 1e-8)
    return torch.log1p(mag)


def frequency_consistency_loss(a, b):
    return F.l1_loss(fft_log_mag(a), fft_log_mag(b))

def setup_optimisers_and_schedulers(args, model):
    optimisers = get_optimisers(
        model=model,
        enc_optim_type="sgd",
        enc_lr=6e-4,
        enc_weight_decay=1e-5,
        enc_momentum=0.9,
        dec_optim_type="sgd",
        dec_lr=6e-3,
        dec_weight_decay=1e-5,
        dec_momentum=0.9,
    )
    schedulers = get_lr_schedulers(
        enc_optim=optimisers[0],
        dec_optim=optimisers[1],
        enc_lr_gamma=0.5,
        dec_lr_gamma=0.5,
        enc_scheduler_type="multistep",
        dec_scheduler_type="multistep",
        epochs_per_stage=(100, 100, 100),
    )
    return optimisers, schedulers

def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]

def main():
    """Create the model and start the training."""

    args = get_arguments()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{args.file_name}-{now}'

    wandb.init(project='FIFO',name=f'{run_name}')
    wandb.config.update(args)

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w_r, h_r = map(int, args.input_size_rf.split(',')) 
    input_size_rf = (w_r, h_r)   

    cudnn.enabled = True
    gpu = args.gpu

    if args.restore_from == RESTORE_FROM:
        start_iter = 0
        model = rf_lw101(num_classes=args.num_classes)
    else:
        restore = torch.load(args.restore_from, weights_only=False)
        model = rf_lw101(num_classes=args.num_classes)
        model.load_state_dict(restore['state_dict'])
        start_iter = 0

    teacher_model = None
    if args.restore_from_teacher:
        teacher_ckpt = torch.load(args.restore_from_teacher, weights_only=False)
        teacher_model = rf_lw101(num_classes=args.num_classes)
        teacher_model.load_state_dict(teacher_ckpt['state_dict'])
        teacher_model.eval()

    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if teacher_model is not None:
            teacher_model = nn.DataParallel(teacher_model)

    model.train()
    model.cuda(args.gpu)
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    if teacher_model is not None:
        teacher_model.cuda(args.gpu)

    lr_fpf1 = 1e-3 
    lr_fpf2 = 1e-3

    if args.modeltrain=='train':
        lr_fpf1 = args.fogpass_lr_ft
        lr_fpf2 = args.fogpass_lr_ft

    FogPassFilter1 = FogPassFilter_conv1(2080)
    FogPassFilter1_optimizer = torch.optim.Adamax([p for p in FogPassFilter1.parameters() if p.requires_grad == True], lr=lr_fpf1)
    FogPassFilter1.cuda(args.gpu)
    FogPassFilter2 = FogPassFilter_res1(32896)
    FogPassFilter2_optimizer = torch.optim.Adamax([p for p in FogPassFilter2.parameters() if p.requires_grad == True], lr=lr_fpf2)
    FogPassFilter2.cuda(args.gpu)

    if args.restore_from_fogpass != RESTORE_FROM_fogpass:
        restore = torch.load(args.restore_from_fogpass, weights_only=False)
        FogPassFilter1.load_state_dict(restore['fogpass1_state_dict'])
        FogPassFilter2.load_state_dict(restore['fogpass2_state_dict'])

    fogpassfilter_loss = losses.ContrastiveLoss(
        pos_margin=0.1,
        neg_margin=0.1,
        distance=CosineSimilarity(),
        reducer=MeanReducer()
        )

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    cwsf_pair_loader = data.DataLoader(Pairedcityscapes(args.data_dir, args.data_dir_cwsf, args.data_list, args.data_list_cwsf,
                                        max_iters=args.num_steps * args.iter_size * args.batch_size,
                                        mean=IMG_MEAN, set=args.set), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                        pin_memory=True)

    rf_loader = data.DataLoader(foggyzurichDataSet(args.data_dir_rf, args.data_list_rf,
                                            max_iters=args.num_steps * args.iter_size * args.batch_size,
                                            mean=IMG_MEAN, set=args.set),
                                            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                            pin_memory=True)

    cwsf_pair_loader_fogpass = data.DataLoader(Pairedcityscapes(args.data_dir, args.data_dir_cwsf, args.data_list, args.data_list_cwsf,
                                                max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                mean=IMG_MEAN, set=args.set), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                                pin_memory=True)

    rf_loader_fogpass = data.DataLoader(foggyzurichDataSet(args.data_dir_rf, args.data_list_rf,
                                                    max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                    mean=IMG_MEAN, set=args.set), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                                    pin_memory=True)

    rf_loader_iter = enumerate(rf_loader)
    cwsf_pair_loader_iter = enumerate(cwsf_pair_loader)
    cwsf_pair_loader_iter_fogpass = enumerate(cwsf_pair_loader_fogpass)
    rf_loader_iter_fogpass = enumerate(rf_loader_fogpass)

    optimisers, schedulers = setup_optimisers_and_schedulers(args, model=base_model)
    opts = make_list(optimisers)
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    m = nn.Softmax(dim=1)
    log_m = nn.LogSoftmax(dim=1)    

    # Gradient accumulation counter
    accum_step = 0

    for i_iter in tqdm(range(start_iter, args.num_steps)): 
        train_fogpass = i_iter >= args.fogpass_unfreeze_iter

        if not train_fogpass:
            for param in FogPassFilter1.parameters():
                param.requires_grad = False
            for param in FogPassFilter2.parameters():
                param.requires_grad = False
        else:
            for param in FogPassFilter1.parameters():
                param.requires_grad = True
            for param in FogPassFilter2.parameters():
                param.requires_grad = True
            if i_iter == args.fogpass_unfreeze_iter:
                for group in FogPassFilter1_optimizer.param_groups:
                    group['lr'] = args.fogpass_lr_ft
                for group in FogPassFilter2_optimizer.param_groups:
                    group['lr'] = args.fogpass_lr_ft

        loss_seg_cw_value = 0
        loss_seg_sf_value = 0
        loss_fsm_value = 0
        loss_pl_value = 0

        # Zero gradients only at the start of accumulation cycle
        if accum_step == 0:
            for opt in opts:
                opt.zero_grad()
            if train_fogpass:
                FogPassFilter1_optimizer.zero_grad()
                FogPassFilter2_optimizer.zero_grad()

        for sub_i in range(args.iter_size):
            total_fpf_loss = 0

            if train_fogpass:
                # train fog-pass filtering module
                # freeze the parameters of segmentation network

                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                for param in FogPassFilter1.parameters():
                    param.requires_grad = True
                for param in FogPassFilter2.parameters():
                    param.requires_grad = True
  
                _, batch = cwsf_pair_loader_iter_fogpass.__next__()
                sf_image, cw_image, label, size, sf_name, cw_name = batch
                interp = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')
                
                _, batch_rf = rf_loader_iter_fogpass.__next__()
                rf_img,rf_size, rf_name = batch_rf
                img_rf = Variable(rf_img).cuda(args.gpu)
                feature_rf0, feature_rf1, feature_rf2, feature_rf3, feature_rf4, feature_rf5 = model(img_rf) 

                images = Variable(sf_image).cuda(args.gpu)
                feature_sf0,feature_sf1,feature_sf2, feature_sf3,feature_sf4,feature_sf5 = model(images)

                images_cw = Variable(cw_image).cuda(args.gpu)
                feature_cw0, feature_cw1, feature_cw2, feature_cw3, feature_cw4, feature_cw5 = model(images_cw)

                fsm_weights = {'layer0':0.5, 'layer1':0.5}
                sf_features = {'layer0':feature_sf0, 'layer1':feature_sf1}                
                cw_features = {'layer0':feature_cw0, 'layer1':feature_cw1}
                rf_features = {'layer0':feature_rf0, 'layer1':feature_rf1}

                for idx, layer in enumerate(fsm_weights):
                    cw_feature = cw_features[layer]
                    sf_feature = sf_features[layer]    
                    rf_feature = rf_features[layer]      
                    fog_pass_filter_loss = 0 
                    
                    if idx == 0:
                        fogpassfilter = FogPassFilter1
                        fogpassfilter_optimizer = FogPassFilter1_optimizer
                    elif idx == 1:
                        fogpassfilter = FogPassFilter2
                        fogpassfilter_optimizer = FogPassFilter2_optimizer

                    fogpassfilter.train()  
                    fogpassfilter_optimizer.zero_grad()
                    
                    sf_gram = [0]*args.batch_size
                    cw_gram = [0]*args.batch_size
                    rf_gram = [0]*args.batch_size 
                    vector_sf_gram = [0]*args.batch_size
                    vector_cw_gram = [0]*args.batch_size
                    vector_rf_gram  = [0]*args.batch_size
                    fog_factor_sf = [0]*args.batch_size
                    fog_factor_cw = [0]*args.batch_size
                    fog_factor_rf = [0]*args.batch_size

                    for batch_idx in range(args.batch_size):
                        sf_gram[batch_idx] = gram_matrix(sf_feature[batch_idx])
                        cw_gram[batch_idx] = gram_matrix(cw_feature[batch_idx])
                        rf_gram[batch_idx] = gram_matrix(rf_feature[batch_idx])

                        vector_sf_gram[batch_idx] = Variable(sf_gram[batch_idx][torch.triu(torch.ones(sf_gram[batch_idx].size()[0], sf_gram[batch_idx].size()[1])) == 1], requires_grad=True)
                        vector_cw_gram[batch_idx] = Variable(cw_gram[batch_idx][torch.triu(torch.ones(cw_gram[batch_idx].size()[0], cw_gram[batch_idx].size()[1])) == 1], requires_grad=True)
                        vector_rf_gram[batch_idx] = Variable(rf_gram[batch_idx][torch.triu(torch.ones(rf_gram[batch_idx].size()[0], rf_gram[batch_idx].size()[1])) == 1], requires_grad=True)

                        fog_factor_sf[batch_idx] = fogpassfilter(vector_sf_gram[batch_idx])
                        fog_factor_cw[batch_idx] = fogpassfilter(vector_cw_gram[batch_idx])
                        fog_factor_rf[batch_idx] = fogpassfilter(vector_rf_gram[batch_idx])                                                                                                                                                                                                

                    # Dynamically build fog_factor_embeddings based on actual batch size
                    fog_factor_list = []
                    fog_factor_labels_list = []
                    for batch_idx in range(args.batch_size):
                        fog_factor_list.extend([
                            torch.unsqueeze(fog_factor_sf[batch_idx], 0),
                            torch.unsqueeze(fog_factor_cw[batch_idx], 0),
                            torch.unsqueeze(fog_factor_rf[batch_idx], 0)
                        ])
                        fog_factor_labels_list.extend([0, 1, 2])
                    
                    fog_factor_embeddings = torch.cat(fog_factor_list, 0)
                    fog_factor_embeddings_norm = torch.norm(fog_factor_embeddings, p=2, dim=1).detach()
                    size_fog_factor = fog_factor_embeddings.size()
                    fog_factor_embeddings = fog_factor_embeddings.div(fog_factor_embeddings_norm.expand(size_fog_factor[1], args.batch_size * 3).t())
                    fog_factor_labels = torch.LongTensor(fog_factor_labels_list)
                    fog_pass_filter_loss = fogpassfilter_loss(fog_factor_embeddings,fog_factor_labels)

                    total_fpf_loss +=  fog_pass_filter_loss 
                  
                    wandb.log({f'layer{idx}/fpf loss': fog_pass_filter_loss}, step=i_iter)
                    wandb.log({f'layer{idx}/total fpf loss': total_fpf_loss}, step=i_iter)

                # Scale fog-pass filter loss by accumulation steps
                (total_fpf_loss / args.accum_steps).backward(retain_graph=False)


            if args.modeltrain=='train':
                # train segmentation network
                # freeze the parameters of fog pass filtering modules

                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                for param in FogPassFilter1.parameters():
                    param.requires_grad = False
                for param in FogPassFilter2.parameters():
                    param.requires_grad = False

                _, batch = cwsf_pair_loader_iter.__next__()
                sf_image, cw_image, label, size, sf_name, cw_name = batch

                interp = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')

                loss_pl = 0

                if i_iter % 3 == 0:
                    images_sf = Variable(sf_image).cuda(args.gpu)
                    feature_sf0,feature_sf1,feature_sf2, feature_sf3,feature_sf4,feature_sf5 = model(images_sf)
                    pred_sf5 = interp(feature_sf5)
                    loss_seg_sf = loss_calc(pred_sf5, label, args.gpu)
                    images_cw = Variable(cw_image).cuda(args.gpu)
                    feature_cw0, feature_cw1, feature_cw2, feature_cw3, feature_cw4, feature_cw5 = model(images_cw)
                    pred_cw5 = interp(feature_cw5)
                    feature_cw5_logsoftmax = log_m(feature_cw5)
                    feature_sf5_softmax = m(feature_sf5)
                    feature_sf5_logsoftmax = log_m(feature_sf5)
                    feature_cw5_softmax = m(feature_cw5)
                    loss_con = kl_loss(feature_sf5_logsoftmax, feature_cw5_softmax)
                    loss_seg_cw = loss_calc(pred_cw5, label, args.gpu)     
                    fsm_weights = {'layer0':0.5, 'layer1':0.5}
                    sf_features = {'layer0':feature_sf0, 'layer1':feature_sf1}                
                    cw_features = {'layer0':feature_cw0, 'layer1':feature_cw1}
                    loss_freq = frequency_consistency_loss(feature_sf5, feature_cw5)

                if i_iter % 3 == 1:
                    _, batch_rf = rf_loader_iter.__next__()
                    rf_img,rf_size, rf_name = batch_rf
                    images_sf = Variable(sf_image).cuda(args.gpu)
                    feature_sf0,feature_sf1,feature_sf2, feature_sf3,feature_sf4,feature_sf5 = model(images_sf)
                    pred_sf5 = interp(feature_sf5)
                    loss_seg_sf = loss_calc(pred_sf5, label, args.gpu)       
                    loss_seg_cw = 0   
                    loss_con = 0
                    img_rf = Variable(rf_img).cuda(args.gpu)
                    feature_rf0, feature_rf1, feature_rf2, feature_rf3, feature_rf4, feature_rf5 = model(img_rf)    
                    rf_features = {'layer0':feature_rf0, 'layer1':feature_rf1}
                    sf_features = {'layer0':feature_sf0, 'layer1':feature_sf1}
                    fsm_weights = {'layer0':0.5, 'layer1':0.5}
                    loss_freq = frequency_consistency_loss(feature_sf5, feature_rf5)
                    # Pseudo-label loss on RF images (teacher-based)
                    loss_pl = 0
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_pred = teacher_model(img_rf)[-1]
                            teacher_pred_up = interp(teacher_pred)
                            teacher_prob = m(teacher_pred_up)
                            conf, pseudo = torch.max(teacher_prob, dim=1)
                            mask = conf >= args.pl_threshold
                            pseudo_masked = pseudo.clone()
                            pseudo_masked[~mask] = 255
                        loss_pl = loss_calc(pred_sf5, pseudo_masked, args.gpu) if mask.any() else 0
                
                if i_iter % 3 == 2:
                    _, batch_rf = rf_loader_iter.__next__()
                    rf_img,rf_size, rf_name = batch_rf
                    images_cw = Variable(cw_image).cuda(args.gpu)
                    feature_cw0, feature_cw1, feature_cw2, feature_cw3, feature_cw4, feature_cw5 = model(images_cw)
                    pred_cw5 = interp(feature_cw5)
                    loss_seg_sf = 0
                    loss_con = 0
                    loss_seg_cw = loss_calc(pred_cw5, label, args.gpu)      
                    img_rf = Variable(rf_img).cuda(args.gpu)
                    feature_rf0, feature_rf1, feature_rf2, feature_rf3, feature_rf4, feature_rf5 = model(img_rf)                  
                    rf_features = {'layer0':feature_rf0, 'layer1':feature_rf1}
                    cw_features = {'layer0':feature_cw0, 'layer1':feature_cw1}
                    fsm_weights = {'layer0':0.5, 'layer1':0.5}
                    loss_freq = frequency_consistency_loss(feature_cw5, feature_rf5)
                    # Pseudo-label loss on RF images (teacher-based)
                    loss_pl = 0
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_pred = teacher_model(img_rf)[-1]
                            teacher_pred_up = interp(teacher_pred)
                            teacher_prob = m(teacher_pred_up)
                            conf, pseudo = torch.max(teacher_prob, dim=1)
                            mask = conf >= args.pl_threshold
                            pseudo_masked = pseudo.clone()
                            pseudo_masked[~mask] = 255
                        loss_pl = loss_calc(pred_cw5, pseudo_masked, args.gpu) if mask.any() else 0

                loss_fsm = 0
                fog_pass_filter_loss = 0

                for idx, layer in enumerate(fsm_weights):
                    # fog pass filter loss between different fog conditions a and b
                    if i_iter % 3 == 0:
                        a_feature = cw_features[layer]
                        b_feature = sf_features[layer]    
                    if i_iter % 3 == 1:
                        a_feature = rf_features[layer]
                        b_feature = sf_features[layer]
                    if i_iter % 3 == 2:
                        a_feature = rf_features[layer]
                        b_feature = cw_features[layer]   

                    layer_fsm_loss = 0
                    fog_pass_filter_loss = 0   
                    na,da,ha,wa = a_feature.size()
                    nb,db,hb,wb = b_feature.size()

                    if idx == 0:
                        fogpassfilter = FogPassFilter1
                        fogpassfilter_optimizer = FogPassFilter1_optimizer
                    elif idx == 1:
                        fogpassfilter = FogPassFilter2
                        fogpassfilter_optimizer = FogPassFilter2_optimizer

                    fogpassfilter.eval()

                    for batch_idx in range(args.batch_size):
                        b_gram = gram_matrix(b_feature[batch_idx])
                        a_gram = gram_matrix(a_feature[batch_idx])

                        if i_iter % 3 == 1 or i_iter % 3 == 2:
                            a_gram = a_gram *(hb*wb)/(ha*wa)

                        vector_b_gram = b_gram[torch.triu(torch.ones(b_gram.size()[0], b_gram.size()[1])).requires_grad_() == 1].requires_grad_()
                        vector_a_gram = a_gram[torch.triu(torch.ones(a_gram.size()[0], a_gram.size()[1])).requires_grad_() == 1].requires_grad_()

                        fog_factor_b = fogpassfilter(vector_b_gram)
                        fog_factor_a = fogpassfilter(vector_a_gram)
                        half = int(fog_factor_b.shape[0]/2)
                        
                        layer_fsm_loss += fsm_weights[layer]*torch.mean((fog_factor_b/(hb*wb) - fog_factor_a/(ha*wa))**2)/half/ b_feature.size(0)

                    loss_fsm += layer_fsm_loss / args.batch_size

                loss_pl_term = loss_pl if 'loss_pl' in locals() else 0
                loss = loss_seg_sf + loss_seg_cw + args.lambda_fsm*loss_fsm + args.lambda_con*loss_con + args.lambda_freq*loss_freq + args.lambda_pl*loss_pl_term
                # Scale loss by both iter_size and accumulation steps
                loss = loss / (args.iter_size * args.accum_steps)
                loss.backward()

                if loss_seg_cw != 0:
                    loss_seg_cw_value += loss_seg_cw.data.cpu().numpy() / args.iter_size
                if loss_seg_sf != 0:
                    loss_seg_sf_value += loss_seg_sf.data.cpu().numpy() / args.iter_size
                if loss_fsm != 0:
                    loss_fsm_value += loss_fsm.data.cpu().numpy() / args.iter_size
                if loss_con != 0:
                    loss_con_value += loss_con.data.cpu().numpy() / args.iter_size
                if loss_freq != 0:
                    loss_freq_value += loss_freq.data.cpu().numpy() / args.iter_size
                if loss_pl_term != 0:
                    loss_pl_value += (loss_pl_term.data.cpu().numpy() if hasattr(loss_pl_term, 'data') else 0) / args.iter_size

            
                wandb.log({"fsm loss": args.lambda_fsm*loss_fsm_value}, step=i_iter)
                wandb.log({'SF_loss_seg': loss_seg_sf_value}, step=i_iter)
                wandb.log({'CW_loss_seg': loss_seg_cw_value}, step=i_iter)
                wandb.log({'consistency loss':args.lambda_con*loss_con_value}, step=i_iter)
                wandb.log({'frequency loss':args.lambda_freq*loss_freq_value}, step=i_iter)
                wandb.log({'pseudo loss':args.lambda_pl*loss_pl_value}, step=i_iter)
                wandb.log({'total_loss': loss}, step=i_iter)
        
        # Increment accumulation step counter
        accum_step += 1
        
        # Only update weights after accumulating enough gradients
        if accum_step >= args.accum_steps:
            accum_step = 0  # Reset counter
            
            # Update all optimizers
            if args.modeltrain == 'train':
                for opt in opts:
                    opt.step()

            if train_fogpass:
                FogPassFilter1_optimizer.step()
                FogPassFilter2_optimizer.step()

        if i_iter < 20000:
            save_pred_every = 5000
            if args.modeltrain=='train':
                save_pred_every = 2000
        else:
            save_pred_every = args.save_pred_every

        if i_iter >= args.num_steps_stop - 1:
            print('save model ..')
            torch.save(base_model.state_dict(), osp.join(args.snapshot_dir, args.file_name + str(args.num_steps_stop) + '.pth'))
            break
        if args.modeltrain != 'train':
            if i_iter == 5000:
                torch.save({'state_dict':base_model.state_dict(),
                'fogpass1_state_dict':FogPassFilter1.state_dict(),
                'fogpass2_state_dict':FogPassFilter2.state_dict(),
                'train_iter':i_iter,
                'args':args
                },osp.join(args.snapshot_dir, run_name)+'_fogpassfilter_'+str(i_iter)+'.pth')

        if i_iter % save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            print(f'Step {i_iter} - SF_loss: {loss_seg_sf_value:.4f}, CW_loss: {loss_seg_cw_value:.4f}, FSM_loss: {args.lambda_fsm*loss_fsm_value:.6f}, Consistency_loss: {args.lambda_con*loss_con_value:.6f}, Freq_loss: {args.lambda_freq*loss_freq_value:.6f}, PL_loss: {args.lambda_pl*loss_pl_value:.6f}')
            save_dir = osp.join(f'./result/FIFO_model', args.file_name)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save({
                'state_dict':base_model.state_dict(),
                'fogpass1_state_dict':FogPassFilter1.state_dict(),
                'fogpass2_state_dict':FogPassFilter2.state_dict(),
                'train_iter':i_iter,
                'args':args
            },osp.join(args.snapshot_dir, run_name)+'_FIFO'+str(i_iter)+'.pth')
            
if __name__ == '__main__':
    main()
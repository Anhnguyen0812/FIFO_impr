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
import matplotlib.pyplot as plt
import subprocess

from model.refinenetlw import rf_lw101
from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1
from model.boundary_head import BoundaryHead, generate_boundary_label
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

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in boundary detection.
    
    Args:
        alpha: Weighting factor in [0, 1] to balance positive/negative examples
        gamma: Exponent of modulating factor (1 - p_t)^gamma
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        # pred: [B, 1, H, W] logits
        # target: [B, 1, H, W] binary labels (0 or 1)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)  # probability of correct class
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

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


    model.train()
    model.cuda(args.gpu)

    lr_fpf1 = 1e-3 
    lr_fpf2 = 1e-3

    if args.modeltrain=='train':
        lr_fpf1 = 5e-4

    FogPassFilter1 = FogPassFilter_conv1(2080)
    FogPassFilter1_optimizer = torch.optim.Adamax([p for p in FogPassFilter1.parameters() if p.requires_grad == True], lr=lr_fpf1)
    FogPassFilter1.cuda(args.gpu)
    FogPassFilter2 = FogPassFilter_res1(32896)
    FogPassFilter2_optimizer = torch.optim.Adamax([p for p in FogPassFilter2.parameters() if p.requires_grad == True], lr=lr_fpf2)
    FogPassFilter2.cuda(args.gpu)

    # Initialize Boundary Detection Head
    # out1 (conv1): 64 channels, out2 (layer1): 256 channels
    BoundaryHead_model = BoundaryHead(in_channels_low=64, in_channels_mid=256, out_channels=1)
    BoundaryHead_model.train()
    BoundaryHead_model.cuda(args.gpu)
    BoundaryHead_optimizer = torch.optim.Adam(BoundaryHead_model.parameters(), lr=1e-3)
    
    # Boundary loss (Focal Loss for better handling of boundary/non-boundary imbalance)
    boundary_criterion = FocalLoss(alpha=0.25, gamma=2.0)

    if args.restore_from_fogpass != RESTORE_FROM_fogpass:
        restore = torch.load(args.restore_from_fogpass, weights_only=False)
        FogPassFilter1.load_state_dict(restore['fogpass1_state_dict'])
        FogPassFilter2.load_state_dict(restore['fogpass2_state_dict'])
        if 'boundary_state_dict' in restore:
            BoundaryHead_model.load_state_dict(restore['boundary_state_dict'])

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

    optimisers, schedulers = setup_optimisers_and_schedulers(args, model=model)
    opts = make_list(optimisers)
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    m = nn.Softmax(dim=1)
    log_m = nn.LogSoftmax(dim=1)    

    # Warmup for boundary loss to let segmentation stabilise first
    boundary_warmup_iters = 3000
    
    # Gradient accumulation counter
    accum_step = 0

    # Loss history for plotting
    loss_history = {'total': [], 'seg_sf': [], 'seg_cw': [], 'fsm': [], 'con': [], 'boundary': []}

    for i_iter in tqdm(range(start_iter, args.num_steps)): 
        loss_seg_cw_value = 0
        loss_seg_sf_value = 0
        loss_fsm_value = 0
        loss_con_value = 0
        loss_boundary_value = 0

        # Zero gradients only at the start of accumulation cycle
        if accum_step == 0:
            for opt in opts:
                opt.zero_grad()
            if i_iter >= boundary_warmup_iters:
                BoundaryHead_optimizer.zero_grad()
            FogPassFilter1_optimizer.zero_grad()
            FogPassFilter2_optimizer.zero_grad()

        for sub_i in range(args.iter_size):
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

            total_fpf_loss = 0

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

                # Boundary Detection Loss (Multi-task Learning)
                loss_boundary = 0
                boundary_weight = 0.0

                # Adaptive boundary weight schedule
                if i_iter < boundary_warmup_iters:
                    boundary_weight = 0.0  # Warmup: no boundary loss
                elif i_iter < 6000:
                    boundary_weight = args.lambda_boundary  # Initial weight (0.01)
                else:
                    boundary_weight = min(args.lambda_boundary * 3, 0.03)  # Increase after 6000 (up to 0.03)

                # Enable boundary loss only after warmup
                if i_iter >= boundary_warmup_iters:
                    BoundaryHead_model.train()

                    # Generate boundary ground truth from segmentation labels
                    label_tensor = label.cuda(args.gpu)  # [B, H, W]
                    boundary_gt = generate_boundary_label(label_tensor)  # [B, 1, H, W]

                    # Compute boundary loss with both SF and CW features + consistency
                    if i_iter % 3 == 0:
                        # Both SF and CW available: supervise both + add consistency
                        boundary_pred_cw = BoundaryHead_model(feature_cw0, feature_cw1)
                        boundary_pred_sf = BoundaryHead_model(feature_sf0, feature_sf1)
                        
                        boundary_gt_resized = F.interpolate(boundary_gt, size=boundary_pred_cw.shape[2:], mode='bilinear', align_corners=True)
                        
                        # GT loss: CW (weight 1.0) + SF (weight 0.5)
                        loss_boundary_cw = boundary_criterion(boundary_pred_cw, boundary_gt_resized)
                        loss_boundary_sf = boundary_criterion(boundary_pred_sf, boundary_gt_resized)
                        
                        # Consistency loss: SF boundary should match CW boundary (use CW as teacher)
                        loss_boundary_consistency = F.mse_loss(
                            torch.sigmoid(boundary_pred_sf),
                            torch.sigmoid(boundary_pred_cw).detach()
                        )
                        
                        # Combined boundary loss
                        loss_boundary = loss_boundary_cw + 0.5 * loss_boundary_sf + 0.3 * loss_boundary_consistency
                        
                    elif i_iter % 3 == 1:
                        # Only SF available
                        boundary_pred_sf = BoundaryHead_model(feature_sf0, feature_sf1)
                        boundary_gt_resized = F.interpolate(boundary_gt, size=boundary_pred_sf.shape[2:], mode='bilinear', align_corners=True)
                        loss_boundary = 0.5 * boundary_criterion(boundary_pred_sf, boundary_gt_resized)
                        
                    elif i_iter % 3 == 2:
                        # Only CW available
                        boundary_pred_cw = BoundaryHead_model(feature_cw0, feature_cw1)
                        boundary_gt_resized = F.interpolate(boundary_gt, size=boundary_pred_cw.shape[2:], mode='bilinear', align_corners=True)
                        loss_boundary = boundary_criterion(boundary_pred_cw, boundary_gt_resized)

                loss = loss_seg_sf + loss_seg_cw + args.lambda_fsm*loss_fsm + args.lambda_con*loss_con + boundary_weight*loss_boundary
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
                if loss_boundary != 0:
                    loss_boundary_value += loss_boundary.data.cpu().numpy() / args.iter_size

            
                wandb.log({"fsm loss": args.lambda_fsm*loss_fsm_value}, step=i_iter)
                wandb.log({'SF_loss_seg': loss_seg_sf_value}, step=i_iter)
                wandb.log({'CW_loss_seg': loss_seg_cw_value}, step=i_iter)
                wandb.log({'consistency loss':args.lambda_con*loss_con_value}, step=i_iter)
                wandb.log({'boundary loss':boundary_weight*loss_boundary_value}, step=i_iter)
                wandb.log({'total_loss': loss}, step=i_iter)

                # Collect losses for plotting
                loss_history['total'].append(loss.item() if hasattr(loss, 'item') else loss)
                loss_history['seg_sf'].append(loss_seg_sf_value)
                loss_history['seg_cw'].append(loss_seg_cw_value)
                loss_history['fsm'].append(args.lambda_fsm*loss_fsm_value)
                loss_history['con'].append(args.lambda_con*loss_con_value)
                loss_history['boundary'].append(boundary_weight*loss_boundary_value)
        
        # Increment accumulation step counter
        accum_step += 1
        
        # Only update weights after accumulating enough gradients
        if accum_step >= args.accum_steps:
            accum_step = 0  # Reset counter
            
            # Update all optimizers
            if args.modeltrain == 'train':
                for opt in opts:
                    opt.step()
                
                # Update boundary head only when it is active
                if i_iter >= boundary_warmup_iters:
                    BoundaryHead_optimizer.step()

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
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, args.file_name + str(args.num_steps_stop) + '.pth'))
            break
        if args.modeltrain != 'train':
            if i_iter == 5000:
                torch.save({'state_dict':model.state_dict(),
                'fogpass1_state_dict':FogPassFilter1.state_dict(),
                'fogpass2_state_dict':FogPassFilter2.state_dict(),
                'boundary_state_dict':BoundaryHead_model.state_dict(),
                'train_iter':i_iter,
                'args':args
                },osp.join(args.snapshot_dir, run_name)+'_fogpassfilter_'+str(i_iter)+'.pth')

        if i_iter % save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            # Compute additional metrics
            enc_lr = opts[0].param_groups[0]['lr'] if opts else 0
            dec_lr = opts[1].param_groups[0]['lr'] if len(opts) > 1 else 0
            grad_norm = 0
            if model.parameters():
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]), 2).item() if any(p.grad is not None for p in model.parameters()) else 0
            memory_mb = torch.cuda.memory_allocated(args.gpu) / 1e6 if torch.cuda.is_available() else 0
            
            print(f'Step {i_iter} - SF_loss: {loss_seg_sf_value:.4f}, CW_loss: {loss_seg_cw_value:.4f}, FSM_loss: {args.lambda_fsm*loss_fsm_value:.6f}, Consistency_loss: {args.lambda_con*loss_con_value:.6f}, Boundary_loss: {boundary_weight*loss_boundary_value:.4f}')
            print(f'LR: Enc {enc_lr:.6f}, Dec {dec_lr:.6f} | Grad Norm: {grad_norm:.4f} | Memory: {memory_mb:.1f} MB')
            
            # Plot losses
            plt.figure(figsize=(12, 8))
            steps = list(range(len(loss_history['total'])))
            plt.subplot(2, 3, 1)
            plt.plot(steps, loss_history['total'], label='Total Loss')
            plt.title('Total Loss')
            plt.subplot(2, 3, 2)
            plt.plot(steps, loss_history['seg_sf'], label='SF Seg Loss')
            plt.title('SF Segmentation Loss')
            plt.subplot(2, 3, 3)
            plt.plot(steps, loss_history['seg_cw'], label='CW Seg Loss')
            plt.title('CW Segmentation Loss')
            plt.subplot(2, 3, 4)
            plt.plot(steps, loss_history['fsm'], label='FSM Loss')
            plt.title('FSM Loss')
            plt.subplot(2, 3, 5)
            plt.plot(steps, loss_history['con'], label='Consistency Loss')
            plt.title('Consistency Loss')
            plt.subplot(2, 3, 6)
            plt.plot(steps, loss_history['boundary'], label='Boundary Loss')
            plt.title('Boundary Loss')
            plt.tight_layout()
            plt.savefig(f'./result/loss_plots_step_{i_iter}.png')
            plt.close()
            
            save_dir = osp.join(f'./result/FIFO_model', args.file_name)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save({
                'state_dict':model.state_dict(),
                'fogpass1_state_dict':FogPassFilter1.state_dict(),
                'fogpass2_state_dict':FogPassFilter2.state_dict(),
                'boundary_state_dict':BoundaryHead_model.state_dict(),
                'train_iter':i_iter,
                'args':args
            },osp.join(args.snapshot_dir, run_name)+'_FIFO'+str(i_iter)+'.pth')
            
            # Run evaluation every 2000 steps
            if i_iter % 2000 == 0:
                print('Running evaluation...')
                eval_cmd = [
                    'python', 'evaluate.py',
                    '--restore-from', osp.join(args.snapshot_dir, run_name)+'_FIFO'+str(i_iter)+'.pth',
                    '--gpu', str(args.gpu),
                    '--file-name', args.file_name
                ]
                try:
                    result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=os.getcwd())
                    print('Evaluation output:')
                    print(result.stdout)
                    if result.stderr:
                        print('Evaluation errors:')
                        print(result.stderr)
                except Exception as e:
                    print(f'Error running evaluation: {e}')
            
if __name__ == '__main__':
    main()
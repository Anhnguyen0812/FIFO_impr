import os
import os.path as osp

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data

import argparse
import numpy as np
from packaging import version
import wandb
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from model.refinenetlw import rf_lw101
from model.segformer_backbone import segformer_fifo
from compute_iou import compute_mIoU
from configs.test_config import get_arguments, IMG_MEAN_RESNET, IMG_NORM_SEGFORMER
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.Foggy_Zurich_test import foggyzurichDataSet
from dataset.foggy_driving import foggydrivingDataSet

RESTORE_FROM = 'without_pretraining'
IMG_MEAN = IMG_MEAN_RESNET  # Backward compatibility

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def sliding_window_inference(model, image, window_size=(1024, 1024), stride=512, device='cuda:0'):
    """
    Sliding window inference for SegFormer to avoid positional embedding issues.
    
    SegFormer is trained with crop size 1024x1024. For larger images, we use sliding
    windows to maintain the correct positional embeddings.
    
    Args:
        model: The segmentation model
        image: Input tensor [B, C, H, W] (B should be 1 for inference)
        window_size: Size of sliding window (default: 1024x1024, matching SegFormer training)
        stride: Stride for sliding window (default: 512, 50% overlap)
        device: CUDA device
        
    Returns:
        pred: Prediction tensor [B, num_classes, H, W]
    """
    B, C, H, W = image.shape
    window_h, window_w = window_size
    
    # If image is smaller than window, use whole image inference
    if H <= window_h and W <= window_w:
        with torch.no_grad():
            _, _, _, _, _, pred = model(Variable(image).to(device))
        return pred
    
    # Initialize prediction and count maps
    num_classes = 19  # Cityscapes
    pred_map = torch.zeros(B, num_classes, H, W, device=device)
    count_map = torch.zeros(B, 1, H, W, device=device)
    
    # Calculate number of windows
    h_steps = max(1, (H - window_h) // stride + 1)
    w_steps = max(1, (W - window_w) // stride + 1)
    
    # Slide window
    for h_idx in range(h_steps):
        for w_idx in range(w_steps):
            # Calculate window position
            h_start = min(h_idx * stride, H - window_h)
            w_start = min(w_idx * stride, W - window_w)
            h_end = h_start + window_h
            w_end = w_start + window_w
            
            # Extract window
            image_window = image[:, :, h_start:h_end, w_start:w_end]
            
            # Inference on window
            with torch.no_grad():
                _, _, _, _, _, pred_window = model(Variable(image_window).to(device))
            
            # Accumulate predictions
            pred_map[:, :, h_start:h_end, w_start:w_end] += pred_window
            count_map[:, :, h_start:h_end, w_start:w_end] += 1
    
    # Average overlapping predictions
    pred_map = pred_map / count_map
    
    return pred_map


def inference_with_strategy(model, image, use_segformer=False, device='cuda:0', output_size=None):
    """
    Unified inference function supporting both ResNet (multi-scale) and SegFormer (sliding window).
    
    Args:
        model: The segmentation model
        image: Input tensor [B, C, H, W]
        use_segformer: If True, use sliding window; if False, use whole image
        device: CUDA device
        output_size: Target output size (H, W) for upsampling
        
    Returns:
        pred: Final prediction after model inference and upsampling [B, C, H, W]
    """
    if use_segformer:
        # SegFormer: Use sliding window to maintain positional embeddings
        # Trained with 1024x1024, so use that as window size
        pred = sliding_window_inference(model, image, window_size=(1024, 1024), stride=512, device=device)
    else:
        # ResNet: Whole image inference
        with torch.no_grad():
            _, _, _, _, _, pred = model(Variable(image).to(device))
    
    # Upsample to target size if specified
    if output_size is not None:
        if version.parse(torch.__version__) >= version.parse('0.4.0'):
            interp = nn.Upsample(size=output_size, mode='bilinear', align_corners=True)
        else:
            interp = nn.Upsample(size=output_size, mode='bilinear')
        pred = interp(pred)
    
    return pred

def eval():
    """Create the model and start the evaluation process."""
    args = get_arguments()

    if args.restore_from == RESTORE_FROM:
        start_iter = 0
        # Default to ResNet if no checkpoint provided
        use_segformer = getattr(args, 'use_segformer', False)
        if use_segformer:
            print("\n" + "="*70)
            print("Using SegFormer MIT-B5 backbone for evaluation")
            print("="*70)
            model = segformer_fifo(num_classes=args.num_classes)
        else:
            print("\nUsing ResNet-101 backbone for evaluation")
            model = rf_lw101(num_classes=args.num_classes)

    else:
        restore = torch.load(args.restore_from, weights_only=False, map_location='cpu')
        
        # Auto-detect model type from checkpoint
        use_segformer = False
        if 'args' in restore and hasattr(restore['args'], 'use_segformer'):
            use_segformer = restore['args'].use_segformer
        elif 'state_dict' in restore:
            # Check if checkpoint has SegFormer layers
            first_key = list(restore['state_dict'].keys())[0]
            if 'backbone.segformer' in first_key or 'segformer' in first_key:
                use_segformer = True
        
        if use_segformer:
            print("\n" + "="*70)
            print("✓ Detected SegFormer checkpoint")
            print("  Loading SegFormer MIT-B5 backbone")
            print("="*70)
            model = segformer_fifo(
                num_classes=args.num_classes,
                pretrained=False  # Load from checkpoint, not pretrained
            )
        else:
            print("\n✓ Detected ResNet-101 checkpoint")
            model = rf_lw101(num_classes=args.num_classes)

        model.load_state_dict(restore['state_dict'])
        start_iter = 0
        print(f"✓ Checkpoint loaded successfully")
        if 'train_iter' in restore:
            print(f"  Trained for {restore['train_iter']} iterations")

    save_dir_fz = osp.join(f'./result_FZ', args.file_name)
    save_dir_fd = osp.join(f'./result_FD', args.file_name)
    save_dir_fdd = osp.join(f'./result_FDD', args.file_name)
    save_dir_clindau = osp.join(f'./result_Clindau', args.file_name)      

    if not os.path.exists(save_dir_fz):
        os.makedirs(save_dir_fz)
    if not os.path.exists(save_dir_fd):
        os.makedirs(save_dir_fd)
    if not os.path.exists(save_dir_fdd):
        os.makedirs(save_dir_fdd)
    if not os.path.exists(save_dir_clindau):
        os.makedirs(save_dir_clindau)
    
    model.eval()
    device = torch.device('cuda:0')
    model.to(device)

    # Select preprocessing based on model type
    if use_segformer:
        print("\n" + "="*70)
        print("Using SegFormer preprocessing:")
        print(f"  • Format: RGB (no BGR conversion)")
        print(f"  • Mean: {IMG_NORM_SEGFORMER['mean']}")
        print(f"  • Std: {IMG_NORM_SEGFORMER['std']}")
        print("="*70 + "\n")
        dataset_kwargs = {
            'use_segformer_norm': True,
            'norm_cfg': IMG_NORM_SEGFORMER
        }
    else:
        print("\n" + "="*70)
        print("Using ResNet preprocessing:")
        print(f"  • Format: BGR (RGB→BGR conversion)")
        print(f"  • Mean subtract: {IMG_MEAN}")
        print("="*70 + "\n")
        dataset_kwargs = {
            'mean': IMG_MEAN,
            'use_segformer_norm': False
        }

    testloader1 = data.DataLoader(foggyzurichDataSet(args.data_dir_eval, args.data_list_eval, 
                                                       crop_size=(1152, 648), **dataset_kwargs),
                                    batch_size=1, shuffle=False, pin_memory=True)
    testloader2 = data.DataLoader(foggyzurichDataSet(args.data_dir_eval, args.data_list_eval, 
                                                       crop_size=(1536, 864), **dataset_kwargs),
                                    batch_size=1, shuffle=False, pin_memory=True)
    testloader3 = data.DataLoader(foggyzurichDataSet(args.data_dir_eval, args.data_list_eval, 
                                                       crop_size=(1920, 1080), **dataset_kwargs),
                                    batch_size=1, shuffle=False, pin_memory=True)

    # Choose inference strategy based on model type
    if use_segformer:
        # SegFormer: Single scale with sliding window (more stable for Transformer)
        print("Using SegFormer inference: Sliding window mode (1024x1024 windows)")
        
        for index, batch1 in enumerate(testloader1):
            image, label_test, _, name = batch1
            
            # Sliding window inference + upsample to target size
            output = inference_with_strategy(
                model, image, use_segformer=True, 
                device=device, output_size=(1080, 1920)
            )
            
            output = output.cpu().numpy()
            output = output[0].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            output_col = colorize_mask(output)
            output = Image.fromarray(output)

            name = name[0].split('/')[-1]
            output.save('%s/%s' % (save_dir_fz, name))
            output_col.save('%s/%s_color.png' % (save_dir_fz, name[:-4]))
    else:
        # ResNet: Multi-scale inference (original approach)
        print("Using ResNet inference: Multi-scale mode")
        
        if version.parse(torch.__version__) >= version.parse('0.4.0'):
            interp_eval = nn.Upsample(size=(1080,1920), mode='bilinear', align_corners=True)
        else:
            interp_eval = nn.Upsample(size=(1080,1920), mode='bilinear')

        testloader_iter2 = enumerate(testloader2)
        testloader_iter3 = enumerate(testloader3)

        for index, batch1 in enumerate(testloader1):
            image, label_test, _, name = batch1
            with torch.no_grad():
                output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
                output_1 = interp_eval(output2)

            _, batch2 = testloader_iter2.__next__()
            image, label_test, _, name = batch2
            with torch.no_grad():
                output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
                output_2 = interp_eval(output2)

            _, batch3 = testloader_iter3.__next__()    
            image, label_test, _, name = batch3
            with torch.no_grad():
                output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
                output_3 = interp_eval(output2)

            output = torch.cat([output_1,output_2,output_3])
            output = torch.mean(output, dim=0)
            output = output.cpu().numpy()
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            output_col = colorize_mask(output)
            output = Image.fromarray(output)

            name = name[0].split('/')[-1]
            output.save('%s/%s' % (save_dir_fz, name))
            output_col.save('%s/%s_color.png' % (save_dir_fz, name[:-4]))
            
    miou_fz = compute_mIoU(args.gt_dir_fz, save_dir_fz, args.devkit_dir_fz, 'FZ')

    # Test on Foggy Driving Dense (if available)
    try:
        fd_kwargs = dataset_kwargs.copy()
        testloader1 = data.DataLoader(foggydrivingDataSet(args.data_dir_eval_fd, args.data_list_eval_fdd, 
                                                           scale=1, **fd_kwargs),
                                        batch_size=1, shuffle=False, pin_memory=True)

        testloader2 = data.DataLoader(foggydrivingDataSet(args.data_dir_eval_fd, args.data_list_eval_fdd, 
                                                           scale=0.8, **fd_kwargs),
                                        batch_size=1, shuffle=False, pin_memory=True) 

        testloader3 = data.DataLoader(foggydrivingDataSet(args.data_dir_eval_fd, args.data_list_eval_fdd, 
                                                           scale=0.6, **fd_kwargs),
                                        batch_size=1, shuffle=False, pin_memory=True)
        if use_segformer:
            # SegFormer: Single scale with sliding window
            for index, batch in enumerate(testloader1):
                image, size, name = batch
                target_size = (size[0][0].item(), size[0][1].item())
                
                output = inference_with_strategy(
                    model, image, use_segformer=True,
                    device=device, output_size=target_size
                )
                
                output = output.cpu().numpy()
                output = output[0].transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

                output_col = colorize_mask(output)
                output = Image.fromarray(output)

                name = name[0].split('/')[-1]
                output.save('%s/%s' % (save_dir_fdd, name))
                output_col.save('%s/%s_color.png' % (save_dir_fdd, name[:-4]))
        else:
            # ResNet: Multi-scale
            testloader_iter2 = enumerate(testloader2)
            testloader_iter3 = enumerate(testloader3)

            for index, batch in enumerate(testloader1):
                image, size, name = batch
                with torch.no_grad():
                    output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
                    interp_eval = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')
                    output_1 = interp_eval(output2)

                _, batch2 = testloader_iter2.__next__()
                image, _, name = batch2
                with torch.no_grad():
                    output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
                    output_2 = interp_eval(output2)

                _, batch3 = testloader_iter3.__next__()    
                image, _, name = batch3
                with torch.no_grad():
                    output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
                    output_3 = interp_eval(output2)

                output = torch.cat([output_1,output_2,output_3])
                output = torch.mean(output, dim=0)
                output = output.cpu().numpy()
                output = output.transpose(1,2,0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

                output_col = colorize_mask(output)
                output = Image.fromarray(output)

                name = name[0].split('/')[-1]
                output.save('%s/%s' % (save_dir_fdd, name))
                output_col.save('%s/%s_color.png' % (save_dir_fdd, name[:-4]))
                
        miou_fdd = compute_mIoU(args.gt_dir_fd, save_dir_fdd, args.devkit_dir_fd, 'FDD')
    except FileNotFoundError as e:
        print(f"Skipping Foggy Driving Dense evaluation (dataset not available): {e}")
        miou_fdd = 0
    except Exception as e:
        print(f"Skipping Foggy Driving Dense evaluation (error): {e}")
        miou_fdd = 0

    # Test on Foggy Driving (if available)
    try:
        fd_kwargs = dataset_kwargs.copy()
        testloader1 = data.DataLoader(foggydrivingDataSet(args.data_dir_eval_fd, args.data_list_eval_fd, 
                                                           scale=1, **fd_kwargs),
                                        batch_size=1, shuffle=False, pin_memory=True) 

        testloader2 = data.DataLoader(foggydrivingDataSet(args.data_dir_eval_fd, args.data_list_eval_fd, 
                                                           scale=0.8, **fd_kwargs),
                                        batch_size=1, shuffle=False, pin_memory=True) 

        testloader3 = data.DataLoader(foggydrivingDataSet(args.data_dir_eval_fd, args.data_list_eval_fd, 
                                                           scale=0.6, **fd_kwargs),
                                        batch_size=1, shuffle=False, pin_memory=True) 
        if use_segformer:
            # SegFormer: Single scale with sliding window
            for index, batch in enumerate(testloader1):
                image, size, name = batch
                target_size = (size[0][0].item(), size[0][1].item())
                
                output = inference_with_strategy(
                    model, image, use_segformer=True,
                    device=device, output_size=target_size
                )
                
                output = output.cpu().numpy()
                output = output[0].transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

                output_col = colorize_mask(output)
                output = Image.fromarray(output)

                name = name[0].split('/')[-1]
                output.save('%s/%s' % (save_dir_fd, name))
                output_col.save('%s/%s_color.png' % (save_dir_fd, name[:-4]))
        else:
            # ResNet: Multi-scale
            testloader_iter2 = enumerate(testloader2)
            testloader_iter3 = enumerate(testloader3)

            for index, batch in enumerate(testloader1):
                image, size, name = batch
                with torch.no_grad():
                    output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
                    interp_eval = nn.Upsample(size=(size[0][0],size[0][1]), mode='bilinear')

                    output_1 = interp_eval(output2)

                _, batch2 = testloader_iter2.__next__()
                image, _, name = batch2
                with torch.no_grad():
                    output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
                    output_2 = interp_eval(output2)

                _, batch3 = testloader_iter3.__next__()    
                image, _, name = batch3
                with torch.no_grad():
                    output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
                    output_3 = interp_eval(output2)

                output = torch.cat([output_1,output_2,output_3])
                output = torch.mean(output, dim=0)
                output = output.cpu().numpy()
                output = output.transpose(1,2,0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

                output_col = colorize_mask(output)
                output = Image.fromarray(output)

                name = name[0].split('/')[-1]
                output.save('%s/%s' % (save_dir_fd, name))
                output_col.save('%s/%s_color.png' % (save_dir_fd, name[:-4]))
                
        miou_fd = compute_mIoU(args.gt_dir_fd, save_dir_fd, args.devkit_dir_fd, 'FD')
    except FileNotFoundError as e:
        print(f"Skipping Foggy Driving evaluation (dataset not available): {e}")
        miou_fd = 0
    except Exception as e:
        print(f"Skipping Foggy Driving evaluation (error): {e}")
        miou_fd = 0

    # Test on Clear Lindau (if available)
    try:
        city_kwargs = dataset_kwargs.copy()
        city_kwargs.update({'scale': False, 'mirror': False, 'set': args.set})
        
        testloader1 = data.DataLoader(cityscapesDataSet(args.data_dir_city, args.data_city_list, 
                                                         crop_size=(2048, 1024), **city_kwargs),
                                batch_size=1, shuffle=False, pin_memory=True)
        testloader2 = data.DataLoader(cityscapesDataSet(args.data_dir_city, args.data_city_list, 
                                                         crop_size=(2048*0.8, 1024*0.8), **city_kwargs),
                                batch_size=1, shuffle=False, pin_memory=True)
        testloader3 = data.DataLoader(cityscapesDataSet(args.data_dir_city, args.data_city_list, 
                                                         crop_size=(2048*0.6, 1024*0.6), **city_kwargs),
                                batch_size=1, shuffle=False, pin_memory=True)   
        if use_segformer:
            # SegFormer: Single scale with sliding window
            for index, batch in enumerate(testloader1):
                image, size, name = batch
                
                output = inference_with_strategy(
                    model, image, use_segformer=True,
                    device=device, output_size=(1024, 2048)
                )
                
                output = output.cpu().numpy()
                output = output[0].transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

                output_col = colorize_mask(output)
                output = Image.fromarray(output)

                name = name[0].split('/')[-1]
                output.save('%s/%s' % (save_dir_clindau, name))
                output_col.save('%s/%s_color.png' % (save_dir_clindau, name.split('.')[0]))
        else:
            # ResNet: Multi-scale
            testloader_iter2 = enumerate(testloader2)
            testloader_iter3 = enumerate(testloader3)

            for index, batch in enumerate(testloader1):
                image, size, name = batch
                with torch.no_grad():
                    output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
                    interp_eval = nn.Upsample(size=(1024, 2048), mode='bilinear')
                    output_1 = interp_eval(output2)

                _, batch2 = testloader_iter2.__next__()
                image, _, name = batch2
                with torch.no_grad():
                    output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
                    output_2 = interp_eval(output2)

                _, batch3 = testloader_iter3.__next__()    
                image, _, name = batch3
                with torch.no_grad():
                    output6, output3, output4, output5, output1, output2 = model(Variable(image).cuda(args.gpu))
                    output_3 = interp_eval(output2)

                output = torch.cat([output_1,output_2,output_3])
                output = torch.mean(output, dim=0)
                output = output.cpu().numpy()
                output = output.transpose(1,2,0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

                output_col = colorize_mask(output)
                output = Image.fromarray(output)

                name = name[0].split('/')[-1]
                output.save('%s/%s' % (save_dir_clindau, name))
                output_col.save('%s/%s_color.png' % (save_dir_clindau, name.split('.')[0]))

        miou_clindau = compute_mIoU(args.gt_dir_clindau, save_dir_clindau, args.devkit_dir_clindau, 'Clindau')
    except FileNotFoundError:
        print("Skipping Clear Lindau evaluation (dataset not available)")
        miou_clindau = 0



if __name__ == '__main__':
    eval()

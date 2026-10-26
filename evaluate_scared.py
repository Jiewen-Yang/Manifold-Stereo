"""
Evaluate model on SCARED test set by computing depth map MSE/MAE.
Converts predicted and ground-truth disparity maps to depth maps using Q matrices,
then calculates depth prediction errors.
"""

import os
import json
import torch
import numpy as np
import cv2
import os.path as osp
from tqdm import tqdm
from pathlib import Path
import logging


def setup_logger(name, log_path):
    """Logger that writes to file and stdout."""
    os.makedirs(osp.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

import core.stereo_datasets as datasets
from core.endo_stereo_fm import EndoStereoFM
from core.utils.utils import InputPadder
from omegaconf import OmegaConf


def get_Q_from_path(data_root, image_path):
    """
    Read Q matrix from stereo calibration file based on image path.
    
    Args:
        data_root: Root directory of the dataset
        image_path: Path to the image file (string)
    
    Returns:
        Q matrix as numpy array (4x4)
    """
    # Convert path to disk folder format
    # e.g., "dataset2/keyframe2" -> "dataset_2/keyframe_2"
    parts = image_path.replace('\\', '/').split('/')
    
    # Find dataset and keyframe folders
    dataset_folder = None
    keyframe_folder = None
    data_split = None
    
    for i, part in enumerate(parts):
        if part.startswith('dataset'):
            dataset_folder = part
            # Determine if it's train_data or test_data based on the path
            if 'train_data' in parts:
                data_split = 'train_data'
            elif 'test_data' in parts:
                data_split = 'test_data'
        elif part.startswith('keyframe'):
            keyframe_folder = part
    
    if dataset_folder is None or keyframe_folder is None:
        raise ValueError(f"Could not extract dataset/keyframe info from {image_path}")
    
    # Convert dataset2 -> dataset_2, keyframe2 -> keyframe_2
    if not dataset_folder.startswith('dataset_'):
        dataset_folder = dataset_folder.replace('dataset', 'dataset_', 1)
    if not keyframe_folder.startswith('keyframe_'):
        keyframe_folder = keyframe_folder.replace('keyframe', 'keyframe_', 1)
    
    calib_path = osp.join(data_root, data_split, dataset_folder, keyframe_folder, 'stereo_calib.json')
    
    with open(calib_path, mode='r') as rf:
        calib = json.load(rf)
    Q = np.array(calib['Q']['data'], dtype=np.float64).reshape(4, 4)
    return Q


def disp2depth(disp, Q):
    """
    Convert disparity map to depth map using Q matrix.
    
    Args:
        disp: Disparity tensor (B, 1, H, W) or (H, W)
        Q: Q matrix from stereo calibration (4x4)
    
    Returns:
        Depth tensor with same shape as input disparity
    """
    if disp.ndim == 4:
        disp_np = disp.squeeze(1).cpu().numpy() if disp.is_cuda else disp.squeeze(1).numpy()
    else:
        disp_np = disp.squeeze().cpu().numpy() if disp.is_cuda else disp.squeeze().numpy()
    
    # Handle batch dimension if present
    if disp_np.ndim == 3:
        depth_list = []
        for i in range(disp_np.shape[0]):
            depth_map = cv2.reprojectImageTo3D(disp_np[i].astype(np.float32), Q)[:, :, -1]
            depth_map = np.nan_to_num(depth_map, posinf=0, neginf=0, nan=0)
            depth_list.append(depth_map)
        depth_np = np.stack(depth_list, axis=0)
        depth_tensor = torch.tensor(depth_np, device=disp.device, dtype=disp.dtype).unsqueeze(1)
    else:
        depth_np = cv2.reprojectImageTo3D(disp_np.astype(np.float32), Q)[:, :, -1]
        depth_np = np.nan_to_num(depth_np, posinf=0, neginf=0, nan=0)
        depth_tensor = torch.tensor(depth_np, device=disp.device, dtype=disp.dtype).unsqueeze(0)
    
    return depth_tensor


@torch.no_grad()
def evaluate_scared(model_path, data_root, split_file, config_path=None, device='cuda', iters=32, test_mode=True, batch_size=1, logger=None):
    """
    Evaluate model on SCARED dataset by computing depth prediction errors (MAE).

    Args:
        model_path: Path to the model checkpoint
        data_root: Root directory of test dataset
        split_file: Path to the split file (e.g., 'test_files.txt')
        config_path: Path to config file (optional)
        device: Device to use ('cuda' or 'cpu')
        iters: Number of iterations for model inference
        test_mode: Whether to use test mode
        batch_size: Batch size for DataLoader
        logger: logging.Logger instance (optional, prints to stdout if None)

    Returns:
        Dictionary with per-sample and average MAE metrics
    """
    _log = logger.info if logger else print

    # Load model
    _log("Loading model from: %s", model_path)
    if config_path is None:
        config_path = './config/train_scared.yaml'
    
    if not osp.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    
    model = EndoStereoFM(cfg)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    _log("Model loaded successfully!")

    # Load SCARED dataset
    _log("Loading SCARED test dataset from: %s", data_root)
    _log("Using split file: %s", split_file)
    aug_params = {}
    val_dataset = datasets.SCARED(aug_params, root=data_root, split_file=split_file)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size,
        pin_memory=True, 
        shuffle=False, 
        num_workers=1, 
        drop_last=False
    )
    
    _log(f"Dataset loaded! Total samples: {len(val_dataset)}")
    
    # Evaluation metrics
    depth_mae_list = []
    abs_rel_list = []
    sq_rel_list = []
    rmse_list = []
    rmse_log_list = []
    delta1_list = []
    
    # Inference
    _log(f"\nStarting evaluation...")
    for batch_idx, data in enumerate(tqdm(val_loader, dynamic_ncols=True)):
        # SCARED dataset returns: path_list, img1, img2, disp, valid
        # path_list contains: [left_img_path, right_img_path, disp_path]
        img_info, left, right, disp_gt, valid = [x for x in data]
        
        # Get Q matrix - img_info is a list of paths
        if isinstance(img_info, (list, tuple)) and len(img_info) > 0:
            # Get the first element which is the left image path
            image_path = img_info[0]
        else:
            _log(f"Warning: Unexpected img_info format at batch {batch_idx}: {type(img_info)}")
            continue
        
        # Handle case where image_path might be a tensor or bytes
        if isinstance(image_path, torch.Tensor):
            image_path = image_path.item() if image_path.numel() == 1 else str(image_path)
        elif isinstance(image_path, bytes):
            image_path = image_path.decode('utf-8')
        else:
            image_path = str(image_path)
        
        try:
            Q = get_Q_from_path(data_root, image_path)
        except Exception as e:
            _log(f"Error reading Q matrix for {image_path}: {e}")
            continue
        
        # Prepare inputs
        left = left.to(device)
        right = right.to(device)
        disp_gt = disp_gt.to(device)
        valid = valid.to(device)
        
        # Model inference
        padder = InputPadder(left.shape, divis_by=32)
        left_padded, right_padded = padder.pad(left, right)
        
        with torch.no_grad():
            disp_pred = model(left_padded, right_padded, iters=iters, test_mode=test_mode)
        
        disp_pred = padder.unpad(disp_pred)
        
        # Convert disparity to depth
        depth_pred = disp2depth(disp_pred, Q)
        depth_gt = disp2depth(disp_gt, Q)
        
        # Ensure valid has same shape as depth tensors for masking
        if valid.dim() == 3:
            valid = valid.unsqueeze(1)
        
        # Filter by valid mask
        valid_mask = valid >= 0.5
        
        if valid_mask.sum() == 0:
            continue
        
        # Compute depth errors
        valid_pred = depth_pred[valid_mask]
        valid_gt = depth_gt[valid_mask]
        
        # Remove invalid depth values (0 or very large)
        valid_idx = (valid_pred > 20) & (valid_gt > 20) & (valid_pred < 168) & (valid_gt < 168)
        
        if valid_idx.sum() > 0:
            valid_pred_filtered = valid_pred[valid_idx]
            valid_gt_filtered = valid_gt[valid_idx]
            
            # Compute MAE
            mae = torch.abs(valid_pred_filtered - valid_gt_filtered).mean().item()
            depth_mae_list.append(mae)
            
            # Compute additional depth metrics
            abs_rel = torch.mean(torch.abs(valid_pred_filtered - valid_gt_filtered) / valid_gt_filtered).item()
            sq_rel = torch.mean(((valid_pred_filtered - valid_gt_filtered) ** 2) / valid_gt_filtered).item()
            rmse = torch.sqrt(torch.mean((valid_pred_filtered - valid_gt_filtered) ** 2)).item()
            rmse_log = torch.sqrt(torch.mean((torch.log(valid_pred_filtered) - torch.log(valid_gt_filtered)) ** 2)).item()
            
            # Compute delta accuracy (threshold at 1.25)
            ratio = torch.maximum(valid_pred_filtered / valid_gt_filtered, valid_gt_filtered / valid_pred_filtered)
            delta1 = (ratio < 1.25).float().mean().item()
            
            abs_rel_list.append(abs_rel)
            sq_rel_list.append(sq_rel)
            rmse_list.append(rmse)
            rmse_log_list.append(rmse_log)
            delta1_list.append(delta1)
            
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                avg_mae = np.mean(depth_mae_list)
                avg_abs_rel = np.mean(abs_rel_list)
                avg_sq_rel = np.mean(sq_rel_list)
                avg_rmse = np.mean(rmse_list)
                avg_rmse_log = np.mean(rmse_log_list)
                avg_delta1 = np.mean(delta1_list)
                _log(f"Sample {batch_idx+1}/{len(val_loader)}: "
                      f"MAE={mae:.4f} | abs_rel={abs_rel:.4f} | sq_rel={sq_rel:.4f} | "
                      f"rmse={rmse:.4f} | rmse_log={rmse_log:.4f} | delta1={delta1:.4f}")
    
    # Final statistics
    if len(depth_mae_list) > 0:
        avg_mae = np.mean(depth_mae_list)
        std_mae = np.std(depth_mae_list)
        avg_abs_rel = np.mean(abs_rel_list)
        std_abs_rel = np.std(abs_rel_list)
        avg_sq_rel = np.mean(sq_rel_list)
        std_sq_rel = np.std(sq_rel_list)
        avg_rmse = np.mean(rmse_list)
        std_rmse = np.std(rmse_list)
        avg_rmse_log = np.mean(rmse_log_list)
        std_rmse_log = np.std(rmse_log_list)
        avg_delta1 = np.mean(delta1_list)
        std_delta1 = np.std(delta1_list)
        
        _log("\n" + "="*70)
        _log("SCARED DEPTH EVALUATION RESULTS")
        _log("="*70)
        _log(f"Mean Absolute Error (MAE):           {avg_mae:.6f} +/- {std_mae:.6f}")
        _log(f"Absolute Relative (abs_rel):        {avg_abs_rel:.6f} +/- {std_abs_rel:.6f}")
        _log(f"Squared Relative (sq_rel):          {avg_sq_rel:.6f} +/- {std_sq_rel:.6f}")
        _log(f"Root Mean Square Error (rmse):       {avg_rmse:.6f} +/- {std_rmse:.6f}")
        _log(f"RMSE in log space (rmse_log):        {avg_rmse_log:.6f} +/- {std_rmse_log:.6f}")
        _log(f"Delta1 accuracy (<1.25):             {avg_delta1:.6f} +/- {std_delta1:.6f}")
        _log("="*70)
        _log(f"Samples evaluated: {len(depth_mae_list)}")
        _log("="*70)

        return {
            'depth_mae': avg_mae,
            'depth_mae_std': std_mae,
            'abs_rel': avg_abs_rel,
            'abs_rel_std': std_abs_rel,
            'sq_rel': avg_sq_rel,
            'sq_rel_std': std_sq_rel,
            'rmse': avg_rmse,
            'rmse_std': std_rmse,
            'rmse_log': avg_rmse_log,
            'rmse_log_std': std_rmse_log,
            'delta1': avg_delta1,
            'delta1_std': std_delta1,
            'num_samples': len(depth_mae_list),
            'mae_list': depth_mae_list
        }
    else:
        _log("No valid samples found for evaluation!")
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate depth prediction on SCARED dataset')
    parser.add_argument('--model_path', type=str, default='/media/Storage3/ug_zy/random_crop/Stereo_Match/checkpoints/kitti/44000.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='/media/Storage3/ug_zy/',
                        help='Root directory of SCARED dataset')
    parser.add_argument('--split_file', type=str, default='/media/Storage3/ug_zy/splits/endovis/test_files.txt',
                        help='Path to SCARED split file')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--iters', type=int, default=32,
                        help='Number of iterations for inference')
    parser.add_argument('--test_mode', action='store_true', default=True,
                        help='Use test mode')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (currently only 1 is supported due to mixed path/tensor collation)')
    
    args = parser.parse_args()

    model_stem = osp.splitext(osp.basename(args.model_path))[0]
    log_dir = osp.join(osp.dirname(args.model_path), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = osp.join(log_dir, f'{model_stem}_scared_depth_eval.log')
    logger = setup_logger('scared_depth_eval', log_path)

    logger.info("=" * 60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Split file: {args.split_file}")
    logger.info(f"Device: {args.device}  iters: {args.iters}  batch_size: {args.batch_size}")
    logger.info("=" * 60)

    results = evaluate_scared(
        model_path=args.model_path,
        data_root=args.data_root,
        split_file=args.split_file,
        config_path=args.config_path,
        device=args.device,
        iters=args.iters,
        test_mode=args.test_mode,
        batch_size=args.batch_size,
        logger=logger,
    )

    if results is not None:
        results_json = {k: v for k, v in results.items() if k not in ['mae_list']}
        output_path = osp.join(osp.dirname(args.model_path), f'{model_stem}_scared_depth_results.json')
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=4)
        logger.info(f"Results saved to: {output_path}")

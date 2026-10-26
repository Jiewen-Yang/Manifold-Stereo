# -*- coding: utf-8 -*-
"""
Evaluate model on Hamlyn endoscopic stereo dataset.
Computes disparity-level metrics (MAE, D1, Bad-X) and depth-level metrics (Abs Rel, Sq Rel, RMSE).

Hamlyn calibration is read per-sequence from calibration/<seq_id>/ directory.
Depth conversion uses: depth = fx * baseline / disparity
"""

import os
import json
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm
from pathlib import Path

import core.stereo_datasets as datasets
from core.endo_stereo_fm import EndoStereoFM
from core.utils.utils import InputPadder
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Disparity <-> Depth conversion (using fx and baseline)
# ---------------------------------------------------------------------------

def disp2depth(disp, fx, baseline):
    """
    Convert disparity map to depth map using fx and baseline.

    Formula: depth = fx * baseline / disparity

    Args:
        disp: Disparity tensor (B, 1, H, W) or (1, H, W) or (H, W)
        fx: Focal length in pixels
        baseline: Baseline distance in mm

    Returns:
        Depth tensor with same shape as input disparity (in mm)
    """
    if disp.ndim == 4:
        disp_np = disp.squeeze(1).cpu().numpy() if disp.is_cuda else disp.squeeze(1).numpy()
    else:
        disp_np = disp.squeeze().cpu().numpy() if disp.is_cuda else disp.squeeze().numpy()

    # Handle batch dimension if present
    if disp_np.ndim == 3:
        depth_list = []
        for i in range(disp_np.shape[0]):
            depth_map = fx * baseline / (disp_np[i] + 1e-8)
            depth_map = np.nan_to_num(depth_map, posinf=0, neginf=0, nan=0)
            depth_list.append(depth_map)
        depth_np = np.stack(depth_list, axis=0)
        depth_tensor = torch.tensor(depth_np, device=disp.device, dtype=disp.dtype).unsqueeze(1)
    else:
        depth_np = fx * baseline / (disp_np + 1e-8)
        depth_np = np.nan_to_num(depth_np, posinf=0, neginf=0, nan=0)
        depth_tensor = torch.tensor(depth_np, device=disp.device, dtype=disp.dtype).unsqueeze(0)

    return depth_tensor


# ---------------------------------------------------------------------------
# Per-pixel metric computation
# ---------------------------------------------------------------------------

def compute_disparity_metrics(disp_pred, disp_gt, valid_mask):
    """
    Compute per-pixel disparity metrics.

    Args:
        disp_pred: predicted disparity tensor (1, H, W)
        disp_gt:   ground-truth disparity tensor (1, H, W)
        valid_mask: boolean tensor (H, W) indicating valid GT pixels

    Returns:
        dict with keys: mae, bad1, bad2, bad3, d1, num_px
    """
    # Absolute error per pixel
    abs_err = torch.abs(disp_pred - disp_gt)          # (1, H, W)
    abs_err = abs_err.squeeze(0)                      # (H, W)

    valid_err = abs_err[valid_mask]

    num_px = valid_err.numel()
    if num_px == 0:
        return None

    mae   = valid_err.sum().item() / num_px
    bad1  = (valid_err > 1.0).sum().item() / num_px * 100.0
    bad2  = (valid_err > 2.0).sum().item() / num_px * 100.0
    bad3  = (valid_err > 3.0).sum().item() / num_px * 100.0

    # D1: error > max(3px, 5% of GT disparity)
    disp_gt_squeezed = disp_gt.squeeze(0)              # (H, W)
    threshold_map = torch.maximum(
        torch.full_like(disp_gt_squeezed, 3.0),
        0.05 * disp_gt_squeezed
    )
    d1_mask = (abs_err > threshold_map) & valid_mask
    d1 = d1_mask.sum().item() / num_px * 100.0

    return {
        'mae': mae,
        'bad1': bad1,
        'bad2': bad2,
        'bad3': bad3,
        'd1': d1,
        'num_px': num_px,
    }


def compute_depth_metrics(depth_pred, depth_gt, valid_mask, depth_max=150.0):
    """
    Compute SCARED-standard depth metrics including delta accuracy.

    Args:
        depth_pred: predicted depth tensor (1, H, W)
        depth_gt:   ground-truth depth tensor (1, H, W)
        valid_mask: boolean tensor (H, W) indicating valid GT pixels
        depth_max:  cap for depth values (default 200 mm)

    Returns:
        dict with keys: abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3, num_px
    """
    depth_pred = depth_pred.squeeze(0)                  # (H, W)
    depth_gt   = depth_gt.squeeze(0)

    # Start from valid GT pixels
    valid_depth_gt   = depth_gt[valid_mask]
    valid_depth_pred = depth_pred[valid_mask]

    # Filter: both > 1 and < depth_max
    keep = (valid_depth_gt > 1) & (valid_depth_pred > 1) & \
           (valid_depth_gt < depth_max) & (valid_depth_pred < depth_max)

    valid_depth_gt   = valid_depth_gt[keep]
    valid_depth_pred = valid_depth_pred[keep]

    num_px = valid_depth_gt.numel()
    if num_px == 0:
        return None

    abs_diff = torch.abs(valid_depth_pred - valid_depth_gt)

    # Abs Relative Error: mean(|e - t| / t)
    abs_rel = abs_diff / valid_depth_gt
    abs_rel = abs_rel.sum().item() / num_px

    # Sq Relative Error: mean((e - t)^2 / t)
    sq_rel = (abs_diff ** 2) / valid_depth_gt
    sq_rel = sq_rel.sum().item() / num_px

    # RMSE: sqrt(mean((e - t)^2))
    rmse = (abs_diff ** 2).sum().item() / num_px
    rmse = np.sqrt(rmse)

    # RMSE in log space: sqrt(mean((log e - log t)^2))
    log_err = torch.log(valid_depth_pred) - torch.log(valid_depth_gt)
    rmse_log = (log_err ** 2).sum().item() / num_px
    rmse_log = np.sqrt(rmse_log)

    # delta accuracy: fraction of pixels where max(d/d*, d*/d) < threshold
    # delta1: threshold = 1.25  (25% relative error)
    # delta2: threshold = 1.25^2  (56% relative error)
    # delta3: threshold = 1.25^3  (95% relative error)
    ratio = torch.maximum(valid_depth_pred / valid_depth_gt, valid_depth_gt / valid_depth_pred)

    delta1 = ((ratio < 1.25).sum().item() / num_px) * 100.0
    delta2 = ((ratio < 1.25 ** 2).sum().item() / num_px) * 100.0
    delta3 = ((ratio < 1.25 ** 3).sum().item() / num_px) * 100.0

    return {
        'abs_rel':  abs_rel,
        'sq_rel':   sq_rel,
        'rmse':     rmse,
        'rmse_log': rmse_log,
        'delta1':   delta1,
        'delta2':   delta2,
        'delta3':   delta3,
        'num_px':   num_px,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_hamlyn(model_path, data_root, config_path=None, device='cuda', iters=32):
    """
    Evaluate model on Hamlyn dataset.

    Computes both disparity-level metrics (MAE, Bad-1/2/3, D1) and
    depth-level metrics (Abs Rel, Sq Rel, RMSE, RMSE_log).

    Args:
        model_path:   Path to the model checkpoint
        data_root:    Root directory of Hamlyn dataset
        config_path:  Path to config file (optional)
        device:       Device to use ('cuda' or 'cpu')
        iters:        Number of iterations for model inference

    Returns:
        Dictionary with aggregate and per-sample metrics
    """
    # ---- Load model ----
    print("Loading model from:", model_path)
    if config_path is None:
        config_path = './config/train_scared.yaml'
    if not osp.exists(config_path):
        raise FileNotFoundError("Config file not found: {}".format(config_path))

    cfg = OmegaConf.load(config_path)
    model = EndoStereoFM(cfg)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # ---- Load dataset ----
    print("Loading Hamlyn dataset from:", data_root)
    aug_params = {}
    val_dataset = datasets.Hamlyn(aug_params, root=data_root, image_set='test')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    print("Dataset loaded! Total samples: {}".format(len(val_dataset)))
    print("Calibration: calibration/<seq_id>/ (intrinsics.txt + extrinsics.txt)")
    print("Depth formula: depth = fx * baseline / disparity")

    # ---- Accumulation variables ----
    # Disparity
    tot_disp_px    = 0
    tot_disp_mae   = 0.0
    tot_disp_bad1  = 0.0
    tot_disp_bad2  = 0.0
    tot_disp_bad3  = 0.0
    tot_disp_d1    = 0.0
    # Depth
    tot_depth_px       = 0
    tot_depth_abs_rel  = 0.0
    tot_depth_sq_rel   = 0.0
    tot_depth_rmse     = 0.0
    tot_depth_rmse_log = 0.0
    tot_depth_delta1   = 0.0
    tot_depth_delta2   = 0.0
    tot_depth_delta3   = 0.0
    # Per-sample
    sample_names      = []
    seq_ids           = []
    disp_mae_list     = []
    abs_rel_list      = []
    sq_rel_list       = []
    rmse_list         = []
    rmse_log_list     = []
    delta1_list       = []
    delta2_list       = []
    delta3_list       = []

    # ---- Inference ----
    print("\nStarting evaluation...")
    for batch_idx, data in enumerate(tqdm(val_loader, dynamic_ncols=True)):
        img_info, left, right, disp_gt, valid, calib = [x for x in data]

        # Extract image path and sample name
        if isinstance(img_info, (list, tuple)) and len(img_info) > 0:
            image_path = img_info[0]
        else:
            print("Warning: Unexpected img_info format at batch {}: {}".format(batch_idx, type(img_info)))
            continue

        if isinstance(image_path, torch.Tensor):
            image_path = image_path.item() if image_path.numel() == 1 else str(image_path)
        elif isinstance(image_path, bytes):
            image_path = image_path.decode('utf-8')
        else:
            image_path = str(image_path)

        sample_name = os.path.splitext(os.path.basename(image_path))[0]

        # Extract sequence ID from path for grouping
        seq_id = None
        parts = image_path.replace('\\', '/').split('/')
        for i, p in enumerate(parts):
            if p.startswith('rectified'):
                try:
                    seq_id = int(parts[i + 1])
                except (IndexError, ValueError):
                    pass
                break
        if seq_id is None:
            seq_id = -1

        # Get fx and baseline from calib
        fx = calib['fx']
        baseline = calib['baseline']
        if isinstance(fx, torch.Tensor):
            fx = fx.item() if fx.numel() == 1 else float(fx)
        if isinstance(baseline, torch.Tensor):
            baseline = baseline.item() if baseline.numel() == 1 else float(baseline)

        # Move to device
        left = left.to(device)
        right = right.to(device)
        disp_gt = disp_gt.to(device)
        valid = valid.to(device)

        # Model inference
        padder = InputPadder(left.shape, divis_by=32)
        left_padded, right_padded = padder.pad(left, right)

        with torch.no_grad():
            disp_pred = model(left_padded, right_padded, iters=iters, test_mode=True)

        disp_pred = padder.unpad(disp_pred)

        # ---- Disparity metrics ----
        if valid.dim() == 3:
            valid = valid.unsqueeze(1)
        valid_mask = (valid >= 0.5).squeeze(0)          # (H, W)

        disp_metrics = compute_disparity_metrics(disp_pred, disp_gt, valid_mask)
        if disp_metrics is not None:
            sample_names.append(sample_name)
            seq_ids.append(seq_id)
            disp_mae_list.append(disp_metrics['mae'])

            tot_disp_px   += disp_metrics['num_px']
            tot_disp_mae  += disp_metrics['mae']   * disp_metrics['num_px']
            tot_disp_bad1 += disp_metrics['bad1']  * disp_metrics['num_px']
            tot_disp_bad2 += disp_metrics['bad2']  * disp_metrics['num_px']
            tot_disp_bad3 += disp_metrics['bad3']  * disp_metrics['num_px']
            tot_disp_d1   += disp_metrics['d1']    * disp_metrics['num_px']

        # ---- Depth metrics ----
        depth_pred = disp2depth(disp_pred, fx, baseline)
        depth_gt   = disp2depth(disp_gt,   fx, baseline)

        depth_metrics = compute_depth_metrics(depth_pred, depth_gt, valid_mask)
        if depth_metrics is not None:
            abs_rel_list.append(depth_metrics['abs_rel'])
            sq_rel_list.append(depth_metrics['sq_rel'])
            rmse_list.append(depth_metrics['rmse'])
            rmse_log_list.append(depth_metrics['rmse_log'])
            delta1_list.append(depth_metrics['delta1'])
            delta2_list.append(depth_metrics['delta2'])
            delta3_list.append(depth_metrics['delta3'])
            tot_depth_px       += depth_metrics['num_px']
            tot_depth_abs_rel  += depth_metrics['abs_rel']  * depth_metrics['num_px']
            tot_depth_sq_rel   += depth_metrics['sq_rel']   * depth_metrics['num_px']
            tot_depth_rmse     += depth_metrics['rmse']     * depth_metrics['num_px']
            tot_depth_rmse_log += depth_metrics['rmse_log'] * depth_metrics['num_px']
            tot_depth_delta1   += depth_metrics['delta1']   * depth_metrics['num_px']
            tot_depth_delta2   += depth_metrics['delta2']   * depth_metrics['num_px']
            tot_depth_delta3   += depth_metrics['delta3']   * depth_metrics['num_px']

        # Progress log
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            cur_disp_mae = tot_disp_mae / tot_disp_px if tot_disp_px > 0 else 0
            cur_abs_rel = tot_depth_abs_rel / tot_depth_px if tot_depth_px > 0 else 0
            cur_sq_rel = tot_depth_sq_rel / tot_depth_px if tot_depth_px > 0 else 0
            cur_rmse = tot_depth_rmse / tot_depth_px if tot_depth_px > 0 else 0
            cur_rmse_log = tot_depth_rmse_log / tot_depth_px if tot_depth_px > 0 else 0
            cur_delta1 = tot_depth_delta1 / tot_depth_px if tot_depth_px > 0 else 0
            print("\nSample {}/{}:".format(batch_idx + 1, len(val_loader)))
            print("  [Current]  disp_MAE={:.4f}  abs_rel={:.4f}  sq_rel={:.4f}  rmse={:.4f}  delta1={:.2f}%".format(
                disp_mae_list[-1], abs_rel_list[-1], sq_rel_list[-1], rmse_list[-1], delta1_list[-1]))
            print("  [Average]  disp_MAE={:.4f}  abs_rel={:.4f}  sq_rel={:.4f}  rmse={:.4f}  rmse_log={:.4f}  delta1={:.2f}%".format(
                cur_disp_mae, cur_abs_rel, cur_sq_rel, cur_rmse, cur_rmse_log, cur_delta1))

    # ---- Aggregate results ----
    if tot_disp_px == 0:
        print("No valid samples found for evaluation!")
        return None

    n = len(sample_names)

    disp_mae_avg  = tot_disp_mae  / tot_disp_px
    disp_bad1_avg = tot_disp_bad1 / tot_disp_px
    disp_bad2_avg = tot_disp_bad2 / tot_disp_px
    disp_bad3_avg = tot_disp_bad3 / tot_disp_px
    disp_d1_avg   = tot_disp_d1   / tot_disp_px

    depth_abs_rel_avg  = tot_depth_abs_rel  / tot_depth_px
    depth_sq_rel_avg   = tot_depth_sq_rel   / tot_depth_px
    depth_rmse_avg     = tot_depth_rmse     / tot_depth_px
    depth_rmse_log_avg = tot_depth_rmse_log / tot_depth_px
    depth_delta1_avg   = tot_depth_delta1   / tot_depth_px
    depth_delta2_avg   = tot_depth_delta2   / tot_depth_px
    depth_delta3_avg   = tot_depth_delta3   / tot_depth_px

    # ---- Print summary ----
    print("\n" + "=" * 80)
    print("HAMLYN EVALUATION RESULTS")
    print("=" * 80)
    print("{:10s} {:>10s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}".format('', 'MAE', 'Bad-1%', 'Bad-2%', 'Bad-3%', 'D1%'))
    print("{:10s} {:10.4f}  {:8.2f}  {:8.2f}  {:8.2f}  {:8.2f}".format(
        'Disparity', disp_mae_avg, disp_bad1_avg, disp_bad2_avg, disp_bad3_avg, disp_d1_avg))
    print("-" * 80)
    print("{:10s} {:>10s}  {:>10s}  {:>10s}  {:>10s}".format('', 'Abs Rel', 'Sq Rel', 'RMSE', 'RMSE_log'))
    print("{:10s} {:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}".format(
        'Depth', depth_abs_rel_avg, depth_sq_rel_avg, depth_rmse_avg, depth_rmse_log_avg))
    print("-" * 80)
    print("{:10s} {:>10s}  {:>10s}  {:>10s}".format('', 'd1 (%)', 'd2 (%)', 'd3 (%)'))
    print("{:10s} {:10.2f}  {:10.2f}  {:10.2f}".format(
        'Accuracy', depth_delta1_avg, depth_delta2_avg, depth_delta3_avg))
    print("=" * 80)
    print("Samples evaluated: {}".format(n))
    print("=" * 80)

    # ---- Per-sample table ----
    print("\nPer-sample metrics:")
    print("{:30s}  {:>4s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>8s}  {:>8s}  {:>8s}".format(
        'Sample', 'Seq', 'Disp MAE', 'Abs Rel', 'Sq Rel', 'RMSE', 'RMSE_log', 'd1', 'd2', 'd3'))
    print("-" * 130)
    for i, name in enumerate(sample_names):
        dm   = disp_mae_list[i]   if i < len(disp_mae_list)  else float('nan')
        ar   = abs_rel_list[i]    if i < len(abs_rel_list)    else float('nan')
        sr   = sq_rel_list[i]     if i < len(sq_rel_list)     else float('nan')
        rm   = rmse_list[i]       if i < len(rmse_list)       else float('nan')
        rml  = rmse_log_list[i]   if i < len(rmse_log_list)   else float('nan')
        d1   = delta1_list[i]     if i < len(delta1_list)     else float('nan')
        d2   = delta2_list[i]     if i < len(delta2_list)     else float('nan')
        d3   = delta3_list[i]     if i < len(delta3_list)     else float('nan')
        sid  = seq_ids[i] if i < len(seq_ids) else -1
        print("  {:28s}  {:4d}  {:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}  {:8.2f}  {:8.2f}  {:8.2f}".format(
            name, sid, dm, ar, sr, rm, rml, d1, d2, d3))
    print("=" * 130)

    # ---- Per-sequence summary ----
    unique_seqs = sorted(set(seq_ids))
    if len(unique_seqs) > 1:
        print("\nPer-sequence summary:")
        print("{:>6s}  {:>6s}  {:>10s}  {:>10s}  {:>8s}".format('Seq', 'Count', 'Disp MAE', 'Abs Rel', 'd1'))
        print("-" * 50)
        for sid in unique_seqs:
            indices = [i for i, s in enumerate(seq_ids) if s == sid]
            count = len(indices)
            seq_disp_mae = np.mean([disp_mae_list[i] for i in indices]) if indices else 0
            seq_abs_rel = np.mean([abs_rel_list[i] for i in indices]) if indices else 0
            seq_delta1 = np.mean([delta1_list[i] for i in indices]) if indices else 0
            print("  {:4d}   {:6d}  {:10.4f}  {:10.4f}  {:8.2f}".format(
                sid, count, seq_disp_mae, seq_abs_rel, seq_delta1))
        print("-" * 50)

    results = {
        # Disparity
        'disp_mae_avg':   disp_mae_avg,
        'disp_bad1_avg':  disp_bad1_avg,
        'disp_bad2_avg':  disp_bad2_avg,
        'disp_bad3_avg':  disp_bad3_avg,
        'disp_d1_avg':    disp_d1_avg,
        # Depth
        'abs_rel_avg':   depth_abs_rel_avg,
        'sq_rel_avg':    depth_sq_rel_avg,
        'rmse_avg':      depth_rmse_avg,
        'rmse_log_avg':  depth_rmse_log_avg,
        'delta1_avg':    depth_delta1_avg,
        'delta2_avg':    depth_delta2_avg,
        'delta3_avg':    depth_delta3_avg,
        # Counts
        'num_samples':   n,
        'num_disp_px':   tot_disp_px,
        'num_depth_px':  tot_depth_px,
        # Per-sample
        'sample_names':  sample_names,
        'seq_ids':       seq_ids,
        'disp_mae_list': disp_mae_list,
        'abs_rel_list':  abs_rel_list,
        'sq_rel_list':   sq_rel_list,
        'rmse_list':     rmse_list,
        'rmse_log_list': rmse_log_list,
        'delta1_list':   delta1_list,
        'delta2_list':   delta2_list,
        'delta3_list':   delta3_list,
    }

    # Save JSON (exclude per-sample lists for brevity)
    saveable = {k: v for k, v in results.items()
                if k not in ['sample_names', 'seq_ids', 'disp_mae_list', 'abs_rel_list',
                             'sq_rel_list', 'rmse_list', 'rmse_log_list',
                             'delta1_list', 'delta2_list', 'delta3_list']}
    output_path = osp.join(osp.dirname(model_path), 'hamlyn_evaluation_results_iters{}.json'.format(iters))
    with open(output_path, 'w') as f:
        json.dump(saveable, f, indent=4)
    print("\nResults saved to: {}".format(output_path))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate model on Hamlyn dataset')
    parser.add_argument('--model_path',  type=str, default='/media/Storage3/ug_zy/random_crop/Stereo_Match/checkpoints/scared_320256/14000.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_root',  type=str, default='/media/Storage3/ug_zy/hamlyn',
                        help='Root directory of Hamlyn dataset')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--device',      type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--iters',        type=int, default=32,
                        help='Number of iterations for inference')

    args = parser.parse_args()

    evaluate_hamlyn(
        model_path=args.model_path,
        data_root=args.data_root,
        config_path=args.config_path,
        device=args.device,
        iters=args.iters,
    )

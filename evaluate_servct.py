"""
Evaluate model on SERV-CT dataset.
Computes disparity-level metrics (MAE, D1, Bad-X) and depth-level metrics (MAE, MSE, RMSE).

SERV-CT calibration is read per-frame from Rectified_calibration/<stem>.json.
"""

import os
import json
import torch
import numpy as np
import cv2
import os.path as osp
from tqdm import tqdm
from pathlib import Path

import core.stereo_datasets as datasets
from core.endo_stereo_fm import EndoStereoFM
from core.utils.utils import InputPadder
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def get_Q(calib_path):
    """
    Read Q matrix from SERV-CT / SCARED-style calibration JSON (OpenCV matrix in "Q" field).

    Args:
        calib_path: Path to e.g. Rectified_calibration/001.json

    Returns:
        Q matrix as numpy array (4x4)
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path, mode='r') as rf:
        calib = json.load(rf)
    q = calib['Q']
    rows, cols = int(q['rows']), int(q['cols'])
    Q = np.array(q['data'], dtype=np.float64).reshape(rows, cols)
    return Q


def batch_calib_to_Q(calib):
    """Convert collated calib from DataLoader to numpy Q (4x4)."""
    if calib is None:
        return None
    if isinstance(calib, torch.Tensor):
        return calib.squeeze(0).detach().cpu().numpy().astype(np.float64)
    if isinstance(calib, (list, tuple)) and len(calib) == 1:
        return batch_calib_to_Q(calib[0])
    return np.asarray(calib, dtype=np.float64)


# ---------------------------------------------------------------------------
# Disparity <-> Depth conversion
# ---------------------------------------------------------------------------

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


def compute_depth_metrics(depth_pred, depth_gt, valid_mask, depth_max=256.0):
    """
    Compute SERV-CT-standard depth metrics including delta accuracy.

    Args:
        depth_pred: predicted depth tensor (1, H, W)
        depth_gt:   ground-truth depth tensor (1, H, W)
        valid_mask: boolean tensor (H, W) indicating valid GT pixels
        depth_max:  cap for depth values (default 1000 mm)

    Returns:
        dict with keys: abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3, num_px
    """
    depth_pred = depth_pred.squeeze(0)                  # (H, W)
    depth_gt   = depth_gt.squeeze(0)

    # Start from valid GT pixels
    valid_depth_gt   = depth_gt[valid_mask]
    valid_depth_pred = depth_pred[valid_mask]

    # Filter: both > 0 and < depth_max
    keep = (valid_depth_gt > 0) & (valid_depth_pred > 0) & \
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
def evaluate_servct(model_path, data_root, config_path=None, device='cuda', iters=32):
    """
    Evaluate model on SERV-CT dataset.

    Computes both disparity-level metrics (MAE, Bad-1/2/3, D1) and
    depth-level metrics (Abs Rel, Sq Rel, RMSE, RMSE_log).

    Args:
        model_path:   Path to the model checkpoint
        data_root:    Root directory of SERV-CT dataset
        config_path:  Path to config file (optional)
        device:      Device to use ('cuda' or 'cpu')
        iters:       Number of iterations for model inference

    Returns:
        Dictionary with aggregate and per-sample metrics
    """
    # ---- Load model ----
    print("Loading model from:", model_path)
    if config_path is None:
        config_path = './config/train_scared.yaml'
    if not osp.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

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
    print("Loading SERV-CT dataset from:", data_root)
    aug_params = {}
    val_dataset = datasets.SERVCT(aug_params, root=data_root, image_set='test')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    print(f"Dataset loaded! Total samples: {len(val_dataset)}")
    print("Calibration: Rectified_calibration/<stem>.json (per-frame, matching left image name)")

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
    disp_mae_list     = []
    abs_rel_list  = []
    sq_rel_list   = []
    rmse_list     = []
    rmse_log_list = []
    delta1_list   = []
    delta2_list   = []
    delta3_list   = []

    # ---- Inference ----
    print("\nStarting evaluation...")
    for batch_idx, data in enumerate(tqdm(val_loader, dynamic_ncols=True)):
        img_info, left, right, disp_gt, valid, calib = [x for x in data]

        # Extract image path and sample name
        if isinstance(img_info, (list, tuple)) and len(img_info) > 0:
            image_path = img_info[0]
        else:
            print(f"Warning: Unexpected img_info format at batch {batch_idx}: {type(img_info)}")
            continue

        if isinstance(image_path, torch.Tensor):
            image_path = image_path.item() if image_path.numel() == 1 else str(image_path)
        elif isinstance(image_path, bytes):
            image_path = image_path.decode('utf-8')
        else:
            image_path = str(image_path)

        sample_name = os.path.splitext(os.path.basename(image_path))[0]

        # Load Q matrix
        Q = batch_calib_to_Q(calib)
        if Q is None or Q.size == 0:
            calib_path = osp.join(data_root, 'Rectified_calibration', f'{sample_name}.json')
            try:
                Q = get_Q(calib_path)
            except FileNotFoundError as e:
                print(f"Warning: {e} — skip batch {batch_idx}")
                continue

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
            disp_mae_list.append(disp_metrics['mae'])

            tot_disp_px   += disp_metrics['num_px']
            tot_disp_mae  += disp_metrics['mae']   * disp_metrics['num_px']
            tot_disp_bad1 += disp_metrics['bad1']  * disp_metrics['num_px']
            tot_disp_bad2 += disp_metrics['bad2']  * disp_metrics['num_px']
            tot_disp_bad3 += disp_metrics['bad3']  * disp_metrics['num_px']
            tot_disp_d1   += disp_metrics['d1']    * disp_metrics['num_px']

        # ---- Depth metrics ----
        depth_pred = disp2depth(disp_pred, Q)
        depth_gt   = disp2depth(disp_gt,   Q)

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
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            cur_disp_mae = tot_disp_mae / tot_disp_px if tot_disp_px > 0 else 0
            cur_abs_rel = tot_depth_abs_rel / tot_depth_px if tot_depth_px > 0 else 0
            cur_delta1 = tot_depth_delta1 / tot_depth_px if tot_depth_px > 0 else 0
            print(f"\nSample {batch_idx+1}/{len(val_loader)}: "
                  f"disp_MAE={disp_mae_list[-1]:.4f}  abs_rel={abs_rel_list[-1]:.4f}  delta1={delta1_list[-1]:.2f}%  "
                  f"[avg] disp_MAE={cur_disp_mae:.4f}  abs_rel={cur_abs_rel:.4f}  delta1={cur_delta1:.2f}%")

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
    print("\n" + "=" * 70)
    print("SERV-CT EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'':10s} {'MAE':>10s}  {'Bad-1%':>8s}  {'Bad-2%':>8s}  {'Bad-3%':>8s}  {'D1%':>8s}")
    print(f"{'Disparity':10s} {disp_mae_avg:10.4f}  {disp_bad1_avg:8.2f}  {disp_bad2_avg:8.2f}  "
          f"{disp_bad3_avg:8.2f}  {disp_d1_avg:8.2f}")
    print("-" * 70)
    print(f"{'':10s} {'Abs Rel':>10s}  {'Sq Rel':>10s}  {'RMSE':>10s}  {'RMSE_log':>10s}")
    print(f"{'Depth':10s} {depth_abs_rel_avg:10.4f}  {depth_sq_rel_avg:10.4f}  "
          f"{depth_rmse_avg:10.4f}  {depth_rmse_log_avg:10.4f}")
    print("-" * 70)
    print(f"{'':10s} {'d1 (%)':>10s}  {'d2 (%)':>10s}  {'d3 (%)':>10s}")
    print(f"{'Accuracy':10s} {depth_delta1_avg:10.2f}  {depth_delta2_avg:10.2f}  {depth_delta3_avg:10.2f}")
    print("=" * 70)
    print(f"Samples evaluated: {n}")
    print("=" * 70)

    # ---- Per-sample table ----
    print("\nPer-sample metrics:")
    print(f"{'Sample':30s}  {'Disp MAE':>10s}  {'Abs Rel':>10s}  {'Sq Rel':>10s}  {'RMSE':>10s}  {'RMSE_log':>10s}  {'d1':>8s}  {'d2':>8s}  {'d3':>8s}")
    print("-" * 104)
    for i, name in enumerate(sample_names):
        dm   = disp_mae_list[i]   if i < len(disp_mae_list)  else float('nan')
        ar   = abs_rel_list[i]    if i < len(abs_rel_list)    else float('nan')
        sr   = sq_rel_list[i]    if i < len(sq_rel_list)     else float('nan')
        rm   = rmse_list[i]      if i < len(rmse_list)      else float('nan')
        rml  = rmse_log_list[i]  if i < len(rmse_log_list)  else float('nan')
        d1   = delta1_list[i]    if i < len(delta1_list)    else float('nan')
        d2   = delta2_list[i]    if i < len(delta2_list)    else float('nan')
        d3   = delta3_list[i]    if i < len(delta3_list)    else float('nan')
        print(f"  {name:28s}  {dm:10.4f}  {ar:10.4f}  {sr:10.4f}  {rm:10.4f}  {rml:10.4f}  {d1:8.2f}  {d2:8.2f}  {d3:8.2f}")
    print("=" * 104)

    results = {
        # Disparity
        'disp_mae_avg':   disp_mae_avg,
        'disp_bad1_avg':  disp_bad1_avg,
        'disp_bad2_avg':  disp_bad2_avg,
        'disp_bad3_avg':  disp_bad3_avg,
        'disp_d1_avg':    disp_d1_avg,
        # Depth
        'abs_rel_avg':  depth_abs_rel_avg,
        'sq_rel_avg':   depth_sq_rel_avg,
        'rmse_avg':     depth_rmse_avg,
        'rmse_log_avg': depth_rmse_log_avg,
        'delta1_avg':   depth_delta1_avg,
        'delta2_avg':   depth_delta2_avg,
        'delta3_avg':   depth_delta3_avg,
        # Counts
        'num_samples':    n,
        'num_disp_px':   tot_disp_px,
        'num_depth_px':  tot_depth_px,
        # Per-sample
        'sample_names':  sample_names,
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
                if k not in ['sample_names', 'disp_mae_list', 'abs_rel_list',
                             'sq_rel_list', 'rmse_list', 'rmse_log_list',
                             'delta1_list', 'delta2_list', 'delta3_list']}
    output_path = osp.join(osp.dirname(model_path), 'servct_evaluation_results.json')
    with open(output_path, 'w') as f:
        json.dump(saveable, f, indent=4)
    print(f"\nResults saved to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate model on SERV-CT dataset')
    parser.add_argument('--model_path',  type=str, default='/media/Storage3/ug_zy/random_crop/Stereo_Match/checkpoints/scared_320256/14000.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_root',  type=str, default='/media/Storage3/ug_zy/SERV-CT/',
                        help='Root directory of SERV-CT dataset')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--device',      type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--iters',        type=int, default=32,
                        help='Number of iterations for inference')

    args = parser.parse_args()

    evaluate_servct(
        model_path=args.model_path,
        data_root=args.data_root,
        config_path=args.config_path,
        device=args.device,
        iters=args.iters,
    )

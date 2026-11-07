"""Evaluate Manifold-Stereo on the directory-level SCARED test set.

The valid ground-truth disparity range is [0.5, 192). Metrics are computed per
frame, averaged over frames inside each ``dataset_*/keyframe_*`` group, and
then averaged over keyframes for the dataset and overall summaries.
"""

import argparse
import csv
import json
import logging
import os
import os.path as osp
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import core.stereo_datasets as datasets
from core.endo_stereo_fm import EndoStereoFM
from core.utils.utils import InputPadder


MIN_VALID_DISP = 0.5
MAX_VALID_DISP = 192.0
DISPARITY_METRICS = ('EPE', 'Bad1', 'Bad2', 'Bad3', 'Bad5')
DEPTH_METRICS = ('Depth_MAE', 'Abs_Rel', 'Sq_Rel', 'RMSE', 'RMSE_log', 'Delta1', 'Delta2', 'Delta3')
ALL_METRICS = DISPARITY_METRICS + DEPTH_METRICS


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


def _unwrap_path(value):
    """Unwrap a path from the tuple/list structure created by DataLoader."""
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError('Empty path container returned by DataLoader')
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return str(value)


def get_sample_group(image_path):
    """Return ``(dataset_*, keyframe_*)`` from a SCARED image path."""
    parts = Path(image_path.replace('\\', '/')).parts
    dataset_name = next((p for p in parts if p.startswith('dataset_')), None)
    keyframe_name = next((p for p in parts if p.startswith('keyframe_')), None)
    if dataset_name is None or keyframe_name is None:
        raise ValueError(f'Cannot extract dataset/keyframe from {image_path}')
    return dataset_name, keyframe_name


def get_Q_from_path(data_root, image_path):
    """Read the SCARED Q matrix belonging to an image path."""
    image_path = osp.abspath(image_path)
    parts = Path(image_path).parts
    dataset_name = next((p for p in parts if p.startswith('dataset_')), None)
    keyframe_name = next((p for p in parts if p.startswith('keyframe_')), None)
    data_split = next((p for p in parts if p in ('train_data', 'test_data')), None)
    if dataset_name is None or keyframe_name is None or data_split is None:
        raise ValueError(f'Cannot locate SCARED calibration for {image_path}')

    calib_path = osp.join(data_root, data_split, dataset_name, keyframe_name, 'stereo_calib.json')
    if not osp.exists(calib_path):
        raise FileNotFoundError(f'Calibration file not found: {calib_path}')
    with open(calib_path, encoding='utf-8') as rf:
        calib = json.load(rf)
    q = calib['Q']
    return np.asarray(q['data'], dtype=np.float64).reshape(q['rows'], q['cols'])


def disp2depth(disp, q_matrix):
    """Convert a disparity tensor to depth using an OpenCV Q matrix."""
    if disp.ndim == 4:
        disp_np = disp[:, 0].detach().cpu().numpy()
    elif disp.ndim == 3:
        disp_np = disp.detach().cpu().numpy()
    else:
        disp_np = disp.detach().cpu().numpy()[None]

    depth = []
    for disparity in disp_np:
        xyz = cv2.reprojectImageTo3D(disparity.astype(np.float32), q_matrix)
        depth.append(np.nan_to_num(xyz[:, :, 2], nan=0.0, posinf=0.0, neginf=0.0))
    depth_tensor = torch.from_numpy(np.stack(depth)).to(device=disp.device, dtype=disp.dtype)
    return depth_tensor[:, None]


def disparity_metrics(disp_pred, disp_gt):
    """Compute EPE and Bad-X rates on [0.5, 192) ground-truth disparities.

    Bad-X values are rates in [0, 1], matching the referenced StereoMamba
    evaluator.
    """
    valid = (torch.isfinite(disp_gt) & torch.isfinite(disp_pred) &
             (disp_gt >= MIN_VALID_DISP) & (disp_gt < MAX_VALID_DISP))
    if valid.sum() == 0:
        return None, valid
    valid_error = torch.abs(disp_pred - disp_gt)[valid]
    return {
        'EPE': valid_error.mean().item(),
        'Bad1': (valid_error > 1.0).float().mean().item(),
        'Bad2': (valid_error > 2.0).float().mean().item(),
        'Bad3': (valid_error > 3.0).float().mean().item(),
        'Bad5': (valid_error > 5.0).float().mean().item(),
    }, valid


def depth_metrics(depth_pred, depth_gt, valid_disp):
    """Compute depth metrics on valid disparity pixels."""
    valid = valid_disp & torch.isfinite(depth_pred) & torch.isfinite(depth_gt)
    valid = valid & (depth_pred > 0) & (depth_gt > 0)
    if valid.sum() == 0:
        return None

    pred = depth_pred[valid]
    gt = depth_gt[valid]
    abs_error = torch.abs(pred - gt)
    ratio = torch.maximum(pred / gt, gt / pred)
    log_error = torch.log(pred) - torch.log(gt)
    return {
        'Depth_MAE': abs_error.mean().item(),
        'Abs_Rel': (abs_error / gt).mean().item(),
        'Sq_Rel': ((abs_error ** 2) / gt).mean().item(),
        'RMSE': torch.sqrt((abs_error ** 2).mean()).item(),
        'RMSE_log': torch.sqrt((log_error ** 2).mean()).item(),
        'Delta1': (ratio < 1.25).float().mean().item(),
        'Delta2': (ratio < 1.25 ** 2).float().mean().item(),
        'Delta3': (ratio < 1.25 ** 3).float().mean().item(),
    }


def _mean_rows(rows):
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in ALL_METRICS}


def _write_rows(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_scared(model_path, data_root, config_path=None, device='cuda', iters=32,
                    test_mode=True, batch_size=1, logger=None):
    """Evaluate SCARED and return frame, keyframe, dataset, and overall metrics."""
    def _log(message, *args):
        if logger is not None:
            logger.info(message, *args)
        else:
            print(message % args if args else message)
    if batch_size != 1:
        raise ValueError('SCARED evaluation requires batch_size=1 for per-frame calibration and grouping')

    if config_path is None:
        config_path = './config/train_scared.yaml'
    if not osp.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')

    _log('Loading model from: %s', model_path)
    cfg = OmegaConf.load(config_path)
    model = EndoStereoFM(cfg)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device).eval()

    _log('Loading SCARED test dataset from: %s', data_root)
    val_dataset = datasets.SCARED({}, root=data_root, image_set='test')
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, pin_memory=True, shuffle=False,
        num_workers=1, drop_last=False,
    )
    _log('Dataset loaded! Total samples: %d', len(val_dataset))

    frame_rows = []
    grouped_rows = defaultdict(list)
    skipped = 0
    for batch_idx, data in enumerate(tqdm(val_loader, dynamic_ncols=True)):
        img_info, left, right, disp_gt, _ = data
        try:
            image_path = _unwrap_path(img_info[0])
            dataset_name, keyframe_name = get_sample_group(image_path)
            q_matrix = get_Q_from_path(data_root, image_path)
        except (TypeError, ValueError, FileNotFoundError, KeyError) as exc:
            _log('Skipping sample %d: %s', batch_idx, exc)
            skipped += 1
            continue

        left, right = left.to(device), right.to(device)
        disp_gt = disp_gt.to(device)
        padder = InputPadder(left.shape, divis_by=32)
        left_pad, right_pad = padder.pad(left, right)
        disp_pred = model(left_pad, right_pad, iters=iters, test_mode=test_mode)
        disp_pred = padder.unpad(disp_pred)

        pred_map = disp_pred[0, 0]
        gt_map = disp_gt[0, 0]
        disp_result, valid_disp = disparity_metrics(pred_map, gt_map)
        if disp_result is None:
            skipped += 1
            continue

        depth_pred = disp2depth(disp_pred, q_matrix)[0, 0]
        depth_gt = disp2depth(disp_gt, q_matrix)[0, 0]
        depth_result = depth_metrics(depth_pred, depth_gt, valid_disp)
        if depth_result is None:
            skipped += 1
            continue

        row = {'dataset': dataset_name, 'keyframe': keyframe_name,
               'frame': osp.splitext(osp.basename(image_path))[0]}
        row.update(disp_result)
        row.update(depth_result)
        frame_rows.append(row)
        grouped_rows[(dataset_name, keyframe_name)].append(row)

        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            _log('%s/%s/%s: EPE=%.4f Bad1=%.4f Bad2=%.4f Bad3=%.4f Bad5=%.4f Depth_MAE=%.4f',
                 dataset_name, keyframe_name, row['frame'], row['EPE'], row['Bad1'],
                 row['Bad2'], row['Bad3'], row['Bad5'], row['Depth_MAE'])

    if not frame_rows:
        _log('No valid samples found for evaluation!')
        return None

    keyframe_rows = []
    for (dataset_name, keyframe_name), rows in sorted(grouped_rows.items()):
        keyframe_rows.append({
            'dataset': dataset_name,
            'keyframe': keyframe_name,
            'num_frames': len(rows),
            **_mean_rows(rows),
        })

    dataset_rows = []
    by_dataset = defaultdict(list)
    for row in keyframe_rows:
        by_dataset[row['dataset']].append(row)
    for dataset_name, rows in sorted(by_dataset.items()):
        dataset_rows.append({
            'dataset': dataset_name,
            'num_keyframes': len(rows),
            **_mean_rows(rows),
        })
    overall = _mean_rows(keyframe_rows)

    def log_metric_row(label, row):
        _log(
            '%s: EPE=%.4f Bad1=%.4f Bad2=%.4f Bad3=%.4f Bad5=%.4f '
            'Depth_MAE=%.4f Abs_Rel=%.4f Sq_Rel=%.4f RMSE=%.4f '
            'RMSE_log=%.4f Delta1=%.4f Delta2=%.4f Delta3=%.4f',
            label, row['EPE'], row['Bad1'], row['Bad2'], row['Bad3'], row['Bad5'],
            row['Depth_MAE'], row['Abs_Rel'], row['Sq_Rel'], row['RMSE'],
            row['RMSE_log'], row['Delta1'], row['Delta2'], row['Delta3'],
        )

    _log('\n=== SCARED averages by keyframe ===')
    for row in keyframe_rows:
        log_metric_row(f"{row['dataset']}/{row['keyframe']}", row)

    _log('\n=== SCARED averages by dataset (keyframe average) ===')
    for row in dataset_rows:
        log_metric_row(row['dataset'], row)
    _log('\n=== SCARED overall average (keyframe average) ===')
    log_metric_row('Overall', overall)
    _log('Valid frames: %d | keyframes: %d | skipped: %d', len(frame_rows), len(keyframe_rows), skipped)

    return {
        'overall': overall,
        'dataset_averages': dataset_rows,
        'keyframe_averages': keyframe_rows,
        'frame_metrics': frame_rows,
        'num_frames': len(frame_rows),
        'num_keyframes': len(keyframe_rows),
        'num_skipped': skipped,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate disparity/depth on SCARED')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str,
                        default='/public2/mvicgs4/jyangcu/zy/dataset/scared',
                        help='Root directory of SCARED dataset')
    parser.add_argument('--config_path', type=str, default=None, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--iters', type=int, default=32, help='Number of inference iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='Evaluation batch size; must be 1')
    args = parser.parse_args()

    model_stem = osp.splitext(osp.basename(args.model_path))[0]
    log_dir = osp.join(osp.dirname(args.model_path), 'logs')
    log_path = osp.join(log_dir, f'{model_stem}_scared_eval.log')
    logger = setup_logger('scared_eval', log_path)
    logger.info('Model: %s', args.model_path)
    logger.info('Data root: %s', args.data_root)
    logger.info('Valid disparity range: [%.1f, %.1f)', MIN_VALID_DISP, MAX_VALID_DISP)

    results = evaluate_scared(
        model_path=args.model_path,
        data_root=args.data_root,
        config_path=args.config_path,
        device=args.device,
        iters=args.iters,
        batch_size=args.batch_size,
        logger=logger,
    )
    if results is not None:
        output_dir = osp.dirname(args.model_path) or '.'
        json_path = osp.join(output_dir, f'{model_stem}_scared_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        _write_rows(osp.join(output_dir, f'{model_stem}_scared_keyframes.csv'),
                    results['keyframe_averages'], ['dataset', 'keyframe', 'num_frames', *ALL_METRICS])
        _write_rows(osp.join(output_dir, f'{model_stem}_scared_datasets.csv'),
                    results['dataset_averages'], ['dataset', 'num_keyframes', *ALL_METRICS])
        logger.info('Results saved to: %s', json_path)

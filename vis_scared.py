"""
Visualize stereo matching predictions on SCARED, Hamlyn, and SERV-CT datasets.
Converts disparity maps to pseudo-colored images for visual comparison.
"""

import os
import cv2
import numpy as np
import torch
import os.path as osp
from tqdm import tqdm
import logging

import core.stereo_datasets as datasets
from core.endo_stereo_fm import EndoStereoFM
from core.utils.utils import InputPadder
from omegaconf import OmegaConf


# SCARED disparity is stored as uint16 disparity * 64.  The values are
# converted to pixel units by frame_utils.readDispSCARED before they reach the
# visualizer.  Keep the visualization range in the same pixel units used by
# evaluation and by StereoMamba's vis_disp.py reference.
DISP_MIN = 0.0
DISP_MAX = 192.0
SCARED_DISP_SCALE = 64.0
DATASET_DISP_SCALES = {
    'scared': 64.0,
    'hamlyn': 256.0,
    'servct': 256.0,
}


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


def render_disparity(disp, min_disp=DISP_MIN, max_disp=DISP_MAX,
                     colormap=cv2.COLORMAP_JET):
    """Convert a disparity map to a fixed-range JET visualization.

    The normalization is deliberately global and deterministic: a disparity of
    ``min_disp`` always has the same color in every frame, as in
    ``StereoMamba/evaluation_scripts/vis_disp.py``.  Disparities outside the
    display range are clipped; NaN/inf values are rendered black.
    """
    if isinstance(disp, torch.Tensor):
        disp_np = disp.detach().cpu().numpy()
    else:
        disp_np = np.asarray(disp)

    # Accept the common [1, H, W] tensor form while preserving one-row images.
    while disp_np.ndim > 2 and disp_np.shape[0] == 1:
        disp_np = disp_np[0]

    if disp_np.ndim != 2:
        raise ValueError(f'Expected a 2-D disparity map, got shape {disp_np.shape}')
    if not np.isfinite(min_disp) or not np.isfinite(max_disp) or max_disp <= min_disp:
        raise ValueError(f'Invalid disparity display range: [{min_disp}, {max_disp}]')

    finite = np.isfinite(disp_np)
    disp_safe = np.nan_to_num(disp_np, nan=min_disp, posinf=max_disp, neginf=min_disp)
    disp_norm = np.clip((disp_safe - min_disp) / (max_disp - min_disp), 0.0, 1.0)
    disp_uint8 = np.rint(disp_norm * 255.0).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_uint8, colormap)
    disp_color[~finite] = 0
    return disp_color


def render_depth(disp, colormap=cv2.COLORMAP_JET, percentile_clip=None):
    """Backward-compatible alias for older callers of this script."""
    # ``percentile_clip`` is retained for call compatibility; the fixed range
    # is intentional and is used for all new visualizations.
    return render_disparity(disp, colormap=colormap)


def save_raw_disparity(path, disp, scale=SCARED_DISP_SCALE):
    """Save pixel-unit disparity as a uint16 PNG using the dataset scale."""
    if isinstance(disp, torch.Tensor):
        disp = disp.detach().cpu().numpy()
    disp = np.asarray(disp)
    while disp.ndim > 2 and disp.shape[0] == 1:
        disp = disp[0]
    if disp.ndim != 2:
        raise ValueError(f'Expected a 2-D disparity map, got shape {disp.shape}')
    disp = np.nan_to_num(disp, nan=0.0, posinf=65535.0 / scale, neginf=0.0)
    disp = np.clip(disp, 0.0, 65535.0 / scale)
    encoded = np.rint(disp * scale).astype(np.uint16)
    if not cv2.imwrite(path, encoded):
        raise IOError(f'Failed to save disparity PNG: {path}')


def generate_colorbar(path, min_disp=DISP_MIN, max_disp=DISP_MAX):
    """Save a colorbar matching :func:`render_disparity`."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2, 8))
    norm = plt.Normalize(min_disp, max_disp)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), cax=ax)
    cb.set_label('Disparity (pixels)', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)


class StereoVisualizer:
    """
    Visualizer for stereo matching predictions on SCARED, Hamlyn, and SERV-CT datasets.
    """

    def __init__(self, model_path, data_root, dataset_name, config_path=None,
                 device='cuda', iters=32, batch_size=1, logger=None,
                 save_raw_disp=False, save_colorbar=False):
        """
        Initialize the visualizer.

        Args:
            model_path: Path to model checkpoint
            data_root: Root directory of dataset
            dataset_name: 'scared', 'hamlyn', or 'servct'
            config_path: Path to config file
            device: Device to use ('cuda' or 'cpu')
            iters: Number of iterations for inference
            batch_size: Batch size for DataLoader
            logger: logging.Logger instance
            save_raw_disp: Also save uint16 predicted disparity PNGs using the
                source dataset's original encoding.
            save_colorbar: Save one colorbar using the fixed display range.
        """
        self.model_path = model_path
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.config_path = config_path
        self.device = device
        self.iters = iters
        self.batch_size = batch_size
        self.logger = logger
        self.save_raw_disp = save_raw_disp
        self.save_colorbar = save_colorbar
        self.disp_scale = DATASET_DISP_SCALES[dataset_name]

        self.model = None
        self.dataset = None

    def _log(self, msg, *args):
        """Log message."""
        if self.logger:
            self.logger.info(msg % args if args else msg)
        else:
            print(msg % args if args else msg)

    def load_model(self):
        """Load the stereo matching model from checkpoint."""
        self._log("Loading model from: %s", self.model_path)

        if self.config_path is None:
            if self.dataset_name == 'scared':
                self.config_path = './config/train_scared.yaml'
            elif self.dataset_name == 'hamlyn':
                self.config_path = './config/train_scared.yaml'
            else:
                self.config_path = './config/train_scared.yaml'

        if not osp.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        cfg = OmegaConf.load(self.config_path)
        self.model = EndoStereoFM(cfg)
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()
        self._log("Model loaded successfully!")

    def load_dataset(self, hamlyn_first_only=False):
        """Load the test dataset.

        Args:
            hamlyn_first_only: If True, for Hamlyn dataset only keep the first image of each sequence.
        """
        self._log("Loading %s dataset from: %s", self.dataset_name, self.data_root)

        aug_params = {}

        if self.dataset_name == 'scared':
            self.dataset = datasets.SCARED(aug_params, root=self.data_root, image_set='test')
        elif self.dataset_name == 'hamlyn':
            self.dataset = datasets.Hamlyn(aug_params, root=self.data_root, image_set='test')
            if hamlyn_first_only:
                self._filter_hamlyn_first()
        elif self.dataset_name == 'servct':
            self.dataset = datasets.SERVCT(aug_params, root=self.data_root, image_set='test')
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. Available: ['scared', 'hamlyn', 'servct']")

        self._log("Dataset loaded! Total samples: %d", len(self.dataset))

    def _filter_hamlyn_first(self):
        """Keep only the first image of each Hamlyn sequence."""
        seen_seqs = set()
        indices_to_keep = []
        for idx, img_path in enumerate(self.dataset.image_list):
            seq_id = None
            parts = img_path[0].replace('\\', '/').split('/')
            for p in parts:
                if p.startswith('rectified'):
                    seq_id = p
                    break
            if seq_id is not None and seq_id not in seen_seqs:
                seen_seqs.add(seq_id)
                indices_to_keep.append(idx)
        self._log("Hamlyn: keeping only first frame per sequence (%d sequences)", len(indices_to_keep))
        self.dataset.image_list = [self.dataset.image_list[i] for i in indices_to_keep]
        self.dataset.disparity_list = [self.dataset.disparity_list[i] for i in indices_to_keep]
        self.dataset.calib_list = [self.dataset.calib_list[i] for i in indices_to_keep]

    def visualize(self, output_dir, max_samples=None):
        """
        Generate visualizations for the dataset.

        Args:
            output_dir: Directory to save visualizations
            max_samples: Maximum number of samples to process (None for all)

        Returns:
            List of output paths for saved visualizations
        """
        os.makedirs(output_dir, exist_ok=True)

        num_samples = len(self.dataset) if max_samples is None else min(max_samples, len(self.dataset))
        self._log("\nGenerating visualizations for %d samples...", num_samples)

        saved_paths = []

        if self.dataset_name == 'scared' and getattr(self, 'save_colorbar', False):
            colorbar_path = osp.join(output_dir, 'colorbar.png')
            generate_colorbar(colorbar_path)
            saved_paths.append(colorbar_path)

        for idx in tqdm(range(num_samples), desc=f"Visualizing {self.dataset_name}"):
            try:
                paths = self._process_sample(idx, output_dir)
                saved_paths.extend(paths)
            except Exception as e:
                self._log("Error processing sample %d: %s", idx, str(e))
                continue

        self._log("\nVisualization complete! Saved to: %s", output_dir)
        return saved_paths

    def _process_sample(self, idx, output_dir):
        """Process a single sample and save visualizations."""
        saved_paths = []

        # Get sample data based on dataset type
        if self.dataset_name == 'scared':
            img_path, left, right, disp_gt, valid = self._get_scared_sample(idx)
            img_name = self._get_scared_name(img_path)
        elif self.dataset_name == 'hamlyn':
            img_path, left, right, disp_gt, valid = self._get_hamlyn_sample(idx)
            img_name = self._get_hamlyn_name(img_path)
        elif self.dataset_name == 'servct':
            img_path, left, right, disp_gt, valid = self._get_servct_sample(idx)
            img_name = self._get_servct_name(img_path)

        # Preserve the source grouping for SCARED so frames from different
        # keyframes cannot overwrite one another.  This mirrors the directory
        # layout used by StereoMamba's save_disp.py.
        if self.dataset_name == 'scared':
            seq_dir = osp.join(output_dir, img_name['dataset'], img_name['keyframe_dir'])
            os.makedirs(seq_dir, exist_ok=True)
        elif self.dataset_name == 'hamlyn':
            seq_dir = osp.join(output_dir, img_name['sequence'])
            os.makedirs(seq_dir, exist_ok=True)
        else:
            seq_dir = output_dir

        # Move to device
        left = left.unsqueeze(0).to(self.device)
        right = right.unsqueeze(0).to(self.device)
        disp_gt = disp_gt.unsqueeze(0).to(self.device)

        # Model inference
        padder = InputPadder(left.shape, divis_by=32)
        left_padded, right_padded = padder.pad(left, right)

        with torch.no_grad():
            disp_pred = self.model(left_padded, right_padded, iters=self.iters, test_mode=True)

        disp_pred = padder.unpad(disp_pred).squeeze()

        # Convert tensors to numpy for visualization
        left_np = left.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        gt_np = disp_gt.squeeze().cpu().numpy()
        pred_np = disp_pred.cpu().numpy()

        # Build filename based on dataset naming convention
        if self.dataset_name == 'scared':
            # The dataset/keyframe directories already identify the sequence;
            # keep the original frame basename as StereoMamba does.
            filename = img_name['frame_id']
        elif self.dataset_name == 'hamlyn':
            filename = f"{img_name['sequence']}_{img_name['frame_id']}"
        else:  # servct
            filename = f"{img_name['sequence']}_{img_name['frame_id']}"

        # Save GT disparity with the same fixed 0--192 pixel range as the
        # prediction.  Per-image percentile normalization would make colors
        # incomparable across frames.
        gt_color = render_disparity(gt_np)
        gt_path = osp.join(seq_dir, f'{filename}_gt.png')
        cv2.imwrite(gt_path, gt_color)
        saved_paths.append(gt_path)

        # Save predicted disparity
        pred_color = render_disparity(pred_np)
        pred_path = osp.join(seq_dir, f'{filename}_pred.png')
        cv2.imwrite(pred_path, pred_color)
        saved_paths.append(pred_path)

        if self.save_raw_disp:
            raw_pred_path = osp.join(seq_dir, f'{filename}_pred_disp.png')
            save_raw_disparity(raw_pred_path, pred_np, scale=self.disp_scale)
            saved_paths.append(raw_pred_path)

        # Save left image
        left_bgr = cv2.cvtColor(left_np, cv2.COLOR_RGB2BGR)
        left_path = osp.join(seq_dir, f'{filename}_left.png')
        cv2.imwrite(left_path, left_bgr)
        saved_paths.append(left_path)

        return saved_paths

    def _get_scared_sample(self, idx):
        """Load a sample from SCARED dataset."""
        img1_path = self.dataset.image_list[idx][0]
        img2_path = self.dataset.image_list[idx][1]
        disp_path = self.dataset.disparity_list[idx]

        # Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Read disparity
        from core.utils import frame_utils
        disp, valid = frame_utils.readDispSCARED(disp_path)
        disp = np.array(disp).astype(np.float32)

        # Convert to tensors
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float()
        disp_tensor = torch.from_numpy(disp).unsqueeze(0)
        valid_tensor = torch.from_numpy(valid.astype(np.float32))

        return [img1_path, img2_path, disp_path], img1_tensor, img2_tensor, disp_tensor, valid_tensor

    def _get_hamlyn_sample(self, idx):
        """Load a sample from Hamlyn dataset with EndoDAC-style cropping."""
        img1_path = self.dataset.image_list[idx][0]
        img2_path = self.dataset.image_list[idx][1]
        disp_path = self.dataset.disparity_list[idx]

        # Read images using cv2 (consistent with original code)
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None:
            raise RuntimeError(f"Failed to read image: {img1_path}")
        if img2 is None:
            raise RuntimeError(f"Failed to read image: {img2_path}")
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Read disparity using the dataset's disparity_reader
        from core.utils import frame_utils
        disp, valid = self.dataset.disparity_reader(disp_path)
        if disp is None:
            raise RuntimeError(f"Failed to read disparity: {disp_path}")
        disp = np.array(disp).astype(np.float32)

        # Extract sequence ID from path for crop decision
        # Hamlyn path: .../rectified01/image01/0001.png -> rectified01 is the dir name
        seq_id = None
        parts = img1_path.replace('\\', '/').split('/')
        for p in parts:
            if p.startswith('rectified') and len(p) > len('rectified'):
                try:
                    seq_id = int(p[len('rectified'):])  # rectified01 -> 1
                except ValueError:
                    pass
                break

        # EndoDAC-style crop: sequences > 13 have 720px-wide concatenated images
        if seq_id is not None and seq_id > 13:
            img1 = img1[:, 180:590]
            img2 = img2[:, 180:590]
            disp = disp[:, 180:590]
            valid = valid[:, 180:590]

        # Convert to tensors
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float()
        disp_tensor = torch.from_numpy(disp).unsqueeze(0)
        valid_tensor = torch.from_numpy(valid.astype(np.float32))

        return [img1_path, img2_path, disp_path], img1_tensor, img2_tensor, disp_tensor, valid_tensor

    def _get_servct_sample(self, idx):
        """Load a sample from SERV-CT dataset."""
        img1_path = self.dataset.image_list[idx][0]
        img2_path = self.dataset.image_list[idx][1]
        disp_path = self.dataset.disparity_list[idx]

        # Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Read disparity
        from core.utils import frame_utils
        disp, valid = frame_utils.readDispSERVCT(disp_path)
        disp = np.array(disp).astype(np.float32)

        # Convert to tensors
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float()
        disp_tensor = torch.from_numpy(disp).unsqueeze(0)
        valid_tensor = torch.from_numpy(valid.astype(np.float32))

        return [img1_path, img2_path, disp_path], img1_tensor, img2_tensor, disp_tensor, valid_tensor

    def _get_scared_name(self, img_path):
        """Extract naming components from SCARED path.
        
        SCARED path: .../dataset_X/keyframe_Y/data/left_rectified/FFFFFF.png
        Output: {sequence: 'seqX', keyframe: 'kfY', frame_id: 'FFFFFF'}
        """
        # img_path[0] is left image path
        path_str = img_path[0].replace('\\', '/')
        parts = path_str.split('/')
        
        sequence = None
        keyframe = None
        frame_id = None
        
        for part in parts:
            if part.startswith('dataset_'):
                sequence = 'seq' + part.replace('dataset_', '')
            elif part.startswith('keyframe_'):
                keyframe = 'kf' + part.replace('keyframe_', '')
        
        # Frame ID is the filename without extension
        frame_id = osp.splitext(osp.basename(img_path[0]))[0]
        
        return {
            'sequence': sequence,
            'keyframe': keyframe,
            'dataset': next((p for p in parts if p.startswith('dataset_')), sequence),
            'keyframe_dir': next((p for p in parts if p.startswith('keyframe_')), keyframe),
            'frame_id': frame_id,
        }

    def _get_hamlyn_name(self, img_path):
        """Extract naming components from Hamlyn path.
        
        Hamlyn path: .../rectifiedXX/image01/YYYY.png
        Output: {sequence: 'seqXX', frame_id: 'YYYY'}
        """
        path_str = img_path[0].replace('\\', '/')
        parts = path_str.split('/')
        
        sequence = None
        frame_id = None
        
        for part in parts:
            if part.startswith('rectified'):
                sequence = 'seq' + part.replace('rectified', '')
        
        frame_id = osp.splitext(osp.basename(img_path[0]))[0]
        
        return {'sequence': sequence, 'frame_id': frame_id}

    def _get_servct_name(self, img_path):
        """Extract naming components from SERV-CT path.
        
        SERV-CT path: .../Left_rectified/XXXXX.png
        Output: {sequence: 'frame', frame_id: 'XXXXX'}
        """
        frame_id = osp.splitext(osp.basename(img_path[0]))[0]
        return {'sequence': 'frame', 'frame_id': frame_id}

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize stereo matching predictions on SCARED/Hamlyn/SERV-CT')
    parser.add_argument('--model_path', type=str, default='/media/Storage3/ug_zy/random_crop/Stereo_Match/checkpoints/scared_320256/14000.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='/public2/mvicgs4/jyangcu/zy/dataset/scared',
                        help='Root directory of dataset')
    parser.add_argument('--dataset', type=str, required=True, choices=['scared', 'hamlyn', 'servct'],
                        help='Dataset name to visualize')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--iters', type=int, default=32,
                        help='Number of iterations for inference')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to visualize')
    parser.add_argument('--save_raw_disp', action='store_true',
                        help='Save predicted disparity as uint16 PNG using the dataset encoding')
    parser.add_argument('--colorbar', action='store_true',
                        help='Save a colorbar for the fixed disparity range [0, 192]')
    parser.add_argument('--hamlyn_first_only', action='store_true', default=True,
                        help='For Hamlyn dataset, only test the first frame of each sequence (default: True)')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        model_stem = osp.splitext(osp.basename(args.model_path))[0]
        args.output_dir = osp.join(osp.dirname(args.model_path), 'visualizations', args.dataset)

    # Setup logger
    log_dir = osp.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = osp.join(log_dir, 'visualization.log')
    logger = setup_logger('vis_scared', log_path)

    logger.info("=" * 60)
    logger.info("STEREO MATCHING VISUALIZATION")
    logger.info("=" * 60)
    logger.info("Model: %s", args.model_path)
    logger.info("Data root: %s", args.data_root)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Device: %s  iters: %d", args.device, args.iters)
    logger.info("Disparity visualization range: [%.1f, %.1f]", DISP_MIN, DISP_MAX)
    logger.info("=" * 60)

    # Create visualizer
    visualizer = StereoVisualizer(
        model_path=args.model_path,
        data_root=args.data_root,
        dataset_name=args.dataset,
        config_path=args.config_path,
        device=args.device,
        iters=args.iters,
        batch_size=args.batch_size,
        logger=logger,
        save_raw_disp=args.save_raw_disp,
        save_colorbar=args.colorbar,
    )

    # Load model and dataset
    visualizer.load_model()
    visualizer.load_dataset(hamlyn_first_only=args.hamlyn_first_only)

    # Generate visualizations
    saved_paths = visualizer.visualize(
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )

    logger.info("\nTotal images saved: %d", len(saved_paths))


if __name__ == '__main__':
    main()

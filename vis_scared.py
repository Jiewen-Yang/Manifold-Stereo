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


def render_depth(disp, colormap=cv2.COLORMAP_INFERNO, percentile_clip=95):
    """
    Convert disparity map to pseudo-colored image for visualization.

    Args:
        disp: Disparity map (H, W) - numpy array or torch tensor
        colormap: OpenCV colormap type (default: INFERNO)
        percentile_clip: Clip percentile for normalization (default: 95).
                        Helps avoid dark images when valid disparity range is narrow.

    Returns:
        Pseudo-colored RGB image (H, W, 3) as uint8
    """
    if isinstance(disp, torch.Tensor):
        disp_np = disp.squeeze().cpu().numpy()
    else:
        disp_np = disp.copy()

    if disp_np.ndim > 2:
        disp_np = disp_np.squeeze()

    # Clip to percentile to avoid dark images from narrow valid range
    valid_mask = disp_np > 0
    if valid_mask.sum() > 0:
        vmin, vmax = np.percentile(disp_np[valid_mask], [2, percentile_clip])
    else:
        vmin, vmax = disp_np.min(), disp_np.max()

    # Normalize to [0, 255]
    if vmax - vmin > 1e-6:
        disp_norm = np.clip((disp_np - vmin) / (vmax - vmin) * 255.0, 0, 255)
    else:
        disp_norm = np.zeros_like(disp_np)

    disp_uint8 = disp_norm.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_uint8, colormap)
    return disp_color


class StereoVisualizer:
    """
    Visualizer for stereo matching predictions on SCARED, Hamlyn, and SERV-CT datasets.
    """

    def __init__(self, model_path, data_root, dataset_name, config_path=None,
                 split_file=None, device='cuda', iters=32, batch_size=1, logger=None):
        """
        Initialize the visualizer.

        Args:
            model_path: Path to model checkpoint
            data_root: Root directory of dataset
            dataset_name: 'scared', 'hamlyn', or 'servct'
            config_path: Path to config file
            split_file: Path to split file (for SCARED dataset)
            device: Device to use ('cuda' or 'cpu')
            iters: Number of iterations for inference
            batch_size: Batch size for DataLoader
            logger: logging.Logger instance
        """
        self.model_path = model_path
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.split_file = split_file
        self.config_path = config_path
        self.device = device
        self.iters = iters
        self.batch_size = batch_size
        self.logger = logger

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
            if self.split_file is None:
                self.split_file = osp.join(self.data_root, 'splits/endovis/test_files.txt')
            self.dataset = datasets.SCARED(aug_params, root=self.data_root, split_file=self.split_file)
            self.dataset.is_test = True
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

        # Create sequence subdirectory for Hamlyn
        if self.dataset_name == 'hamlyn':
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
            filename = f"{img_name['sequence']}_{img_name['keyframe']}_{img_name['frame_id']}"
        elif self.dataset_name == 'hamlyn':
            filename = f"{img_name['sequence']}_{img_name['frame_id']}"
        else:  # servct
            filename = f"{img_name['sequence']}_{img_name['frame_id']}"

        # Save GT disparity
        gt_color = render_depth(gt_np)
        gt_path = osp.join(seq_dir, f'{filename}_gt.png')
        cv2.imwrite(gt_path, gt_color)
        saved_paths.append(gt_path)

        # Save predicted disparity
        pred_color = render_depth(pred_np)
        pred_path = osp.join(seq_dir, f'{filename}_pred.png')
        cv2.imwrite(pred_path, pred_color)
        saved_paths.append(pred_path)

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
        
        return {'sequence': sequence, 'keyframe': keyframe, 'frame_id': frame_id}

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
    parser.add_argument('--data_root', type=str, default='/media/Storage3/ug_zy/hamlyn',
                        help='Root directory of dataset')
    parser.add_argument('--dataset', type=str, required=True, choices=['scared', 'hamlyn', 'servct'],
                        help='Dataset name to visualize')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--split_file', type=str, default=None,
                        help='Path to split file (for SCARED dataset)')
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
    logger.info("=" * 60)

    # Create visualizer
    visualizer = StereoVisualizer(
        model_path=args.model_path,
        data_root=args.data_root,
        dataset_name=args.dataset,
        config_path=args.config_path,
        split_file=args.split_file,
        device=args.device,
        iters=args.iters,
        batch_size=args.batch_size,
        logger=logger
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

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import copy
import random
from glob import glob
import os.path as osp

import sys
sys.path.append(os.getcwd())

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []
        self.calib_list = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])

        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)

        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()


    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.image_list)


class SERVCT(StereoDataset):
    def __init__(self, aug_params=None, root=r'/media/Storage3/ug_zy/SERV-CT', image_set='train'):
        super(SERVCT, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispSERVCT)
        assert os.path.exists(root), f"SERV-CT root path {root} does not exist!"

        self.root = root
        self.is_test = (image_set == 'test')

        image1_list = sorted(glob(os.path.join(root, 'Left_rectified', '*.png')))
        image2_list = [f.replace('Left_rectified', 'Right_rectified') for f in image1_list]
        disp_list   = [f.replace('Left_rectified', 'Disparity') for f in image1_list]

        assert len(image1_list) == len(image2_list) == len(disp_list), \
            f"Mismatch: {len(image1_list)} left, {len(image2_list)} right, {len(disp_list)} disp"

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
            stem = os.path.splitext(os.path.basename(img1))[0]
            calib_path = os.path.join(root, 'Rectified_calibration', f'{stem}.json')
            self.calib_list += [ calib_path ]

        missing_calib = [p for p in self.calib_list if not os.path.exists(p)]
        if missing_calib:
            preview = missing_calib[:5]
            raise FileNotFoundError(
                f"SERV-CT: {len(missing_calib)} Rectified_calibration/*.json missing "
                f"(expect same basename as left image). Examples: {preview}"
            )

    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            disp, valid = self.disparity_reader(self.disparity_list[index])
            disp = torch.from_numpy(np.array(disp).astype(np.float32)).unsqueeze(0)
            valid = torch.from_numpy(valid.astype(np.float32))
            path_list = self.image_list[index] + [self.disparity_list[index]]
            calib_path = self.calib_list[index]
            calib = frame_utils.readCalibSERVCT(calib_path)
            return path_list, img1, img2, disp, valid, calib

        return super().__getitem__(index)


class Hamlyn(StereoDataset):
    """
    Hamlyn endoscopic stereo dataset, supporting EndoDAC-style cropping for
    fair comparison: sequences > 13 have left+right concatenated horizontally,
    so we crop out the valid region (180:590) to get the proper single-image
    width (410 px) before resize.

    Crop box: (left, top, right, bottom) = (180, 0, 590, 288)
    """
    CROP_BOX = (180, 0, 590, 288)

    def __init__(self, aug_params=None, root=r'f:\stereo_dataset\hamlyn', image_set='train'):
        super(Hamlyn, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispHamlyn)
        assert os.path.exists(root), f"Hamlyn root path {root} does not exist!"

        self.root = root
        self.is_test = (image_set == 'test')

        calibration_dir = os.path.join(root, 'calibration')
        if not os.path.exists(calibration_dir):
            raise FileNotFoundError(f"Calibration directory not found: {calibration_dir}")

        sequence_ids = []
        for seq_path in sorted(os.listdir(calibration_dir)):
            seq_full = os.path.join(calibration_dir, seq_path)
            if os.path.isdir(seq_full) and seq_path.isdigit() and len(seq_path) == 2:
                sequence_ids.append(seq_path)

        for seq_id in sequence_ids:
            rect_dir = os.path.join(root, f'rectified{seq_id}')
            if not os.path.exists(rect_dir):
                continue

            image_exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
            image1_list_seq = []
            for ext in image_exts:
                image1_list_seq.extend(sorted(glob(os.path.join(rect_dir, 'image01', ext))))
            image1_list_seq = sorted(image1_list_seq)
            image2_list_seq = [f.replace('image01', 'image02') for f in image1_list_seq]
            disp_list_seq = []
            for f in image1_list_seq:
                disp_path = f.replace('image01', 'disparity01')
                base, _ = os.path.splitext(disp_path)
                disp_list_seq.append(base + '.png')

            for img1, img2, disp in zip(image1_list_seq, image2_list_seq, disp_list_seq):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
                calib_path = os.path.join(calibration_dir, seq_id)
                self.calib_list += [calib_path]

        if len(self.image_list) == 0:
            raise RuntimeError(f"No Hamlyn samples found in {root}")

        print(f"Hamlyn dataset loaded: {len(self.image_list)} samples from {len(set(self.calib_list))} sequences")

    def __getitem__(self, index):
        if self.is_test:
            try:
                img1 = frame_utils.read_gen(self.image_list[index][0])
                img2 = frame_utils.read_gen(self.image_list[index][1])
                if img1 is None or img2 is None:
                    raise RuntimeError(f"Failed to read image: {self.image_list[index]}")
                img1 = np.array(img1).astype(np.uint8)[..., :3]
                img2 = np.array(img2).astype(np.uint8)[..., :3]
                disp, valid = self.disparity_reader(self.disparity_list[index])
                if disp is None:
                    raise RuntimeError(f"Failed to read disparity: {self.disparity_list[index]}")
                disp = np.array(disp).astype(np.float32)

                img1_path = self.image_list[index][0]
                seq_id = None
                parts = img1_path.split(os.sep)
                for i, p in enumerate(parts):
                    if p.startswith('rectified'):
                        try:
                            seq_id = int(parts[i + 1])
                        except (IndexError, ValueError):
                            pass
                        break

                if seq_id is not None and seq_id > 13:
                    img1 = img1[:, 180:590]
                    img2 = img2[:, 180:590]
                    disp = disp[:, 180:590]
                    valid = valid[:, 180:590]

                img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
                img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
                disp = torch.from_numpy(disp).unsqueeze(0)
                valid = torch.from_numpy(valid.astype(np.float32))
                path_list = self.image_list[index] + [self.disparity_list[index]]
                calib_path = self.calib_list[index]
                calib = self._read_hamlyn_calib(calib_path)
                return path_list, img1, img2, disp, valid, calib
            except Exception as e:
                print(f"Error loading sample {index}: {self.image_list[index]}")
                print(f"Exception: {e}")
                raise

        return super().__getitem__(index)

    def _read_hamlyn_calib(self, calib_dir):
        """
        Read Hamlyn calibration files and extract fx and baseline for disparity->depth conversion.
        Formula: depth = fx * baseline / disparity
        """
        intrinsics_path = os.path.join(calib_dir, 'intrinsics.txt')
        extrinsics_path = os.path.join(calib_dir, 'extrinsics.txt')

        if not os.path.exists(intrinsics_path) or not os.path.exists(extrinsics_path):
            return None

        with open(intrinsics_path, 'r') as f:
            rows = []
            for line in f:
                line = line.strip()
                if line:
                    rows.append([float(x) for x in line.split()])
            k = np.array(rows, dtype=np.float64)

        with open(extrinsics_path, 'r') as f:
            rows = []
            for line in f:
                line = line.strip()
                if line:
                    rows.append([float(x) for x in line.split()])
            rt = np.array(rows, dtype=np.float64)

        if k.shape != (3, 4) or rt.shape != (3, 4):
            return None

        fx = k[0, 0]
        baseline_mm = np.linalg.norm(rt[:, 3])

        return {'fx': fx, 'baseline': baseline_mm}


class SCARED(StereoDataset):
    """SCARED loaded from the directory-level train/test split.

    The local SCARED copy is organized as follows::

        train_data/dataset_{1,2,3,6,7}/...
        test_data/dataset_{8,9}/...

    We intentionally scan these directories instead of using the original
    frame-list files.  A sample is added only when its left image, right image,
    and disparity map are all present.
    """

    TRAIN_DATASETS = (1, 2, 3, 6, 7)
    TEST_DATASETS = (8, 9)

    def __init__(self, aug_params=None, root='/path/to/scared', image_set='train'):
        super(SCARED, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispSCARED)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"SCARED root path {root} does not exist or is not a directory")
        if image_set not in ('train', 'test'):
            raise ValueError(f"SCARED image_set must be 'train' or 'test', got {image_set!r}")

        self.root = root
        self.image_set = image_set
        data_split = 'train_data' if image_set == 'train' else 'test_data'
        dataset_ids = self.TRAIN_DATASETS if image_set == 'train' else self.TEST_DATASETS
        split_root = os.path.join(root, data_split)

        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"SCARED {image_set} directory not found: {split_root}")

        missing = []
        for dataset_id in dataset_ids:
            dataset_root = os.path.join(split_root, f'dataset_{dataset_id}')
            if not os.path.isdir(dataset_root):
                logging.warning("SCARED %s dataset directory is missing: %s", image_set, dataset_root)
                continue

            keyframe_dirs = sorted(
                glob(os.path.join(dataset_root, 'keyframe_*')),
                key=lambda p: self._numeric_suffix(p, 'keyframe_'),
            )
            for keyframe_root in keyframe_dirs:
                left_root = os.path.join(keyframe_root, 'data', 'left_rectified')
                right_root = os.path.join(keyframe_root, 'data', 'right_rectified')
                disp_root = os.path.join(keyframe_root, 'data', 'disparity')

                left_paths = sorted(
                    glob(os.path.join(left_root, '*.png')),
                    key=self._numeric_stem,
                )
                for left_img in left_paths:
                    filename = os.path.basename(left_img)
                    right_img = os.path.join(right_root, filename)
                    disp_path = os.path.join(disp_root, filename)
                    if not (os.path.isfile(right_img) and os.path.isfile(disp_path)):
                        missing.append((left_img, right_img, disp_path))
                        continue
                    self.image_list.append([left_img, right_img])
                    self.disparity_list.append(disp_path)

        if not self.image_list:
            raise RuntimeError(
                f"No complete SCARED samples found in {split_root} for datasets {dataset_ids}"
            )

        logging.info(
            "Added %d SCARED %s samples from datasets %s%s",
            len(self.image_list), image_set, dataset_ids,
            f"; skipped {len(missing)} incomplete samples" if missing else '',
        )

    @staticmethod
    def _numeric_stem(path):
        stem = osp.splitext(osp.basename(path))[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    @staticmethod
    def _numeric_suffix(path, prefix):
        name = os.path.basename(path)
        suffix = name[len(prefix):] if name.startswith(prefix) else name
        return (0, int(suffix)) if suffix.isdigit() else (1, suffix)


def fetch_dataloader(args):
    """Create the data loader for the corresponding training set."""
    aug_params = {
        'crop_size': list(args.image_size),
        'min_scale': args.spatial_scale[0],
        'max_scale': args.spatial_scale[1],
        'do_flip': False,
        'yjitter': not args.noyjitter
    }
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = list(args.saturation_range)
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    print('train_datasets', args.train_datasets)
    for dataset_name in args.train_datasets:
        if dataset_name == 'scared':
            new_dataset = SCARED(aug_params, root=args.scared_root, image_set='train')
            logging.info(f"Adding {len(new_dataset)} samples from SCARED Train")
        elif dataset_name == 'scared_val':
            new_dataset = SCARED(aug_params, root=args.scared_root, image_set='test')
            logging.info(f"Adding {len(new_dataset)} samples from SCARED Val")
        elif dataset_name == 'scared_test':
            new_dataset = SCARED(aug_params, root=args.scared_root, image_set='test')
            logging.info(f"Adding {len(new_dataset)} samples from SCARED Test")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    return train_dataset


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import cv2

    def gray_2_colormap_np(img, cmap='rainbow', max=None):
        img = img.cpu().detach().numpy().squeeze()
        assert img.ndim == 2
        img[img < 0] = 0
        mask_invalid = img < 1e-10
        if max == None:
            img = img / (img.max() + 1e-8)
        else:
            img = img / (max + 1e-8)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
        cmap_m = matplotlib.colormaps.get_cmap(cmap)
        map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
        colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
        colormap[mask_invalid] = 0
        return colormap

    def viz_disp(disp, scale=1, COLORMAP=cv2.COLORMAP_JET):
        disp_np = (torch.abs(disp[0].squeeze())).data.cpu().numpy()
        disp_np = (disp_np * scale).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_np, COLORMAP)
        return disp_color

    plot_dir = './temp/plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Quick test: try loading SCARED dataset
    try:
        from core.utils.frame_utils import readDispSCARED
        dataset = SCARED(root='/path/to/scared', image_set='train')
        for i in range(min(5, len(dataset))):
            path_list, img1, img2, disp_gt, valid = dataset[i][0], *dataset[i][1:]
            print(f"Sample {i}: {path_list[0]}")
    except Exception as e:
        print(f"SCARED test skipped: {e}")

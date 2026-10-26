import os
import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
from core.utils.utils import InputPadder
from core.endo_stereo_fm import EndoStereoFM
from omegaconf import OmegaConf
import torch.nn.functional as F
from accelerate import Accelerator
import core.stereo_datasets as datasets
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import wandb
import logging
from pathlib import Path

def gray_2_colormap_np(img, cmap = 'rainbow', max = None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img/(max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.colormaps.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:,:,:3]*255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap


# ---- Geodesic-aware / Flow-Matching loss switches and weights (for ablation) ----
USE_GEO_LOSS = True   # geodesic-aware loss, main switch
W_GEO_DISP = 1      # geodesic disparity supervision
W_DISP_GEO_CONS = 1 # final disp and geodesic disp consistency
W_UNCERT_CALIB = 0.5  # uncertainty calibration
W_UNCERT_LAPLACE = 0.5# Laplace heteroscedastic NLL
W_MANIFOLD = 0.1      # geometric consistency (manifold regularization)

USE_FM_LOSS = True    # Flow Matching / ODE related loss switch
W_FM_LATENT = 1      # latent matching loss: ||z_L_pred - z_L||^2
W_FM_PATH = 0.5       # path smoothing regularization: sum||x_{t+1}-x_t||^2


def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid,
                  loss_gamma=0.9, max_disp=192, geo_info=None):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    # quantile = torch.quantile((disp_init_pred - disp_gt).abs(), 0.9)
    init_valid = valid.bool() & ~torch.isnan(disp_init_pred)#  & ((disp_init_pred - disp_gt).abs() < quantile)
    disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[init_valid], disp_gt[init_valid], reduction='mean')
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        # quantile = torch.quantile(i_loss, 0.9)
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()

    metrics = {
        'train/epe': epe.mean(),
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean(),
    }

    # ---- Geodesic-aware loss ----
    if USE_GEO_LOSS and (geo_info is not None):
        disp_geo = geo_info.get("disp_geo", None)          # [B,1,H,W] or [B,H,W]
        uncertainty = geo_info.get("uncertainty", None)    # [B,H,W]
        geo_consistency = geo_info.get("geo_consistency", None)  # [B,H,W]

        # unify dimensions: [B,H,W]
        if disp_geo is not None:
            if disp_geo.dim() == 4 and disp_geo.shape[1] == 1:
                disp_geo = disp_geo[:, 0]
        if uncertainty is not None and uncertainty.dim() == 4 and uncertainty.shape[1] == 1:
            uncertainty = uncertainty[:, 0]
        if geo_consistency is not None and geo_consistency.dim() == 4 and geo_consistency.shape[1] == 1:
            geo_consistency = geo_consistency[:, 0]

        # geo_valid needs to match disp_geo shape: [B, H, W]
        # valid is [B, 1, H, W], need to squeeze or adjust
        if valid.dim() == 4 and valid.shape[1] == 1:
            geo_valid = valid.squeeze(1).bool()  # [B, H, W]
        else:
            geo_valid = valid.bool()
        
        # disp_gt also needs adjustment to [B, H, W] to match disp_geo
        if disp_gt.dim() == 4 and disp_gt.shape[1] == 1:
            disp_gt_2d = disp_gt.squeeze(1)  # [B, H, W]
        else:
            disp_gt_2d = disp_gt
        
        disp_final = disp_preds[-1]
        # disp_final also needs adjustment to [B, H, W]
        if disp_final.dim() == 4 and disp_final.shape[1] == 1:
            disp_final_2d = disp_final.squeeze(1)  # [B, H, W]
        else:
            disp_final_2d = disp_final

        # 1) geodesic disparity supervision
        if disp_geo is not None and W_GEO_DISP > 0:
            disp_loss = disp_loss + W_GEO_DISP * F.smooth_l1_loss(
                disp_geo[geo_valid], disp_gt_2d[geo_valid], reduction='mean'
            )

        # 2) final disp and geodesic disp consistency (adaptive weighting by uncertainty/geo_consistency)
        if disp_geo is not None and W_DISP_GEO_CONS > 0:
            # pixel-level L1 consistency (using adjusted 2D shapes)
            prior_cons = F.smooth_l1_loss(
                disp_final_2d, disp_geo, reduction='none'
            )  # [B,H,W]

            # default weight = 1; if uncertainty / geo_consistency exists, construct weights based on them
            w = torch.ones_like(prior_cons)

            if uncertainty is not None:
                # higher uncertainty -> more reliance on mono prior
                u = uncertainty.clamp(min=1e-4)
                # simple normalization + truncation, prevent extreme values
                u_norm = (u / (u.mean() + 1e-6)).clamp(0.0, 3.0)
                w = u_norm / (u_norm.max() + 1e-6)

            if geo_consistency is not None:
                # can also use (1 - consistency) as auxiliary weight: more inconsistent -> more reliance on mono
                gc = geo_consistency.clamp(0.0, 1.0)
                w_gc = 1.0 - gc
                # fuse two types of info (if both exist): take average
                if uncertainty is not None:
                    w = 0.5 * w + 0.5 * w_gc
                else:
                    w = w_gc

            # only count in valid regions, weighted average
            w_valid = w[geo_valid]
            prior_valid = prior_cons[geo_valid]
            # avoid all-zero weights
            if w_valid.numel() > 0:
                prior_weighted = (w_valid * prior_valid).sum() / (w_valid.sum() + 1e-6)
                disp_loss = disp_loss + W_DISP_GEO_CONS * prior_weighted

        # 3) uncertainty calibration + Laplace NLL
        if uncertainty is not None:
            err = (disp_final_2d - disp_gt_2d).abs().detach()
            u = uncertainty.clamp(min=1e-4)
            if W_UNCERT_CALIB > 0:
                disp_loss = disp_loss + W_UNCERT_CALIB * F.l1_loss(
                    u[geo_valid], err[geo_valid], reduction='mean'
                )
            if W_UNCERT_LAPLACE > 0:
                disp_loss = disp_loss + W_UNCERT_LAPLACE * (
                    (err[geo_valid] / u[geo_valid] + torch.log(2 * u[geo_valid])).mean()
                )

        # 4) Manifold regularization (geometric consistency should be larger)
        if geo_consistency is not None and W_MANIFOLD > 0:
            disp_loss = disp_loss + W_MANIFOLD * (-geo_consistency[geo_valid].mean())

    # ---- Flow Matching / ODE latent loss ----
    if USE_FM_LOSS and (geo_info is not None):
        fm_z_L = geo_info.get("fm_z_L", None)           # [B,D] or [B,L,H,W], detached target latent
        fm_z_L_pred = geo_info.get("fm_z_L_pred", None) # [B,D] or [B,L,H,W], predicted latent
        fm_traj = geo_info.get("fm_traj", None)         # [B,T,D] or [B,T,L,H,W], discrete path

        if (fm_z_L is not None) and (fm_z_L_pred is not None) and W_FM_LATENT > 0:
            # compatible with global latent and spatial latent field:
            # flatten to [B, -1] then compute L2
            if fm_z_L.dim() > 2:
                B = fm_z_L.shape[0]
                fm_z_L_flat = fm_z_L.view(B, -1)
                fm_z_L_pred_flat = fm_z_L_pred.view(B, -1)
            else:
                fm_z_L_flat = fm_z_L
                fm_z_L_pred_flat = fm_z_L_pred

            # simplified latent refinement loss
            fm_latent_loss = ((fm_z_L_pred_flat - fm_z_L_flat) ** 2).mean()
            disp_loss = disp_loss + W_FM_LATENT * fm_latent_loss

        if (fm_traj is not None) and W_FM_PATH > 0:
            # path smoothing regularization
            # fm_traj: [B, T, ...]
            diffs = fm_traj[:, 1:] - fm_traj[:, :-1]  # [B, T-1, ...]
            fm_path_loss = (diffs ** 2).mean()
            disp_loss = disp_loss + W_FM_PATH * fm_path_loss

    return disp_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    DPT_params = list(map(id, model.feat_decoder.parameters())) 
    rest_params = filter(lambda x:id(x) not in DPT_params and x.requires_grad, model.parameters())

    params_dict = [{'params': model.feat_decoder.parameters(), 'lr': args.lr/2.0}, 
                   {'params': rest_params, 'lr': args.lr}, ]
    optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, [args.lr/2.0, args.lr], args.total_step+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')


    return optimizer, scheduler

@hydra.main(version_base=None, config_path='config', config_name='train_scared')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='fp16', dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), log_with='wandb', kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
    accelerator.init_trackers(project_name=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True), init_kwargs={'wandb': cfg.wandb})

    # Use fetch_dataloader to load SCARED dataset
    train_dataset = datasets.fetch_dataloader(cfg)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size//cfg.num_gpu,
        pin_memory=True, shuffle=True, num_workers=int(4), drop_last=True)

    # Load SCARED test set for validation
    val_scared = datasets.SCARED(
        aug_params={}, 
        root=cfg.scared_root, 
        split_file=os.path.join(cfg.scared_splits_dir, 'test_files.txt')
    )
    val_scared_loader = torch.utils.data.DataLoader(val_scared, batch_size=int(1),
        pin_memory=True, shuffle=False, num_workers=int(4), drop_last=False)
    model = EndoStereoFM(cfg)
    if cfg.restore_ckpt is not None and cfg.restore_ckpt != "":
        assert cfg.restore_ckpt.endswith(".pth")
        print(f"Loading checkpoint from {cfg.restore_ckpt}")
        assert os.path.exists(cfg.restore_ckpt)
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            ckpt[key.replace('module.', '')] = checkpoint[key]
        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")
        del ckpt, checkpoint
    optimizer, lr_scheduler = fetch_optimizer(cfg, model)
    result = accelerator.prepare(
        train_loader, val_scared_loader,
        model, optimizer, lr_scheduler
    )
    train_loader, val_scared_loader, model, optimizer, lr_scheduler = result
    val_loaders = {
        'scared': val_scared_loader,
    }
    model.to(accelerator.device)

    val_loggers = {}
    for ds_name in val_loaders:
        logger_ds = logging.getLogger(f"val_{ds_name}")
        logger_ds.setLevel(logging.INFO)
        logger_ds.handlers.clear()
        log_path = Path(cfg.save_path) / f"{ds_name}_val.log"
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s'))
        logger_ds.addHandler(fh)
        val_loggers[ds_name] = logger_ds

    total_step = 0
    should_keep_training = True
    while should_keep_training:
        active_train_loader = train_loader
        model.train()
        if hasattr(model, 'module'):
            model.module.freeze_bn()
        else:
            model.freeze_bn()
        for data in tqdm(active_train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
            _, left, right, disp_gt, valid = [x for x in data]
            with accelerator.autocast():
                out = model(left, right, iters=cfg.train_iters)
            # output results with geodesic info
            if isinstance(out, (list, tuple)) and len(out) == 4:
                disp_init_pred, disp_preds, depth_mono, geo_info = out
            else:
                disp_init_pred, disp_preds, depth_mono = out
                geo_info = None
            loss, metrics = sequence_loss(
                disp_preds, disp_init_pred, disp_gt, valid,
                max_disp=cfg.max_disp, geo_info=geo_info
            )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_step += 1
            loss = accelerator.reduce(loss.detach(), reduction='mean')
            metrics = accelerator.reduce(metrics, reduction='mean')
            accelerator.log({'train/loss': loss, 'train/learning_rate': optimizer.param_groups[0]['lr']}, total_step)
            accelerator.log(metrics, total_step)

            ####visualize the depth_mono and disp_preds
            # wrap wandb related code with try-except to prevent crash if library not loaded
            if total_step % 200 == 0 and accelerator.is_main_process:
                try:
                    import wandb
                    image1_np = left[0].squeeze().cpu().numpy()
                    image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
                    image1_np = image1_np.astype(np.uint8)
                    image1_np = np.transpose(image1_np, (1, 2, 0))

                    image2_np = right[0].squeeze().cpu().numpy()
                    image2_np = (image2_np - image2_np.min()) / (image2_np.max() - image2_np.min()) * 255.0
                    image2_np = image2_np.astype(np.uint8)
                    image2_np = np.transpose(image2_np, (1, 2, 0))

                    depth_mono_np = gray_2_colormap_np(depth_mono[0].squeeze())
                    disp_preds_np = gray_2_colormap_np(disp_preds[-1][0].squeeze())
                    disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze())
                    
                    accelerator.log({"disp_pred": wandb.Image(disp_preds_np, caption="step:{}".format(total_step))}, total_step)
                    accelerator.log({"disp_gt": wandb.Image(disp_gt_np, caption="step:{}".format(total_step))}, total_step)
                    accelerator.log({"depth_mono": wandb.Image(depth_mono_np, caption="step:{}".format(total_step))}, total_step)
                except Exception as e:
                    print(f"Failed to log images to wandb: {e}")

            if (total_step > 0) and (total_step % cfg.save_frequency == 0):
                if accelerator.is_main_process:
                    save_path = Path(cfg.save_path + '/%d.pth' % (total_step))
                    model_save = accelerator.unwrap_model(model)
                    torch.save(model_save.state_dict(), save_path)
                    del model_save
        
            if (total_step > 0) and (total_step % cfg.val_frequency == 0):
                model.eval()
                for ds_name, val_loader in val_loaders.items():
                    total_px_count, total_epe = 0, 0.0
                    total_bad1, total_bad2, total_bad3, total_d1 = 0.0, 0.0, 0.0, 0.0
                    for data in tqdm(val_loader, dynamic_ncols=True, desc=f"val_{ds_name}", disable=not accelerator.is_main_process):
                        extra = [x for x in data]
                        left = extra[1]
                        right = extra[2]
                        disp_gt = extra[3]
                        valid = extra[4]

                        padder = InputPadder(left.shape, divis_by=32)
                        left_pad, right_pad = padder.pad(left, right)
                        with torch.no_grad():
                            disp_pred = model(left_pad, right_pad, iters=cfg.valid_iters, test_mode=True)
                        disp_pred = padder.unpad(disp_pred)
                        assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
                        epe = torch.abs(disp_pred - disp_gt)
                        epe = torch.squeeze(epe, dim=1)
                        valid_mask = (valid >= 0.5)
                        epe_valid = epe[valid_mask]
                        px_count = epe_valid.numel()
                        total_epe += epe_valid.sum().item()
                        total_bad1 += (epe_valid > 1.0).sum().item()
                        total_bad2 += (epe_valid > 2.0).sum().item()
                        total_bad3 += (epe_valid > 3.0).sum().item()
                        disp_gt_squeezed = torch.squeeze(disp_gt, dim=1)
                        d1_mask = (epe > 3.0) & (epe > 0.05 * disp_gt_squeezed)
                        total_d1 += d1_mask[valid_mask].sum().item()
                        total_px_count += px_count

                    gathered_stats = accelerator.gather_for_metrics(
                        torch.tensor([total_px_count, total_epe, total_bad1, total_bad2, total_bad3, total_d1],
                                     device=accelerator.device, dtype=torch.float64)
                    )
                    # gather_for_metrics may return 1D (single process) or 2D (multi-process)
                    if gathered_stats.dim() == 1:
                        gathered_stats = gathered_stats.unsqueeze(0)
                    tot_px = gathered_stats[:, 0].sum().item()
                    tot_epe = gathered_stats[:, 1].sum().item()
                    tot_b1 = gathered_stats[:, 2].sum().item()
                    tot_b2 = gathered_stats[:, 3].sum().item()
                    tot_b3 = gathered_stats[:, 4].sum().item()
                    tot_d1 = gathered_stats[:, 5].sum().item()

                    mean_mae = tot_epe / tot_px if tot_px > 0 else 0
                    mean_bad1 = 100.0 * tot_b1 / tot_px if tot_px > 0 else 0
                    mean_bad2 = 100.0 * tot_b2 / tot_px if tot_px > 0 else 0
                    mean_bad3 = 100.0 * tot_b3 / tot_px if tot_px > 0 else 0
                    mean_d1 = 100.0 * tot_d1 / tot_px if tot_px > 0 else 0

                    if accelerator.is_main_process:
                        if ds_name == 'scared':
                            val_loggers[ds_name].info(f"step={total_step}  mae={mean_mae:.4f}  d1={mean_d1:.2f}%")
                            print(f"[val_{ds_name}]  step={total_step}  mae={mean_mae:.4f}  d1={mean_d1:.2f}%")

                model.train()
                if hasattr(model, 'module'):
                    model.module.freeze_bn()
                else:
                    model.freeze_bn()

            if total_step == cfg.total_step:
                should_keep_training = False
                break

    if accelerator.is_main_process:
        save_path = Path(cfg.save_path + '/final.pth')
        model_save = accelerator.unwrap_model(model)
        torch.save(model_save.state_dict(), save_path)
        del model_save

    accelerator.end_training()


if __name__ == '__main__':
    main()

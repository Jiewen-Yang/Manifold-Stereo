"""
EndoStereoFM 模型推理脚本
功能：
1. 加载 test_image 中的左右目图
2. 运行模型推理
3. 保存中间结果：
   - 初始视差图 (init_disp)
   - z_field
   - Z_T (z_field_pred)
   - ΔDmono (disp_offset_4x)
   - Dmono (disp_mono_4x)
   - 最终视差图 (final_disp)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# 添加项目路径
sys.path.append('./Depth-Anything-V2-list3')

from core.endo_stereo_fm import EndoStereoFM, compute_scale_shift


def load_config(config_path):
    """加载配置文件"""
    cfg = OmegaConf.load(config_path)
    return cfg


def load_images(left_path, right_path):
    """加载左右目图像"""
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    if left_img is None or right_img is None:
        raise ValueError(f"无法读取图像: {left_path} 或 {right_path}")

    # BGR 转 RGB
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    return left_img, right_img


def preprocess_images(left_img, right_img):
    """预处理图像"""
    # 转为 float32 并归一化到 [0, 255]
    left_tensor = torch.from_numpy(left_img).float()
    right_tensor = torch.from_numpy(right_img).float()

    # 调整维度: HWC -> BCHW
    left_tensor = left_tensor.permute(2, 0, 1).unsqueeze(0)
    right_tensor = right_tensor.permute(2, 0, 1).unsqueeze(0)

    return left_tensor, right_tensor


def visualize_and_save_disp(disp, save_path, title="Disparity", vmin=None, vmax=None):
    """可视化并保存视差图"""
    # disp shape: [1, 1, H, W] or [H, W]
    if disp.dim() == 4:
        disp = disp.squeeze().cpu().detach().numpy()
    elif disp.dim() == 3:
        disp = disp.squeeze().cpu().detach().numpy()

    # 确保是 2D
    if disp.ndim == 3:
        disp = disp[0] if disp.shape[0] == 1 else disp[0]

    # 归一化用于显示
    disp_vis = disp.copy()
    # 使用 2%~100% 分位数归一化（和 render_depth 保持一致，避免极端值影响显示）
    if vmin is None:
        vmin = np.percentile(disp_vis, 2)
    if vmax is None:
        vmax = np.percentile(disp_vis, 100)

    disp_normalized = (disp_vis - vmin) / (vmax - vmin + 1e-8)

    plt.figure(figsize=(12, 8))
    plt.imshow(disp_normalized, cmap='magma')
    plt.colorbar(label=f'Disparity (range: {vmin:.2f} - {vmax:.2f})')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")

    return disp


def visualize_and_save_latent(z_field, save_path, title="Latent Field"):
    """可视化并保存 latent field (取通道均值)"""
    # z_field shape: [B, L, H, W]
    z = z_field.squeeze(0).cpu().detach().numpy()  # [L, H, W]

    # 取所有通道的均值
    z_mean = z.mean(axis=0)

    # 也计算通道标准差作为辅助信息
    z_std = z.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    im1 = axes[0].imshow(z_mean, cmap='viridis')
    axes[0].set_title(f'{title} - Mean (L={z.shape[0]})')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(z_std, cmap='plasma')
    axes[1].set_title(f'{title} - Std')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")

    return z_mean, z_std


def visualize_and_save_flow_matching_traj(traj, save_path, title="Flow Matching Trajectory"):
    """可视化 Flow Matching 轨迹"""
    # traj shape: [B, T, L, H, W] 或 List[[B, L, H, W]]
    if isinstance(traj, torch.Tensor):
        # [B, T, L, H, W]
        traj_np = traj[0].cpu().detach().numpy()  # [T, L, H, W]
    else:
        # List
        traj_np = np.stack([t[0].cpu().detach().numpy() for t in traj], axis=0)

    T = traj_np.shape[0]
    n_show = min(T, 6)  # 最多显示6个时间步

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(n_show):
        t_idx = int(i * (T - 1) / (n_show - 1)) if n_show > 1 else 0
        z_t = traj_np[t_idx].mean(axis=0)  # [H, W]

        im = axes[i].imshow(z_t, cmap='coolwarm')
        axes[i].set_title(f't = {t_idx / (T - 1):.2f}' if T > 1 else 't = 0')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])

    # 隐藏多余的子图
    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")


def main():
    # 配置路径
    config_path = 'config/train_scared.yaml'
    left_image_path = 'test_image/left.png'
    right_image_path = 'test_image/right.png'
    output_dir = 'output/inference_debug'
    checkpoint_path = "/media/Storage3/ug_zy/random_crop/Stereo_Match/checkpoints/scared_320256/14000.pth"  # 设置为你的模型权重路径

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载配置
    print("加载配置...")
    cfg = load_config(config_path)

    # 加载图像
    print("加载图像...")
    left_img, right_img = load_images(left_image_path, right_image_path)
    print(f"图像尺寸: 左图 {left_img.shape}, 右图 {right_img.shape}")

    # 预处理
    left_tensor, right_tensor = preprocess_images(left_img, right_img)

    # 移动到 GPU
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    left_tensor = left_tensor.to(device)
    right_tensor = right_tensor.to(device)

    # 创建模型
    print("创建模型...")
    model = EndoStereoFM(cfg)

    # 加载权重（如果有）
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # 兼容不同的 checkpoint 格式
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # 直接就是 state_dict
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        print("警告: 未指定权重文件或文件不存在，使用随机初始化权重")

    model = model.to(device)
    model.eval()

    # ==================== 修改 forward 函数以获取中间变量 ====================
    # 为了获取中间变量，我们需要 monkey-patch 或者重写部分前向过程
    # 这里我们重新实现关键部分

    print("\n开始推理...")

    # 图像预处理
    image1 = (2 * (left_tensor / 255.0) - 1.0).contiguous()
    image2 = (2 * (right_tensor / 255.0) - 1.0).contiguous()

    # 单目深度估计
    depth_mono, features_mono_left, features_mono_right = model.infer_mono(image1, image2)

    # 单目视差 (Dmono 初值)
    scale_factor = 0.25
    size = (int(depth_mono.shape[-2] * scale_factor), int(depth_mono.shape[-1] * scale_factor))
    disp_mono_4x = F.interpolate(depth_mono, size=size, mode='bilinear', align_corners=False)

    # 特征提取
    features_left = model.feat_transfer(features_mono_left)
    features_right = model.feat_transfer(features_mono_right)

    stem_2x = model.stem_2(image1)
    stem_4x = model.stem_4(stem_2x)
    stem_8x = model.stem_8(stem_4x)
    stem_16x = model.stem_16(stem_8x)

    stem_x_list = [stem_16x, stem_8x, stem_4x]
    features_left[0] = torch.cat((features_left[0], stem_4x), 1)

    stem_2y = model.stem_2(image2)
    stem_4y = model.stem_4(stem_2y)
    features_right[0] = torch.cat((features_right[0], stem_4y), 1)

    match_left = model.desc(model.conv(features_left[0]))
    match_right = model.desc(model.conv(features_right[0]))

    # 构建 GWC Volume
    from core.stereo_submodule import build_gwc_volume
    gwc_volume = build_gwc_volume(match_left, match_right, cfg.max_disp // 4, 8)
    gwc_volume = model.corr_stem(gwc_volume)
    gwc_volume = model.corr_feature_att(gwc_volume, features_left[0])

    # 3D Hourglass 生成代价体 (geo_encoding_volume)
    geo_encoding_volume = model.cost_agg(gwc_volume, features_left)
    print(f"代价体 geo_encoding_volume 形状: {geo_encoding_volume.shape}")

    # Soft Argmax 得到初始视差图 (init_disp)
    prob = F.softmax(model.classifier(geo_encoding_volume).squeeze(1), dim=1)
    from core.stereo_submodule import disparity_regression
    init_disp = disparity_regression(prob, cfg.max_disp // 4)
    print(f"初始视差图 init_disp 形状: {init_disp.shape}")

    # ==================== Flow Matching 分支 ====================
    if model.use_flow_matching and model.use_spatial_latent:
        # GeoLatentField 生成 z_field
        z_field = model.geo_latent_field(geo_encoding_volume)
        print(f"z_field 形状: {z_field.shape}")

        # 条件场
        cond_field = model.flow_cond_conv(match_right)

        # Flow Matching 采样得到 Z_T (z_field_pred)
        z_field_pred, traj_field = model.flow_sampler_field.sample(z_field, cond_field)
        print(f"Z_T (z_field_pred) 形状: {z_field_pred.shape}")
        print(f"  z_field 数值范围: [{z_field.min():.6f}, {z_field.max():.6f}], 均值: {z_field.mean():.6f}")
        print(f"  z_field_pred 数值范围: [{z_field_pred.min():.6f}, {z_field_pred.max():.6f}], 均值: {z_field_pred.mean():.6f}")
        print(f"  z_field_pred - z_field 差异范围: [{(z_field_pred - z_field).min():.6f}, {(z_field_pred - z_field).max():.6f}], 均值: {(z_field_pred - z_field).mean():.6f}")

        # latent field 解码为视差偏置 ΔDmono
        disp_offset_4x = model.latent_field_to_disp(z_field_pred)
        print(f"ΔDmono (disp_offset_4x) 形状: {disp_offset_4x.shape}, 范围: [{disp_offset_4x.min():.4f}, {disp_offset_4x.max():.4f}], 均值: {disp_offset_4x.mean():.4f}")

        # Dmono = 原始单目视差 + ΔDmono
        disp_mono_4x_corrected = disp_mono_4x + disp_offset_4x
        print(f"Dmono (修正后) 形状: {disp_mono_4x_corrected.shape}")
    else:
        z_field = None
        z_field_pred = None
        disp_offset_4x = None
        disp_mono_4x_corrected = disp_mono_4x

    # ==================== 完整迭代 refinement (获取最终视差图) ====================
    print("\n执行完整迭代 refinement...")

    # 初始化 context network
    from core.stereo_submodule import context_upsample, disparity_regression
    from core.geo_encoder import Combined_Geo_Encoding_Volume
    from core.disp_warp import disp_warp

    cnet_list = model.feat_transfer_cnet(features_mono_left, stem_x_list)
    net_list = [torch.tanh(x[0]) for x in cnet_list]
    inp_list = [torch.relu(x[1]) for x in cnet_list]
    inp_list = [torch.relu(x) for x in inp_list]
    inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i, conv in zip(inp_list, model.context_zqr_convs)]
    net_list_mono = [x.clone() for x in net_list]

    # 构建几何编码
    geo_fn = Combined_Geo_Encoding_Volume(
        match_left.float(), match_right.float(), geo_encoding_volume.float(),
        radius=cfg.corr_radius, num_levels=cfg.corr_levels
    )

    # 坐标
    b, c, h, w = match_left.shape
    coords = torch.arange(w).float().to(match_left.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1).contiguous()

    # 迭代 refinement
    disp = init_disp
    iters = 12  # 默认迭代次数

    with torch.no_grad():
        for itr in range(iters):
            disp = disp.detach()
            if itr >= 1:
                disp_mono_4x = disp_mono_4x.detach()

            geo_feat = geo_fn(disp, coords)

            if itr > iters - 8:
                if itr == iters - 7:
                    # scale-shift 对齐
                    for i in range(b):
                        scale, shift = compute_scale_shift(
                            disp_mono_4x[i].clone().squeeze(1).to(torch.float32),
                            disp[i].clone().squeeze(1).to(torch.float32)
                        )
                        disp_mono_4x[i] = scale * disp_mono_4x[i] + shift

                warped_right_mono = disp_warp(features_right[0], disp_mono_4x.clone().to(features_right[0].dtype))[0]
                flaw_mono = warped_right_mono - features_left[0]

                warped_right_stereo = disp_warp(features_right[0], disp.clone().to(features_right[0].dtype))[0]
                flaw_stereo = warped_right_stereo - features_left[0]
                geo_feat_mono = geo_fn(disp_mono_4x, coords)

                # 几何一致性
                geo_dist_stereo = flaw_stereo.pow(2).sum(dim=1)
                geo_consistency_stereo = torch.exp(-geo_dist_stereo / (features_left[0].shape[1] * 0.1 + 1e-6))

                if model.use_geo_gating:
                    w_stereo = geo_consistency_stereo
                    w_mono = 1.0 - geo_consistency_stereo
                    flaw_stereo_g = flaw_stereo * w_stereo.unsqueeze(1)
                    flaw_mono_g = flaw_mono * w_mono.unsqueeze(1)
                    geo_feat_g = geo_feat * w_stereo.unsqueeze(1) + geo_feat_mono * w_mono.unsqueeze(1)
                else:
                    flaw_stereo_g = flaw_stereo
                    flaw_mono_g = flaw_mono
                    geo_feat_g = geo_feat

                # 双分支更新
                net_list, mask_feat_4, delta_disp = model.update_block_mix_stereo(
                    net_list, inp_list,
                    flaw_stereo_g, disp, geo_feat_g,
                    flaw_mono_g, disp_mono_4x, geo_feat_mono,
                    iter16=model.args.n_gru_layers == 3, iter08=model.args.n_gru_layers >= 2
                )
                net_list_mono, mask_feat_4_mono, delta_disp_mono = model.update_block_mix_mono(
                    net_list_mono, inp_list,
                    flaw_mono_g, disp_mono_4x, geo_feat_mono,
                    flaw_stereo_g, disp, geo_feat,
                    iter16=model.args.n_gru_layers == 3, iter08=model.args.n_gru_layers >= 2
                )
                disp_mono_4x = disp_mono_4x + delta_disp_mono
                disp_mono_4x_up = model.upsample_disp(disp_mono_4x, mask_feat_4_mono, stem_2x)
            else:
                # 标准单分支更新
                net_list, mask_feat_4, delta_disp = model.update_block(
                    net_list, inp_list, geo_feat, disp,
                    iter16=model.args.n_gru_layers == 3, iter08=model.args.n_gru_layers >= 2
                )

            disp = disp + delta_disp
            disp_up = model.upsample_disp(disp, mask_feat_4, stem_2x)

            # 最后一次迭代：使用 mono 分支的上采样结果作为最终视差（与训练 loss 一致）
            if itr == iters - 1:
                # stereo 分支的 REMP 暂不应用（训练时 loss 没有用它）
                disp_up_final = disp_mono_4x_up
            else:
                disp_up_final = disp_up

            print(f"  迭代 {itr + 1}/{iters} 完成")

    print(f"最终视差图 disp_up_final 形状: {disp_up_final.shape}")

    # ==================== 保存中间结果 ====================
    print("\n保存中间结果...")

    # 1. 保存初始视差图
    visualize_and_save_disp(
        init_disp,
        os.path.join(output_dir, '01_init_disp.png'),
        title='Initial Disparity (Soft Argmax)',
        vmin=0,
        vmax=cfg.max_disp
    )

    # 2. 保存 z_field
    if z_field is not None:
        visualize_and_save_latent(
            z_field,
            os.path.join(output_dir, '02_z_field.png'),
            title='z_field (Latent Field Encoder)'
        )

    # 3. 保存 Z_T (z_field_pred)
    if z_field_pred is not None:
        visualize_and_save_latent(
            z_field_pred,
            os.path.join(output_dir, '03_Z_T.png'),
            title='Z_T (z_field_pred - After Flow Matching)'
        )

    # 4. 保存 ΔDmono
    if disp_offset_4x is not None:
        # ΔDmono 可能值很小，需要放大显示
        offset_vis = disp_offset_4x.squeeze().cpu().detach().numpy()
        vmin, vmax = offset_vis.min(), offset_vis.max()
        vabs_max = max(abs(vmin), abs(vmax))

        plt.figure(figsize=(12, 8))
        plt.imshow(offset_vis, cmap='RdBu_r', vmin=-vabs_max, vmax=vabs_max)
        plt.colorbar(label='Disparity Offset')
        plt.title('ΔDmono (Disparity Offset from Flow Matching)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '04_delta_dmono.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存: {output_dir}/04_delta_dmono.png")

        # 也保存原始数据
        np.save(os.path.join(output_dir, '04_delta_dmono_raw.npy'), offset_vis)

    # 5. 保存 Dmono (修正后)
    visualize_and_save_disp(
        disp_mono_4x_corrected,
        os.path.join(output_dir, '05_dmono_corrected.png'),
        title='Dmono (Corrected Monocular Disparity)',
        vmin=0,
        vmax=cfg.max_disp
    )

    # 6. 保存原始单目视差 (修正前)
    visualize_and_save_disp(
        disp_mono_4x,
        os.path.join(output_dir, '05b_dmono_raw.png'),
        title='Dmono Raw (Before Correction)',
        vmin=0,
        vmax=cfg.max_disp
    )

    # 7. 保存 Flow Matching 轨迹 (如果有)
    if model.use_flow_matching and hasattr(model, 'flow_sampler_field'):
        # 重新采样获取轨迹用于可视化
        with torch.no_grad():
            _, traj_field_vis = model.flow_sampler_field.sample(z_field, cond_field)
        visualize_and_save_flow_matching_traj(
            traj_field_vis,
            os.path.join(output_dir, '06_flow_matching_trajectory.png'),
            title='Flow Matching Trajectory (z_field → Z_T)'
        )

    # 8. 保存原始图像用于对比
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(left_img)
    plt.title('Left Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(right_img)
    plt.title('Right Image')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '00_input_images.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_dir}/00_input_images.png")

    # 9. 保存上采样的初始视差图 (和原始尺寸对齐)
    from core.stereo_submodule import context_upsample
    xspx = model.spx_4(features_left[0])
    xspx = model.spx_2(xspx, stem_2x)
    spx_pred = model.spx(xspx)
    spx_pred = F.softmax(spx_pred, 1)
    init_disp_up = context_upsample(init_disp * 4.0, spx_pred.float()).unsqueeze(1)
    visualize_and_save_disp(
        init_disp_up,
        os.path.join(output_dir, '01b_init_disp_upsampled.png'),
        title='Initial Disparity (Upsampled to Full Resolution)',
        vmin=0,
        vmax=cfg.max_disp
    )

    # 10. 保存最终视差图 (使用 render_depth 函数)
    print("\n保存最终视差图...")
    final_disp_np = disp_up_final.squeeze().cpu().detach().numpy()
    # 转为 HxW 格式
    if final_disp_np.ndim == 3:
        final_disp_np = final_disp_np[0]

    # 使用 vis_scared.py 中的 render_depth 函数绘制（兼容有无 percentile_clip 两种版本）
    from vis_scared import render_depth
    import inspect
    sig = inspect.signature(render_depth)
    if 'percentile_clip' in sig.parameters:
        # 去掉 percentile_clip，保持和中间结果可视化一致（全数据范围归一化）
        final_disp_color = render_depth(final_disp_np, colormap=cv2.COLORMAP_INFERNO, percentile_clip=100)
    else:
        final_disp_color = render_depth(final_disp_np, colormap=cv2.COLORMAP_INFERNO)

    # 保存彩色视差图
    cv2.imwrite(os.path.join(output_dir, '07_final_disp_color.png'), final_disp_color)
    print(f"保存: {output_dir}/07_final_disp_color.png")

    # 同时保存原始数值格式
    np.save(os.path.join(output_dir, '07_final_disp_raw.npy'), final_disp_np)

    # 保存灰度视差图 (使用 matplotlib, 和之前风格一致)
    visualize_and_save_disp(
        disp_up_final,
        os.path.join(output_dir, '07_final_disp.png'),
        title='Final Disparity (After Refinement)',
        vmin=None,
        vmax=None
    )

    print("\n推理完成！所有中间结果已保存到:", output_dir)
    print("\n保存的文件列表:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == '__main__':
    main()

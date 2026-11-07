import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.feat_update import BasicMultiUpdateBlock, BasicMultiUpdateBlock_mix2
from core.geo_encoder import Combined_Geo_Encoding_Volume
from core.stereo_submodule import *
from core.depth_refine import REMP
from core.disp_warp import disp_warp
import matplotlib.pyplot as plt

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

try:
    from torchdiffeq import odeint as odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

import sys
sys.path.append('./Depth-Anything-V2-list3')
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2_decoder

    
def compute_scale_shift(monocular_depth, gt_depth, mask=None):
    """
    计算 monocular depth 和 ground truth depth 之间的 scale 和 shift.
    
    参数:
    monocular_depth (torch.Tensor): 单目深度图，形状为 (H, W) 或 (N, H, W)
    gt_depth (torch.Tensor): ground truth 深度图，形状为 (H, W) 或 (N, H, W)
    mask (torch.Tensor, optional): 有效区域的掩码，形状为 (H, W) 或 (N, H, W)
    
    返回:
    scale (float): 计算得到的 scale
    shift (float): 计算得到的 shift
    """
    
    flattened_depth_maps = monocular_depth.clone().view(-1).contiguous()
    sorted_depth_maps, _ = torch.sort(flattened_depth_maps)
    percentile_10_index = int(0.2 * len(sorted_depth_maps))
    threshold_10_percent = sorted_depth_maps[percentile_10_index]

    if mask is None:
        mask = (gt_depth > 0) & (monocular_depth > 1e-2) & (monocular_depth > threshold_10_percent)
    
    monocular_depth_flat = monocular_depth[mask]
    gt_depth_flat = gt_depth[mask]
    
    X = torch.stack([monocular_depth_flat, torch.ones_like(monocular_depth_flat)], dim=1)
    y = gt_depth_flat
    
    # 使用最小二乘法计算 [scale, shift]
    A = torch.matmul(X.t(), X) + 1e-6 * torch.eye(2, device=X.device)
    b = torch.matmul(X.t(), y)
    params = torch.linalg.solve(A, b)
    
    scale, shift = params[0].item(), params[1].item()
    
    return scale, shift


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv


class Feat_transfer_cnet(nn.Module):
    def __init__(self, dim_list, output_dim):
        super(Feat_transfer_cnet, self).__init__()

        self.res_16x = nn.Conv2d(dim_list[0]+192, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0]+96, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0]+48, output_dim, kernel_size=3, padding=1, stride=1)

    def forward(self, features, stem_x_list):
        features_list = []
        feat_16x = self.res_16x(torch.cat((features[2], stem_x_list[0]), 1))
        feat_8x = self.res_8x(torch.cat((features[1], stem_x_list[1]), 1))
        feat_4x = self.res_4x(torch.cat((features[0], stem_x_list[2]), 1))
        features_list.append([feat_4x, feat_4x])
        features_list.append([feat_8x, feat_8x])
        features_list.append([feat_16x, feat_16x])
        return features_list


class Feat_transfer(nn.Module):
    def __init__(self, dim_list):
        super(Feat_transfer, self).__init__()
        self.conv4x = nn.Sequential(
            nn.Conv2d(in_channels=int(48+dim_list[0]), out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(48), nn.ReLU()
            )
        self.conv8x = nn.Sequential(
            nn.Conv2d(in_channels=int(64+dim_list[0]), out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU()
            )
        self.conv16x = nn.Sequential(
            nn.Conv2d(in_channels=int(192+dim_list[0]), out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(192), nn.ReLU()
            )
        self.conv32x = nn.Sequential(
            nn.Conv2d(in_channels=dim_list[0], out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(160), nn.ReLU()
            )
        self.conv_up_32x = nn.ConvTranspose2d(160,
                                192,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_16x = nn.ConvTranspose2d(192,
                                64,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_8x = nn.ConvTranspose2d(64,
                                48,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        
        self.res_16x = nn.Conv2d(dim_list[0], 192, kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0], 64, kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0], 48, kernel_size=1, padding=0, stride=1)

    def forward(self, features):
        features_mono_list = []
        feat_32x = self.conv32x(features[3])
        feat_32x_up = self.conv_up_32x(feat_32x)
        feat_16x = self.conv16x(torch.cat((features[2], feat_32x_up), 1)) + self.res_16x(features[2])
        feat_16x_up = self.conv_up_16x(feat_16x)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        feat_8x_up = self.conv_up_8x(feat_8x)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)
        features_mono_list.append(feat_32x)
        return features_mono_list


class GeoLatentEncoder3D(nn.Module):
    """
    将 3D 几何编码体 geo_encoding_volume 映射到一个紧凑的 latent 表示 z_L。
    对应 P1 中 latent 流形上的坐标，但这里是以 cost volume 几何为基底的工程化近似。
    """

    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),  # [B,C,D,H,W] -> [B,latent_dim,1,1,1]
        )

    def forward(self, geo_vol: torch.Tensor) -> torch.Tensor:
        # geo_vol: [B,Cg,D,H,W]
        z = self.encoder(geo_vol).view(geo_vol.shape[0], -1)  # [B,latent_dim]
        return z


class GeoLatentField(nn.Module):
    """
    spatially-aware latent encoder：
    只在 D 维做 pooling，保留 H,W ⇒ 得到 z_field ∈ R^{B×L×H4×W4}，
    让 Flow Matching 在“局部几何流形”上起作用，而不是单一全局向量。
    """

    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, None, None)),  # [B,L,1,H,W]
        )

    def forward(self, geo_vol: torch.Tensor) -> torch.Tensor:
        # geo_vol: [B,Cg,D,H,W]
        z = self.encoder(geo_vol)[:, :, 0]  # [B,L,H,W]
        return z


class FlowVectorField(nn.Module):
    """
    条件向量场 v_θ(t, x, cond)，近似 P1 中的 Flow Matching 向量场。
    这里 cond 可以来自右图编码或几何编码（例如 match_right 的全局池化）。
    """

    def __init__(self, latent_dim: int, cond_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(1 + latent_dim + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # t: [B,1], x: [B,latent_dim], cond: [B,cond_dim]
        inp = torch.cat([t, x, cond], dim=-1)
        return self.mlp(inp)


class FlowSampler(nn.Module):
    """
    简单的 ODE 积分器：离散时间上的 Euler 积分，产生一条 latent 轨迹 φ_t 以及终点 z_L'。
    这条轨迹可视为在“几何流形”上的一条近似测地线（Geodesic Path Integration）。
    """

    def __init__(self, latent_dim: int, cond_dim: int, num_steps: int = 8):
        super().__init__()
        self.v_theta = FlowVectorField(latent_dim, cond_dim)
        self.num_steps = num_steps
        self.latent_dim = latent_dim

    def sample(self, cond: torch.Tensor):
        """
        cond: [B,cond_dim]
        返回：
            z_L_pred: [B,latent_dim]，从噪声 x0 经 ODE 推理到 t=1 的终点
            traj: List[[B,latent_dim]]，离散时间上的 latent 轨迹（可用于后续分析/正则）
        """
        B = cond.shape[0]
        device = cond.device
        x = torch.randn(B, self.latent_dim, device=device)  # x0 ~ N(0,I)
        dt = 1.0 / self.num_steps
        traj = []
        for i in range(self.num_steps):
            t = torch.full((B, 1), (i + 0.5) * dt, device=device)
            v = self.v_theta(t, x, cond)
            x = x + dt * v
            traj.append(x)
        return x, traj


class FlowSamplerField(nn.Module):
    """
    在 spatial latent field z_field(b, :, h, w) 上做 Euler/ODE 采样。
    用于更细粒度的“局部流形” Flow Matching。
    """

    def __init__(self, latent_dim: int, cond_dim: int, num_steps: int = 8, use_ode: bool = False):
        super().__init__()
        self.v_theta = FlowVectorField(latent_dim, cond_dim)
        self.num_steps = num_steps
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.use_ode = use_ode and HAS_TORCHDIFFEQ

    def _euler_sample(self, z0_field: torch.Tensor, cond_field: torch.Tensor):
        """
        z0_field: [B,L,H,W] latent 初始场（可用噪声或 encoder 输出）
        cond_field: [B,Cc,H,W] 条件场（例如 match_right 经过 conv 压缩）
        """
        B, L, H, W = z0_field.shape
        x = z0_field  # 也可以改为 torch.randn_like(z0_field)
        dt = 1.0 / self.num_steps
        traj = []
        for i in range(self.num_steps):
            t_scalar = (i + 0.5) * dt
            # 将 (B,H,W) 像素视为 batch 维，逐像素调用 v_theta
            x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, L)  # [BHW,L]
            cond_flat = cond_field.permute(0, 2, 3, 1).reshape(B * H * W, self.cond_dim)  # [BHW,Cc]
            t = torch.full((B * H * W, 1), t_scalar, device=x.device, dtype=x.dtype)
            v_flat = self.v_theta(t, x_flat, cond_flat)
            x_flat = x_flat + dt * v_flat
            x = x_flat.view(B, H, W, L).permute(0, 3, 1, 2).contiguous()
            traj.append(x)
        return x, traj

    def _ode_sample(self, z0_field: torch.Tensor, cond_field: torch.Tensor):
        """
        使用 torchdiffeq.odeint 做更高精度求解（若安装了 torchdiffeq）。
        这里同样把 (B,H,W) 像素 flatten 成 batch。
        """
        if not HAS_TORCHDIFFEQ:
            # fallback 到 Euler
            return self._euler_sample(z0_field, cond_field)

        B, L, H, W = z0_field.shape
        x0_flat = z0_field.permute(0, 2, 3, 1).reshape(B * H * W, L)  # [BHW,L]
        cond_flat = cond_field.permute(0, 2, 3, 1).reshape(B * H * W, self.cond_dim)  # [BHW,Cc]

        class ODEFunc(nn.Module):
            def __init__(self, v_theta, cond):
                super().__init__()
                self.v_theta = v_theta
                self.cond = cond

            def forward(self, t, x_flat):
                # t: scalar tensor, x_flat: [BHW,L]
                t_batch = t.expand(x_flat.shape[0], 1)
                return self.v_theta(t_batch, x_flat, self.cond)

        func = ODEFunc(self.v_theta, cond_flat)
        t_eval = torch.linspace(0.0, 1.0, steps=self.num_steps, device=x0_flat.device, dtype=x0_flat.dtype)
        x_traj = odeint(func, x0_flat, t_eval, method="rk4")  # [T,BHW,L]
        traj = []
        for i in range(self.num_steps):
            x_i = x_traj[i]
            x_field = x_i.view(B, H, W, L).permute(0, 3, 1, 2).contiguous()
            traj.append(x_field)
        x_final = traj[-1]
        return x_final, traj

    def sample(self, z0_field: torch.Tensor, cond_field: torch.Tensor):
        if self.use_ode:
            return self._ode_sample(z0_field, cond_field)
        else:
            return self._euler_sample(z0_field, cond_field)


class LatentToDispOffset(nn.Module):
    """
    将 latent z_L' 解码为 1/4 尺度上的视差偏置 disp_offset_4x。
    这里通过一个全局 embedding + 局部特征 (例如 match_left) 的小网络实现。
    """

    def __init__(self, latent_dim: int, local_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim + local_channels, hidden_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, z: torch.Tensor, local_feat: torch.Tensor) -> torch.Tensor:
        """
        z: [B,latent_dim]
        local_feat: [B,C_l,H,W]，例如 match_left 特征
        返回：
            disp_offset: [B,1,H,W]，1/4 尺度上的视差偏置
        """
        B, C_l, H, W = local_feat.shape
        z_map = self.fc(z).view(B, -1, 1, 1).expand(B, -1, H, W)
        x = torch.cat([z_map, local_feat], dim=1)
        disp_offset = self.conv(x)
        return disp_offset


class EndoStereoFM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        mono_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        dim_list_ = mono_model_configs[self.args.encoder]['features']
        dim_list = []
        dim_list.append(dim_list_)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feat_transfer = Feat_transfer(dim_list)
        self.feat_transfer_cnet = Feat_transfer_cnet(dim_list, output_dim=args.hidden_dims[0])


        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.stem_8 = nn.Sequential(
            BasicConv_IN(48, 96, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(96), nn.ReLU()
            )

        self.stem_16 = nn.Sequential(
            BasicConv_IN(96, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 192, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(192), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        depth_anything = DepthAnythingV2(**mono_model_configs[args.encoder])
        depth_anything_decoder = DepthAnythingV2_decoder(**mono_model_configs[args.encoder])
        # Keep the architecture unchanged, but make the frozen monocular
        # checkpoint configurable.  The old Windows path made training fail
        # immediately on Linux machines.
        depth_ckpt = getattr(args, 'depth_ckpt', None)
        if depth_ckpt is None or str(depth_ckpt).strip() == '':
            depth_ckpt = f'F:/depth_anything_v2_{args.encoder}.pth'
        depth_ckpt = os.path.expanduser(str(depth_ckpt))
        if not os.path.isfile(depth_ckpt):
            raise FileNotFoundError(
                f'Depth Anything checkpoint not found: {depth_ckpt}. '
                'Set depth_ckpt in the training config.'
            )
        state_dict_dpt = torch.load(depth_ckpt, map_location='cpu')
        depth_anything.load_state_dict(state_dict_dpt, strict=True)
        depth_anything_decoder.load_state_dict(state_dict_dpt, strict=False)
        self.mono_encoder = depth_anything.pretrained
        self.mono_decoder = depth_anything.depth_head
        self.feat_decoder = depth_anything_decoder.depth_head
        self.mono_encoder.requires_grad_(False)
        self.mono_decoder.requires_grad_(False)

        del depth_anything, state_dict_dpt, depth_anything_decoder
        self.REMP = REMP()


        self.update_block_mix_stereo = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)
        self.update_block_mix_mono = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)


        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

        # ---- Geodesic-aware ----
        # 是否在迭代更新中使用几何一致性门控 mono/stereo 分支
        self.use_geo_gating = getattr(args, "use_geo_gating", True)
        # 是否在 forward 输出 geodesic 信息，用于额外 loss / 可视化
        self.use_geo_outputs = getattr(args, "use_geo_outputs", True)

        # ---- Flow Matching + ODE（显式 Geodesic Path Integration） ----
        # 是否启用 latent 流匹配分支，用于改进 mono 先验 disp_mono_4x
        self.use_flow_matching = getattr(args, "use_flow_matching", True)
        # 是否使用 spatially-aware latent（保留 H,W 结构）
        self.use_spatial_latent = getattr(args, "use_spatial_latent", True)
        # 是否在 spatial latent 上使用 torchdiffeq.odeint（若环境已安装）
        self.use_ode_solver = getattr(args, "use_ode_solver", True)
        self.flow_latent_dim = getattr(args, "flow_latent_dim", 128)
        self.flow_cond_dim = getattr(args, "flow_cond_dim", 128)
        self.flow_steps = getattr(args, "flow_steps", 8)

        if self.use_flow_matching:
            # geo_encoding_volume 的通道数与 hourglass 输出一致（这里为 8）
            if self.use_spatial_latent:
                # spatially-aware：latent field + 条件 field + field 级别的 Flow Matching
                self.geo_latent_field = GeoLatentField(in_channels=8, latent_dim=self.flow_latent_dim)
                # 条件场来自右图匹配特征的卷积压缩
                self.flow_cond_conv = nn.Conv2d(96, self.flow_cond_dim, kernel_size=1, stride=1, padding=0)
                self.flow_sampler_field = FlowSamplerField(
                    self.flow_latent_dim, self.flow_cond_dim,
                    num_steps=self.flow_steps,
                    use_ode=self.use_ode_solver,
                )
                # 直接把 latent field 解码成 1/4 尺度 disp 偏置
                self.latent_field_to_disp = nn.Conv2d(self.flow_latent_dim, 1, kernel_size=3, stride=1, padding=1)
            else:
                # 旧版：全局 pooling latent + 全局 cond 向量
                self.geo_latent_encoder = GeoLatentEncoder3D(in_channels=8, latent_dim=self.flow_latent_dim)
                # 使用 match_right 的全局特征作为条件向量
                self.flow_cond_proj = nn.Linear(96, self.flow_cond_dim)  # 96 是 self.desc 输出通道
                self.flow_sampler = FlowSampler(self.flow_latent_dim, self.flow_cond_dim, num_steps=self.flow_steps)
                # 将 latent 解码为 1/4 尺度上的 disp 偏置，local_feat 取 match_left
                self.latent_to_disp = LatentToDispOffset(self.flow_latent_dim, local_channels=96)

    def infer_mono(self, image1, image2):
        height_ori, width_ori = image1.shape[2:]
        resize_image1 = F.interpolate(image1, scale_factor=14 / 16, mode='bilinear', align_corners=True)
        resize_image2 = F.interpolate(image2, scale_factor=14 / 16, mode='bilinear', align_corners=True)

        patch_h, patch_w = resize_image1.shape[-2] // 14, resize_image1.shape[-1] // 14
        features_left_encoder = self.mono_encoder.get_intermediate_layers(resize_image1, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        features_right_encoder = self.mono_encoder.get_intermediate_layers(resize_image2, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        depth_mono = self.mono_decoder(features_left_encoder, patch_h, patch_w)
        depth_mono = F.relu(depth_mono)
        depth_mono = F.interpolate(depth_mono, size=(height_ori, width_ori), mode='bilinear', align_corners=False)
        features_left_4x, features_left_8x, features_left_16x, features_left_32x = self.feat_decoder(features_left_encoder, patch_h, patch_w)
        features_right_4x, features_right_8x, features_right_16x, features_right_32x = self.feat_decoder(features_right_encoder, patch_h, patch_w)

        return depth_mono, [features_left_4x, features_left_8x, features_left_16x, features_left_32x], [features_right_4x, features_right_8x, features_right_16x, features_right_32x]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        # with autocast(enabled=self.args.mixed_precision):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with torch.autocast(device_type='cuda', dtype=torch.float32): 
            depth_mono, features_mono_left, features_mono_right = self.infer_mono(image1, image2)

        scale_factor = 0.25
        size = (int(depth_mono.shape[-2] * scale_factor), int(depth_mono.shape[-1] * scale_factor))

        disp_mono_4x = F.interpolate(depth_mono, size=size, mode='bilinear', align_corners=False)

        features_left = self.feat_transfer(features_mono_left)
        features_right = self.feat_transfer(features_mono_right)
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)
        stem_16x = self.stem_16(stem_8x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)

        stem_x_list = [stem_16x, stem_8x, stem_4x]
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
        gwc_volume = self.corr_stem(gwc_volume)
        gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
        geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

        # Init disp from geometry encoding volume
        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
        init_disp = disparity_regression(prob, self.args.max_disp//4)
        
        del prob, gwc_volume

        # ---- Flow Matching + ODE 分支：在 latent 几何流形上改进 mono 先验 ----
        # Flow Matching latent 信息（供 loss 使用）
        fm_z_L = None
        fm_z_L_pred = None
        fm_traj = None

        if self.use_flow_matching:
            B, C_m, H4, W4 = match_right.shape
            if self.use_spatial_latent:
                # 1) spatially-aware latent field：只在 D 上 pooling，保留 (H4, W4)
                z_field = self.geo_latent_field(geo_encoding_volume)  # [B,L,H4,W4]
                fm_z_L = z_field.detach()
                # 2) 条件场：来自 match_right 的 1×1 conv 压缩
                cond_field = self.flow_cond_conv(match_right)  # [B,cond_dim,H4,W4]
                # 3) Flow Matching 采样：在每个 pixel latent 上沿 ODE/离散流积分
                z_field_pred, traj_field = self.flow_sampler_field.sample(z_field, cond_field)
                fm_z_L_pred = z_field_pred
                # [B,T,L,H,W] 方便后续可视化/正则
                fm_traj = torch.stack(traj_field, dim=1)
                # 4) latent field → 直接预测 1/4 尺度 disp 偏置
                disp_offset_4x = self.latent_field_to_disp(z_field_pred)  # [B,1,H4,W4]
                disp_mono_4x = disp_mono_4x + disp_offset_4x
            else:
                # 1) 从 geo_encoding_volume 提取全局 latent 表示 z_L
                z_L = self.geo_latent_encoder(geo_encoding_volume)  # [B,latent_dim]
                fm_z_L = z_L.detach()
                # 2) 从 match_right 提取全局条件向量 cond
                cond_feat = F.adaptive_avg_pool2d(match_right, output_size=1).view(B, C_m)  # [B,96]
                cond_vec = self.flow_cond_proj(cond_feat)  # [B,cond_dim]
                # 3) Flow Matching 采样：从噪声出发沿 ODE 生成 z_L_pred 以及轨迹
                z_L_pred, traj = self.flow_sampler.sample(cond_vec)  # [B,latent_dim], List[[B,latent_dim]]
                fm_z_L_pred = z_L_pred
                fm_traj = torch.stack(traj, dim=1)  # [B,T,latent_dim]
                # 4) 将 latent 解码为 1/4 尺度的 disp 偏置，直接校正 mono 先验 disp_mono_4x
                disp_offset_4x = self.latent_to_disp(z_L_pred, match_left)  # [B,1,H4,W4]
                disp_mono_4x = disp_mono_4x + disp_offset_4x

        if not test_mode:
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)

        # cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
        cnet_list = self.feat_transfer_cnet(features_mono_left, stem_x_list)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [torch.relu(x) for x in inp_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]
        net_list_mono = [x.clone() for x in net_list]

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(),
                           radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1).contiguous()
        disp = init_disp
        disp_preds = []

        # 用于记录最后一次几何一致性 / 不确定性（供 geodesic loss 使用）
        last_geo_consistency = None
        last_uncertainty = None
        for itr in range(iters):
            disp = disp.detach()
            if itr >= int(1):
                disp_mono_4x = disp_mono_4x.detach()
            geo_feat = geo_fn(disp, coords)
            if itr > int(iters-8):
                if itr == int(iters-7):
                    bs, _, _, _ = disp.shape
                    for i in range(bs):
                        with torch.autocast(device_type='cuda', dtype=torch.float32): 
                            scale, shift = compute_scale_shift(disp_mono_4x[i].clone().squeeze(1).to(torch.float32), disp[i].clone().squeeze(1).to(torch.float32))
                        disp_mono_4x[i] = scale * disp_mono_4x[i] + shift
                
                warped_right_mono = disp_warp(features_right[0], disp_mono_4x.clone().to(features_right[0].dtype))[0]
                flaw_mono = warped_right_mono - features_left[0]

                warped_right_stereo = disp_warp(features_right[0], disp.clone().to(features_right[0].dtype))[0]
                flaw_stereo = warped_right_stereo - features_left[0]
                geo_feat_mono = geo_fn(disp_mono_4x, coords)

                # ---- Geodesic-aware prior & uncertainty (在特征流形上的几何误差) ----
                # 利用 warp 后特征差异作为几何距离：dist 大 ⇒ 视差不可靠（高不确定性），一致性小
                geo_dist_stereo = flaw_stereo.pow(2).sum(dim=1)  # [B, H4, W4]
                geo_consistency_stereo = torch.exp(
                    -geo_dist_stereo / (features_left[0].shape[1] * 0.1 + 1e-6)
                )
                uncertainty_stereo = torch.sqrt(geo_dist_stereo + 1e-6)  # 几何不确定性

                # 记录最后一次，用于 geodesic loss / 输出
                last_geo_consistency = geo_consistency_stereo
                last_uncertainty = uncertainty_stereo

                # 基于几何一致性在 stereo / mono 之间做门控
                if self.use_geo_gating:
                    w_stereo = geo_consistency_stereo          # [B, H4, W4]
                    w_mono = 1.0 - geo_consistency_stereo      # [B, H4, W4]
                    flaw_stereo_g = flaw_stereo * w_stereo.unsqueeze(1)
                    flaw_mono_g = flaw_mono * w_mono.unsqueeze(1)
                    geo_feat_g = geo_feat * w_stereo.unsqueeze(1) + geo_feat_mono * w_mono.unsqueeze(1)
                else:
                    flaw_stereo_g = flaw_stereo
                    flaw_mono_g = flaw_mono
                    geo_feat_g = geo_feat

            if itr <= int(iters-8):
                net_list, mask_feat_4, delta_disp = self.update_block(
                    net_list, inp_list, geo_feat, disp,
                    iter16=self.args.n_gru_layers == 3, iter08=self.args.n_gru_layers >= 2
                )
            else:
                # 将几何先验融合进双分支更新块
                net_list, mask_feat_4, delta_disp = self.update_block_mix_stereo(
                    net_list, inp_list,
                    flaw_stereo_g, disp, geo_feat_g,
                    flaw_mono_g, disp_mono_4x, geo_feat_mono,
                    iter16=self.args.n_gru_layers == 3, iter08=self.args.n_gru_layers >= 2
                )
                net_list_mono, mask_feat_4_mono, delta_disp_mono = self.update_block_mix_mono(
                    net_list_mono, inp_list,
                    flaw_mono_g, disp_mono_4x, geo_feat_mono,
                    flaw_stereo_g, disp, geo_feat,
                    iter16=self.args.n_gru_layers == 3, iter08=self.args.n_gru_layers >= 2
                )
                disp_mono_4x = disp_mono_4x + delta_disp_mono
                disp_mono_4x_up = self.upsample_disp(disp_mono_4x, mask_feat_4_mono, stem_2x)
                disp_preds.append(disp_mono_4x_up)

            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)

            if itr == iters - 1:
                refine_value = self.REMP(disp_mono_4x_up, disp_up, image1, image2)
                disp_up = disp_up + refine_value
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        init_disp = context_upsample(init_disp * 4.0, spx_pred.float()).unsqueeze(1)

        # 输出 geodesic / flow-matching 相关量，供 loss / 可视化使用
        if self.use_geo_outputs and last_geo_consistency is not None and last_uncertainty is not None:
            # 上采样到最终视差分辨率
            B, _, H_full, W_full = disp_up.shape
            geo_cons_full = F.interpolate(
                last_geo_consistency.unsqueeze(1),
                size=(H_full, W_full),
                mode="bilinear", align_corners=False
            ).squeeze(1)
            uncert_full = F.interpolate(
                last_uncertainty.unsqueeze(1),
                size=(H_full, W_full),
                mode="bilinear", align_corners=False
            ).squeeze(1)

            # 选择一个 geodesic-like 视差先验：这里用 mono 分支的 upsample 结果
            disp_geo_full = disp_mono_4x_up.detach()  # [B,1,H_full,W_full]
            geo_info = {
                "disp_geo": disp_geo_full,
                "uncertainty": uncert_full.detach(),
                "geo_consistency": geo_cons_full.detach(),
            }
            # 若启用 Flow Matching，同时输出 latent 信息以便计算 FM loss / path 正则
            if self.use_flow_matching and fm_z_L is not None and fm_z_L_pred is not None and fm_traj is not None:
                geo_info["fm_z_L"] = fm_z_L        # 目标 latent（不回传梯度）
                geo_info["fm_z_L_pred"] = fm_z_L_pred  # 预测 latent（需参与 FM loss 的梯度）
                geo_info["fm_traj"] = fm_traj       # [B,T,D]

            return init_disp, disp_preds, depth_mono, geo_info

        return init_disp, disp_preds, depth_mono

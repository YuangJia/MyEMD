#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, read_extrinsics_text_test
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import open3d as o3d
import cv2
import torch


def build_K_from_caminfo(cam):
    p = np.asarray(cam.K, dtype=np.float64).ravel()
    if p.size == 3:
        fx = fy = p[0];
        cx, cy = p[1], p[2]
    elif p.size == 4:
        fx, fy, cx, cy = p
    else:
        raise ValueError(f"Unexpected intrinsics length: {p.size}")
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float64)


def caminfo_w2c(cam):
    R_w2c = cam.R.T  # 你的 R 是 C2W
    t_w2c = cam.T  # 你的 T 是 W2C
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R_w2c
    T_w2c[:3, 3] = t_w2c
    return T_w2c


def caminfo_w2c_train(cam, R_is_c2w=True, T_is_w2c=True):
    """
    把 CameraInfo 里的 R/T 统一成 W2C (R_w2c, t_w2c).
    你的读取里默认 R 是 C2W、T 是 W2C，所以默认参数就是 (True, True)。
    如果以后改成一致的 C2W（T_c2w），传 R_is_c2w=True, T_is_w2c=False 即可。
    """
    R = np.asarray(cam.R, dtype=np.float64)
    T = np.asarray(cam.T, dtype=np.float64).reshape(3)

    if R_is_c2w:
        R_w2c = R.T
    else:
        R_w2c = R

    if T_is_w2c:
        t_w2c = T
    else:
        # 若传入的是 t_c2w，则 t_w2c = - R_w2c @ t_c2w
        t_w2c = - R_w2c @ T

    T_w2c = np.eye(4, dtype=np.float64)
    T_w2c[:3, :3] = R_w2c
    T_w2c[:3, 3] = t_w2c
    return T_w2c


def to_bool_sky_mask(sky_mask_any, W, H):
    """
    将 sky_mask（可能是 torch.Tensor / numpy 灰度 / 0/1）统一成 (H,W) 的 bool numpy。
    """
    if isinstance(sky_mask_any, torch.Tensor):
        m = sky_mask_any.detach().cpu().numpy()
    else:
        m = np.asarray(sky_mask_any)

    # squeeze 到 (H,W)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]

    # 若是 0..255 灰度，阈值到 bool（>0 或 >127 都行，按你数据而定）
    if m.dtype != np.bool_:
        # 通常 sky_mask 是 0/255 灰度，这里设 >0 即为天空
        m = (m > 0).astype(np.uint8)

    # 尺寸对齐
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    return m.astype(bool)


def visible_ids_per_camera_zbuffer(points_w, cam):
    H, W = cam.height, cam.width
    K = build_K_from_caminfo(cam)
    T = caminfo_w2c(cam)  # 你的 R 是 C2W、T 是 W2C

    R = T[:3, :3]
    t = T[:3, 3]
    Xc = (R @ points_w.T + t[:, None])  # (3, N)
    z = Xc[2, :]

    # 1) 先过滤非法/背后点，避免除零与 NaN/Inf
    valid0 = (z > 1e-6) & np.isfinite(z) & np.isfinite(Xc[0, :]) & np.isfinite(Xc[1, :])
    if not np.any(valid0):
        return None, None, None

    idx0 = np.nonzero(valid0)[0]
    Xc = Xc[:, valid0]
    z = z[valid0]

    # 2) 投影；此处不直接 cast，先保持 float
    with np.errstate(divide='ignore', invalid='ignore'):
        u_f = K[0, 0] * (Xc[0, :] / z) + K[0, 2]
        v_f = K[1, 1] * (Xc[1, :] / z) + K[1, 2]

    # 3) 过滤非有限的 u,v
    finite_uv = np.isfinite(u_f) & np.isfinite(v_f)
    if not np.any(finite_uv):
        return None, None, None

    sel = np.nonzero(finite_uv)[0]
    u_f = u_f[sel]
    v_f = v_f[sel]
    z = z[sel]
    idx = idx0[sel]

    # —— 有些 numpy 版本在 astype 前仍可能报 cast warning，稳妥起见再做一次兜底 —— #
    finite_uv2 = np.isfinite(u_f) & np.isfinite(v_f) & np.isfinite(z)
    if not np.any(finite_uv2):
        return None, None, None
    u_f = u_f[finite_uv2]
    v_f = v_f[finite_uv2]
    z = z[finite_uv2]
    idx = idx[finite_uv2]

    # 4) 四舍五入后再转 int32（此时不应再含 NaN/Inf）
    u = np.rint(u_f).astype(np.int32)
    v = np.rint(v_f).astype(np.int32)

    # 5) 图像边界过滤
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_img):
        return None, None, None
    u = u[in_img];
    v = v[in_img];
    z = z[in_img];
    idx = idx[in_img]

    # 6) z-buffer：每个像素只保留最近点
    lin = v * W + u
    order = np.argsort(z)  # 近 -> 远
    lin_sorted = lin[order]
    first = np.unique(lin_sorted, return_index=True)[1]
    keep = order[first]

    return idx[keep], u[keep], v[keep]


def visible_ids_per_camera(points_w, cam):
    """
    返回所有投影到图像范围内的点（不做 z-buffer）
    输入:
        points_w: (N,3) 世界坐标点
        cam: 包含 height, width, 内参和外参的相机对象
    输出:
        idx: 点的索引 (M,)
        u:   像素 u 坐标 (M,)
        v:   像素 v 坐标 (M,)
    """
    H, W = cam.height, cam.width
    K = build_K_from_caminfo(cam)
    T = caminfo_w2c(cam)

    # 世界坐标 -> 相机坐标
    R = T[:3, :3]
    t = T[:3, 3]
    Xc = (R @ points_w.T + t[:, None])  # (3,N)
    z = Xc[2, :]

    # 1) 过滤非法/背后点
    valid0 = (z > 1e-6) & np.isfinite(z) & np.isfinite(Xc[0, :]) & np.isfinite(Xc[1, :])
    if not np.any(valid0):
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    idx0 = np.nonzero(valid0)[0]
    Xc = Xc[:, valid0]
    z = z[valid0]

    # 2) 投影到像素
    with np.errstate(divide='ignore', invalid='ignore'):
        u_f = K[0, 0] * (Xc[0, :] / z) + K[0, 2]
        v_f = K[1, 1] * (Xc[1, :] / z) + K[1, 2]

    finite_uv = np.isfinite(u_f) & np.isfinite(v_f)
    if not np.any(finite_uv):
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    idx = idx0[finite_uv]
    u_f = u_f[finite_uv]
    v_f = v_f[finite_uv]

    # 3) 四舍五入 + 边界过滤
    u = np.rint(u_f).astype(np.int32)
    v = np.rint(v_f).astype(np.int32)
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    if not np.any(in_img):
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    return idx[in_img], u[in_img], v[in_img]


def bake_colors_from_train_cams_visible(points_w, train_cams, fallback_colors_uint8):
    N = len(points_w)
    acc = np.zeros((N, 3), dtype=np.float64)
    cnt = np.zeros((N,), dtype=np.int32)

    for cam in train_cams:
        vis_ids, uu, vv = visible_ids_per_camera_zbuffer(points_w, cam)
        if vis_ids is None: continue
        img = np.array(cam.image)  # RGB uint8
        cols = img[vv, uu, :]  # (M,3) uint8
        acc[vis_ids] += cols
        cnt[vis_ids] += 1

    colors = fallback_colors_uint8.copy()
    seen = cnt > 0
    colors[seen] = np.rint(acc[seen] / cnt[seen][:, None]).astype(np.uint8)
    return colors


def render_points_zbuf(points_w, colors_uint8, cam, radius_px=0):
    H, W = cam.height, cam.width
    K = build_K_from_caminfo(cam)
    T = caminfo_w2c(cam)

    R = T[:3, :3]
    t = T[:3, 3]
    Xc = (R @ points_w.T + t[:, None])  # (3, N)
    z = Xc[2, :]

    # 1) 过滤非法/背后点
    valid = (z > 1e-6) & np.isfinite(z) \
            & np.isfinite(Xc[0, :]) & np.isfinite(Xc[1, :])
    img = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), np.inf, dtype=np.float64)
    wrote = np.zeros((H, W), dtype=bool)  # 有像素被写过的掩码

    if not np.any(valid):
        return img, wrote, depth

    Xc = Xc[:, valid]
    z = z[valid]
    cols = colors_uint8[valid]

    # 2) 投影
    with np.errstate(divide='ignore', invalid='ignore'):
        xn = Xc[0, :] / z
        yn = Xc[1, :] / z
        u_f = K[0, 0] * xn + K[0, 2]
        v_f = K[1, 1] * yn + K[1, 2]

    # 3) 过滤无效/越界
    finite_uv = np.isfinite(u_f) & np.isfinite(v_f)
    if not np.any(finite_uv):
        return img, wrote, depth
    u = np.round(u_f[finite_uv]).astype(np.int32)
    v = np.round(v_f[finite_uv]).astype(np.int32)
    z = z[finite_uv]
    cols = cols[finite_uv]

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_img):
        return img, wrote, depth
    u = u[in_img];
    v = v[in_img];
    z = z[in_img];
    cols = cols[in_img]

    # 4) z-buffer 上色（并标记 wrote）
    if radius_px <= 0:
        for ui, vi, zi, ci in zip(u, v, z, cols):
            if zi < depth[vi, ui]:
                depth[vi, ui] = zi
                img[vi, ui] = ci
                wrote[vi, ui] = True
    else:
        r2 = radius_px * radius_px
        for ui, vi, zi, ci in zip(u, v, z, cols):
            x0 = max(0, ui - radius_px);
            x1 = min(W - 1, ui + radius_px)
            y0 = max(0, vi - radius_px);
            y1 = min(H - 1, vi + radius_px)
            for yy in range(y0, y1 + 1):
                dy2 = (yy - vi) * (yy - vi)
                for xx in range(x0, x1 + 1):
                    if (xx - ui) * (xx - ui) + dy2 <= r2 and zi < depth[yy, xx]:
                        depth[yy, xx] = zi
                        img[yy, xx] = ci
                        wrote[yy, xx] = True

    return img, wrote, depth


def downsample_pointcloud(pcd, ratio=0.6):
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    n = pts.shape[0]

    # 随机采样
    idx = np.random.choice(n, int(n * ratio), replace=False)
    pts_ds, cols_ds, normals_ds = pts[idx], cols[idx], normals[idx]

    # 构造新的点云
    pcd_down = BasicPointCloud(points=pts_ds, colors=cols_ds, normals=normals_ds)
    return pcd_down


def circle_offsets(radius_px):
    offs = []
    r2 = radius_px * radius_px
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            if dx * dx + dy * dy <= r2:
                offs.append((dx, dy))
    return np.asarray(offs, dtype=np.int32)  # (K,2)


def render_points_zbuf_splat_fast(points_w, colors_uint8, cam, radius_px=2):
    assert radius_px > 0
    H, W = cam.height, cam.width
    K = build_K_from_caminfo(cam)
    T = caminfo_w2c(cam)

    R = T[:3, :3]
    t = T[:3, 3]
    Xc = (R @ points_w.T + t[:, None])  # (3, N)
    z = Xc[2, :]

    valid = (z > 1e-6) & np.isfinite(z) & np.isfinite(Xc[0, :]) & np.isfinite(Xc[1, :])
    if not np.any(valid):
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), bool), np.full((H, W), np.inf)

    Xc = Xc[:, valid];
    z = z[valid];
    cols = colors_uint8[valid]

    with np.errstate(divide='ignore', invalid='ignore'):
        xn = Xc[0, :] / z
        yn = Xc[1, :] / z
        u_f = K[0, 0] * xn + K[0, 2]
        v_f = K[1, 1] * yn + K[1, 2]

    finite_uv = np.isfinite(u_f) & np.isfinite(v_f)
    if not np.any(finite_uv):
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), bool), np.full((H, W), np.inf)

    u0 = np.round(u_f[finite_uv]).astype(np.int32)
    v0 = np.round(v_f[finite_uv]).astype(np.int32)
    z = z[finite_uv]
    cols = cols[finite_uv]

    img_flat = np.zeros((H * W, 3), dtype=np.uint8)
    depth_flat = np.full(H * W, np.inf, dtype=np.float64)
    wrote_flat = np.zeros(H * W, dtype=bool)

    offs = circle_offsets(radius_px)  # (K,2)
    # K 次循环；每次都是“全量点 + 矢量化”处理
    for dx, dy in offs:
        u = u0 + dx
        v = v0 + dy
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_img):
            continue
        u = u[in_img];
        v = v[in_img];
        zc = z[in_img];
        cc = cols[in_img]

        pix = v * W + u
        order = np.lexsort((zc, pix))  # 主键：pix；组内 z 升序
        pix_sorted = pix[order]
        first_idx = np.unique(pix_sorted, return_index=True)[1]
        sel = order[first_idx]

        # 对当前偏移产生的候选，尝试写入全局 z-buffer
        p_sel = pix[sel]
        z_sel = zc[sel]

        # 对比并更新（向量化）
        better = z_sel < depth_flat[p_sel]
        if np.any(better):
            idx = p_sel[better]
            depth_flat[idx] = z_sel[better]
            img_flat[idx] = cc[sel][better]
            wrote_flat[idx] = True

    img = img_flat.reshape(H, W, 3)
    depth = depth_flat.reshape(H, W)
    wrote = wrote_flat.reshape(H, W)
    return img, wrote, depth


def render_points_zbuf_splat_roi(points_w, colors_uint8, cam, radius_small=2, radius_big=5, roi_ratio=0.6):
    """
    ROI (右下角) 半径更大
    """
    H, W = cam.height, cam.width
    K = build_K_from_caminfo(cam)
    T = caminfo_w2c(cam)

    R = T[:3, :3];
    t = T[:3, 3]
    Xc = (R @ points_w.T + t[:, None])
    z = Xc[2, :]

    valid = (z > 1e-6) & np.isfinite(z) & np.isfinite(Xc[0, :]) & np.isfinite(Xc[1, :])
    if not np.any(valid):
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), bool), np.full((H, W), np.inf)

    Xc = Xc[:, valid];
    z = z[valid];
    cols = colors_uint8[valid]

    with np.errstate(divide='ignore', invalid='ignore'):
        u_f = K[0, 0] * (Xc[0, :] / z) + K[0, 2]
        v_f = K[1, 1] * (Xc[1, :] / z) + K[1, 2]

    finite_uv = np.isfinite(u_f) & np.isfinite(v_f)
    if not np.any(finite_uv):
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), bool), np.full((H, W), np.inf)

    u0 = np.round(u_f[finite_uv]).astype(np.int32)
    v0 = np.round(v_f[finite_uv]).astype(np.int32)
    z = z[finite_uv]
    cols = cols[finite_uv]

    img_flat = np.zeros((H * W, 3), dtype=np.uint8)
    depth_flat = np.full(H * W, np.inf, dtype=np.float64)
    wrote_flat = np.zeros(H * W, dtype=bool)

    # 划分 ROI
    in_roi = (u0 > W * roi_ratio) & (v0 > H * roi_ratio)
    u_roi, v_roi, z_roi, cols_roi = u0[in_roi], v0[in_roi], z[in_roi], cols[in_roi]
    u_non, v_non, z_non, cols_non = u0[~in_roi], v0[~in_roi], z[~in_roi], cols[~in_roi]

    def splat(u0, v0, z, cols, radius):
        offs = circle_offsets(radius)
        for dx, dy in offs:
            u = u0 + dx;
            v = v0 + dy
            in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not np.any(in_img): continue
            u = u[in_img];
            v = v[in_img];
            zc = z[in_img];
            cc = cols[in_img]

            pix = v * W + u
            order = np.lexsort((zc, pix))
            pix_sorted = pix[order]
            first_idx = np.unique(pix_sorted, return_index=True)[1]
            sel = order[first_idx]

            p_sel = pix[sel];
            z_sel = zc[sel]
            better = z_sel < depth_flat[p_sel]
            if np.any(better):
                idx = p_sel[better]
                depth_flat[idx] = z_sel[better]
                img_flat[idx] = cc[sel][better]
                wrote_flat[idx] = True

    # 非 ROI 部分
    if len(u_non) > 0:
        splat(u_non, v_non, z_non, cols_non, radius_small)
    # ROI 部分
    if len(u_roi) > 0:
        splat(u_roi, v_roi, z_roi, cols_roi, radius_big)

    img = img_flat.reshape(H, W, 3)
    depth = depth_flat.reshape(H, W)
    wrote = wrote_flat.reshape(H, W)
    return img, wrote, depth


def render_points_zbuf_fast_no(points_w, colors_uint8, cam):
    H, W = cam.height, cam.width
    K = build_K_from_caminfo(cam)
    T = caminfo_w2c(cam)

    R = T[:3, :3]
    t = T[:3, 3]
    Xc = (R @ points_w.T + t[:, None])  # (3, N)
    z = Xc[2, :]

    # 1) 过滤非法/背后点
    valid = (z > 1e-6) & np.isfinite(z) & np.isfinite(Xc[0, :]) & np.isfinite(Xc[1, :])
    if not np.any(valid):
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), bool), np.full((H, W), np.inf)

    Xc = Xc[:, valid]
    z = z[valid]
    cols = colors_uint8[valid]

    # 2) 投影（全向量化）
    with np.errstate(divide='ignore', invalid='ignore'):
        xn = Xc[0, :] / z
        yn = Xc[1, :] / z
        u_f = K[0, 0] * xn + K[0, 2]
        v_f = K[1, 1] * yn + K[1, 2]

    finite_uv = np.isfinite(u_f) & np.isfinite(v_f)
    if not np.any(finite_uv):
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), bool), np.full((H, W), np.inf)

    u = np.round(u_f[finite_uv]).astype(np.int32)
    v = np.round(v_f[finite_uv]).astype(np.int32)
    z = z[finite_uv]
    cols = cols[finite_uv]

    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_img):
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), bool), np.full((H, W), np.inf)

    u = u[in_img];
    v = v[in_img];
    z = z[in_img];
    cols = cols[in_img]

    # 3) z-buffer：按像素索引分组取最小深度（排序 + unique）
    pix = v * W + u  # 展平后的像素索引
    order = np.lexsort((z, pix))  # 主键：pix；组内按 z 升序
    pix_sorted = pix[order]
    first_idx = np.unique(pix_sorted, return_index=True)[1]
    sel = order[first_idx]  # 每个像素 z 最小的那个点

    img_flat = np.zeros((H * W, 3), dtype=np.uint8)
    depth_flat = np.full(H * W, np.inf, dtype=np.float64)
    wrote_flat = np.zeros(H * W, dtype=bool)

    img_flat[pix[sel]] = cols[sel]
    depth_flat[pix[sel]] = z[sel]
    wrote_flat[pix[sel]] = True

    img = img_flat.reshape(H, W, 3)
    depth = depth_flat.reshape(H, W)
    wrote = wrote_flat.reshape(H, W)
    return img, wrote, depth


def project_points_mask(points_w, K, T_w2c, width, height):
    """最基础：落到像素置 1（无可见性/无畸变）"""
    R = T_w2c[:3, :3];
    t = T_w2c[:3, 3]
    Xc = (R @ points_w.T + t[:, None])  # (3,N)
    z = Xc[2, :]
    valid = z > 1e-6
    if not np.any(valid):
        return np.zeros((height, width), dtype=np.uint8)

    Xc = Xc[:, valid]
    xn = Xc[0, :] / Xc[2, :]
    yn = Xc[1, :] / Xc[2, :]
    u = K[0, 0] * xn + K[0, 2]
    v = K[1, 1] * yn + K[1, 2]
    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)

    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_img];
    v = v[in_img]

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[v, u] = 1
    return mask


def project_points_image(points_w, colors_uint8, cam):
    """
    简单点投影（无 z-buffer，无 splatting）
    输入:
        points_w: (N,3) 世界坐标点
        colors_uint8: (N,3) 每个点的颜色 (uint8)
        cam: 含有 height, width, 内参和外参的相机对象
    输出:
        img: (H,W,3) 投影后的图像，uint8
    """
    H, W = cam.height, cam.width
    K = build_K_from_caminfo(cam)
    T_w2c = caminfo_w2c(cam)

    # 世界点 -> 相机坐标
    R = T_w2c[:3, :3]
    t = T_w2c[:3, 3]
    Xc = (R @ points_w.T + t[:, None])  # (3,N)
    z = Xc[2, :]

    # 只保留在相机前方的点
    valid = z > 1e-6
    if not np.any(valid):
        return np.zeros((H, W, 3), dtype=np.uint8)

    Xc = Xc[:, valid]
    cols = colors_uint8[valid]

    # 归一化投影
    xn = Xc[0, :] / Xc[2, :]
    yn = Xc[1, :] / Xc[2, :]
    u = K[0, 0] * xn + K[0, 2]
    v = K[1, 1] * yn + K[1, 2]

    # 转换成像素坐标
    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)

    # 只保留在图像范围内的点
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_img):
        return np.zeros((H, W, 3), dtype=np.uint8)

    u = u[in_img]
    v = v[in_img]
    cols = cols[in_img]

    # 生成图像并直接覆盖像素（后写入的点会覆盖前面的点）
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[v, u] = cols

    return img


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    dynamic_image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    K: np.array
    sky_mask: np.array
    normal: np.array
    depth: np.array
    time: float
    time_diff: float
    cam_no: int = 0


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    cam_frustum_aabb: np.array


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, dynamic_mask_folder=None, sky_seg=False,
                      load_normal=False,
                      load_depth=False, time_diff=1.0, total_length=100):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        time = (extr.id - 1) * time_diff
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # with Image.open(image_path) as img:
        #     image = img.convert("RGB")
        dynamic_image = None
        if dynamic_mask_folder != None:
            filename = os.path.splitext(os.path.basename(extr.name))[0] + ".png"
            dynamic_path = os.path.join(dynamic_mask_folder, filename)
            # dynamic_image = Image.open(dynamic_path)
            with Image.open(dynamic_path) as dynamic:
                dynamic_image = dynamic.convert("L")

        # #sky mask
        if sky_seg:
            if "test_images" in images_folder:
                sky_path = image_path.replace("GaussianPro/euvs_data/trainset", "submission/third_version_skymask")
                sky_path = sky_path.replace("test_images", "images")
                # print(f"sky_path_2:{sky_path}")
            else:
                sky_path = image_path.replace("images", "sky_mask")
                # print(f"sky_path_1:{sky_path}")

            sky_mask = cv2.imread(sky_path, cv2.IMREAD_GRAYSCALE)  # shape=(H,W), uint8
            # 反转（原来白色255→0，黑色0→255）
            # sky_mask = 255 - sky_mask
            # sky_mask = (sky_mask > 0).astype(np.uint8)
        else:
            sky_mask = None

        if load_normal:
            normal_path = image_path.replace("images", "normals")[:-4] + ".npy"
            normal = np.load(normal_path).astype(np.float32)
            normal = (normal - 0.5) * 2.0
        else:
            normal = None

        if load_depth:
            # depth_path = image_path.replace("images", "monodepth")[:-4]+".npy"
            depth_path = image_path.replace("images", "metricdepth")[:-4] + ".npy"
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = None

        # to do level3
        cam_no = 0
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, dynamic_image=dynamic_image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              K=intr.params, sky_mask=sky_mask, normal=normal, depth=depth, time=time,
                              time_diff=time_diff, cam_no=cam_no)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def fetchPlyAnchor(path, sample_ratio=0.6):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    # =========== 随机降采样 ===========
    # N = positions.shape[0]
    # if sample_ratio < 1.0 and "level1" in path:
    #     idx = np.random.choice(N, int(N * sample_ratio), replace=False)
    #     positions = positions[idx]
    #     colors = colors[idx]
    #     normals = normals[idx]
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def load_and_merge_plys(colmap_path, anchor_path, sample_ratio=1.0):
    """
    加载并合并 COLMAP 点云和 Anchor 点云

    参数:
        colmap_path: 普通 COLMAP 点云路径 (.ply)
        anchor_path: 带法向量的 Anchor 点云路径 (.ply)
        sample_ratio: 对 Anchor 点云进行下采样比例 (0.0~1.0)

    返回:
        合并后的 BasicPointCloud
    """
    # 加载 COLMAP 点云 (无 normals)
    colmap_ply = PlyData.read(colmap_path)
    colmap_vertices = colmap_ply['vertex']
    colmap_points = np.vstack([colmap_vertices['x'], colmap_vertices['y'], colmap_vertices['z']]).T
    colmap_colors = np.vstack([colmap_vertices['red'], colmap_vertices['green'], colmap_vertices['blue']]).T / 255.0
    colmap_normals = np.zeros_like(colmap_points)  # COLMAP 无 normals，用零填充

    # 加载 Anchor 点云 (带 normals)
    anchor_ply = PlyData.read(anchor_path)
    anchor_vertices = anchor_ply['vertex']
    anchor_points = np.vstack([anchor_vertices['x'], anchor_vertices['y'], anchor_vertices['z']]).T
    anchor_colors = np.vstack([anchor_vertices['red'], anchor_vertices['green'], anchor_vertices['blue']]).T / 255.0
    anchor_normals = np.vstack([anchor_vertices['nx'], anchor_vertices['ny'], anchor_vertices['nz']]).T

    # 对 Anchor 点云下采样 (可选)
    if sample_ratio < 1.0:
        num_samples = int(len(anchor_points) * sample_ratio)
        indices = np.random.choice(len(anchor_points), num_samples, replace=False)
        anchor_points = anchor_points[indices]
        anchor_colors = anchor_colors[indices]
        anchor_normals = anchor_normals[indices]

    # 合并点云
    merged_points = np.concatenate([colmap_points, anchor_points], axis=0)
    merged_colors = np.concatenate([colmap_colors, anchor_colors], axis=0)
    merged_normals = np.concatenate([colmap_normals, anchor_normals], axis=0)

    aabb_min = merged_points.min(axis=0)
    aabb_max = merged_points.max(axis=0)
    aabb = np.stack([aabb_min, aabb_max], axis=0)

    return BasicPointCloud(points=merged_points, colors=merged_colors, normals=merged_normals), aabb


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8, sky_seg=False, load_normal=False, load_depth=False):
    if not eval:
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        reading_dir = "images" if images == None else images
        # dynamic_dir = "masks"
        dynamic_dir = "/data/ljl/jya_repo/GaussianPro/output/level1/scene_2/test/sky_30000/renders_skymask"

        # cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
        #                                        images_folder=os.path.join(path, reading_dir), dynamic_mask_folder=os.path.join(path,dynamic_dir),
        #                                        sky_seg=sky_seg, load_normal=load_normal, load_depth=load_depth)
        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                               images_folder=os.path.join(path, reading_dir),
                                               dynamic_mask_folder=dynamic_dir,
                                               sky_seg=sky_seg, load_normal=load_normal, load_depth=load_depth)
        cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    else:

        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        total_length = len(cam_extrinsics)
        time_diff = 1.0 / total_length

        reading_dir = "images" if images == None else images
        dynamic_dir = "masks"

        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                               images_folder=os.path.join(path, reading_dir),
                                               dynamic_mask_folder=os.path.join(path, dynamic_dir),
                                               sky_seg=sky_seg, load_normal=load_normal, load_depth=load_depth,
                                               time_diff=time_diff, total_length=total_length)
        cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

        cameras_extrinsic_file = os.path.join(path, "images.txt")
        cameras_intrinsic_file = os.path.join(path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text_test(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        reading_dir = "test_images"
        cam_infos_unsorted_test = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                                    images_folder=os.path.join(path, reading_dir),
                                                    dynamic_mask_folder=None,
                                                    sky_seg=False, load_normal=load_normal, load_depth=load_depth,
                                                    time_diff=time_diff, total_length=total_length)
        cam_infos_test = sorted(cam_infos_unsorted_test.copy(), key=lambda x: x.image_name)
        # cam_infos_unsorted_test = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
        #                                 images_folder=os.path.join(path, reading_dir), dynamic_mask_folder=None,
        #                                 sky_seg=sky_seg, load_normal=load_normal, load_depth=load_depth，time_diff=time_diff,total_length=total_length)

    if eval:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos_test

    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "sparse/0/points.ply")
    # anchor_ply_path = os.path.join(path, "sparse/0/pointsAnchor_0.ply")
    anchor_ply_path = os.path.join(path, "sparse/0/merged_anchor.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    try:
        print(f"ply_path:{ply_path}")
        print(f"anchor_ply_path:{anchor_ply_path}")
        pcd, aabb = load_and_merge_plys(ply_path, anchor_ply_path)
        # pcd = fetchPly(ply_path)
        # pcd = fetchPlyAnchor(anchor_ply_path)

    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           cam_frustum_aabb=aabb)

    # pcd = scene_info.point_cloud
    # pts = np.asarray(pcd.points).astype(np.float64)

    # for cam in scene_info.train_cameras:
    #     K = build_K_from_caminfo(cam)
    #     # 你的 CameraInfo: R 是 C2W, T 是 W2C
    #     T_w2c = caminfo_w2c_train(cam, R_is_c2w=True, T_is_w2c=True)

    #     # 1) 可见性投影掩码
    #     mask = project_points_mask(pts, K, T_w2c, cam.width, cam.height)  # (H,W) 0/1 或 bool

    #     # 2) 原图
    #     img = np.array(cam.image)[:, :, ::-1]  # PIL->BGR（若用 cv2 保存）
    #     H, W = img.shape[:2]

    #     masked = img.copy()
    #     masked[mask == 0] = 0

    #     # 3) 准备 sky_mask（来自 cam.sky_mask 或你原始灰度的 sky_mask）
    #     # 如果 cam.sky_mask 已在 Camera 内（你的封装里有这个字段）：
    #     if  cam.sky_mask is not None:
    #         sky_bool = to_bool_sky_mask(cam.sky_mask, W, H)
    #     else:
    #         # 如果你想从磁盘重新读（最初的加载方式）
    #         # sky_path = image_path.replace("images", "sky_mask")
    #         # gray = cv2.imread(sky_path, cv2.IMREAD_GRAYSCALE)
    #         # sky_bool = to_bool_sky_mask(gray, W, H)
    #         raise RuntimeError("sky_mask not found on cam; please provide path or cam.sky_mask")

    #     # 4) 用天空掩码从原图覆盖回 masked（不管该像素是否为空，都覆盖）
    #     masked[sky_bool] = img[sky_bool]
    #     project_folder = path.replace("GaussianPro","pseudo")
    #     os.makedirs(project_folder,exist_ok=True)
    #     # 5) 保存
    #     cv2.imwrite(os.path.join(project_folder, f"{cam.image_name}.png"), masked)

    # TEST
    # 原色做回退
    # pcd = scene_info.point_cloud
    # pts = np.asarray(pcd.points).astype(np.float64)
    # pc_cols = np.asarray(pcd.colors)
    # if pc_cols.dtype != np.uint8:
    #     pc_cols = (np.clip(pc_cols,0,1)*255.0 + 0.5).astype(np.uint8)

    # # 只用“可见点”给颜色（避免被遮挡点乱染色）
    # baked = bake_colors_from_train_cams_visible(pts, scene_info.train_cameras, pc_cols)
    # project_dir = path.replace("euvs_data/trainset","test_projection")
    # os.makedirs(project_dir,exist_ok=True)
    # if "scene" in path:
    #     last = ".png"
    # else:
    #     last = ".jpg"
    # project_folder = path.replace("GaussianPro","pseudo_test_up")
    # os.makedirs(project_folder,exist_ok=True)
    # render_root = os.path.join(path.replace("euvs_data/trainset", "output"), "test")
    # render_dir = None
    # for ckpt in ["ours_45000", "ours_35000"]:
    #     path = os.path.join(render_root, ckpt, "renders")
    #     if os.path.isdir(path):
    #         render_dir = path
    #         break
    # if render_dir is None:
    #     raise FileNotFoundError("没有找到 ours_35000 或 ours_30000 的渲染目录")
    # # 测试偏移视角用 z-buffer 渲染（可把 radius_px 设 1~2）
    # for cam in scene_info.test_cameras:
    #     img,_,_ = render_points_zbuf_splat_fast(pts, baked, cam, radius_px=2)
    #     # img,_,_ = render_points_zbuf_splat_roi(pts, baked, cam)
    #     # cv2.imwrite(os.path.join(project_dir,cam.image_name+last), img[:, :, ::-1])
    #     # cv2.imwrite(f"/data/ljl/jya_repo/GaussianPro/test_projection/test_{cam.image_name}.png", img[:, :, ::-1])
    #     proj_img = img[:, :, ::-1]

    #     # --- 2) 读取对应渲染图 render_img ---
    #     render_path = os.path.join(render_dir, cam.image_name + last)
    #     if not os.path.exists(render_path):
    #         print(f"[WARN] 找不到渲染图: {render_path}")
    #         continue
    #     render_img = cv2.imread(render_path)

    #     # --- 3) 获取天空掩码 ---
    #     if cam.sky_mask is not None:
    #         H, W = render_img.shape[:2]
    #         sky_bool = to_bool_sky_mask(cam.sky_mask, W, H)  # bool mask, H×W
    #     else:
    #         print(f"[WARN] {cam.image_name} 没有 sky_mask，直接保存投影图")
    #         cv2.imwrite(os.path.join(project_folder, cam.image_name + ".png"), proj_img)
    #         continue

    #     # --- 4) 合并 ---
    #     merged = np.where(sky_bool[:, :, None], render_img, proj_img)

    #     # --- 5) 保存 ---
    #     out_path = os.path.join(project_folder, cam.image_name + ".png")
    #     cv2.imwrite(out_path, merged)
    #     print(f"[OK] 保存合并图像: {out_path}")

    # TRAIN
    # pcd = scene_info.point_cloud
    # pts = np.asarray(pcd.points).astype(np.float64)
    # pc_cols = np.asarray(pcd.colors)
    # if pc_cols.dtype != np.uint8:
    #     pc_cols = (np.clip(pc_cols, 0, 1) * 255.0 + 0.5).astype(np.uint8)

    # # 可见性烘焙（你已有）
    # baked = bake_colors_from_train_cams_visible(pts, scene_info.train_cameras, pc_cols)

    # final_path = path.replace("euvs_data/trainset","final_render")

    # project_dir = path.replace("euvs_data/trainset","test_projection")
    # project_dir = path.replace("euvs_data/trainset","train_dense_project_2")
    # os.makedirs(project_dir,exist_ok=True)
    # #输出目录
    # out_dir = os.path.join(final_path, "images")
    # os.makedirs(out_dir, exist_ok=True)
    # if "scene" in path:
    #     last = ".png"
    # else:
    #     last = ".jpg"
    # GS_path = path.replace("/data/ljl/jya_repo/GaussianPro/euvs_data/trainset", "/data/ljl/jya_repo/submission/second_version")
    # # 批处理每个测试视角
    # project_folder = path.replace("GaussianPro","pseudo")
    # os.makedirs(project_folder,exist_ok=True)
    # render_root = os.path.join(path.replace("euvs_data/trainset", "output"), "train")
    # render_dir = None
    # for ckpt in ["ours_40000", "ours_35000", "ours_30000"]:
    #     path = os.path.join(render_root, ckpt, "renders")
    #     if os.path.isdir(path):
    #         render_dir = path
    #         break
    # if render_dir is None:
    #     raise FileNotFoundError("没有找到 ours_35000 或 ours_30000 的渲染目录")

    # # os.makedirs(project_dir, exist_ok=True)

    # for cam in scene_info.train_cameras:
    #     # --- 1) 生成投影图 proj_img ---
    #     proj_img_rgb, wrote_mask, _ = render_points_zbuf_splat_fast(
    #         pts, baked, cam, radius_px=2
    #     )
    #     proj_img = proj_img_rgb[:, :, ::-1]  # 转回 BGR，方便与 cv2.imread 对齐

    #     # --- 2) 读取对应渲染图 render_img ---
    #     render_path = os.path.join(render_dir, cam.image_name + last)
    #     if not os.path.exists(render_path):
    #         print(f"[WARN] 找不到渲染图: {render_path}")
    #         continue
    #     render_img = cv2.imread(render_path)

    #     # --- 3) 获取天空掩码 ---
    #     if cam.sky_mask is not None:
    #         H, W = render_img.shape[:2]
    #         sky_bool = to_bool_sky_mask(cam.sky_mask, W, H)  # bool mask, H×W
    #     else:
    #         print(f"[WARN] {cam.image_name} 没有 sky_mask，直接保存投影图")
    #         cv2.imwrite(os.path.join(project_folder, cam.image_name + ".png"), proj_img)
    #         continue
    #     # --- 4) 合并 ---
    #     merged = np.where(sky_bool[:, :, None], render_img, proj_img)
    #     # --- 5) 保存 ---
    #     out_path = os.path.join(project_folder, cam.image_name + ".png")
    #     cv2.imwrite(out_path, merged)
    #     print(f"[OK] 保存合并图像: {out_path}")

    # cv2.imwrite(os.path.join(project_dir,cam.image_name+last), proj_img_rgb[:, :, ::-1])

    # 2) 读取 3DGS 渲染图
    # render_path = os.path.join(GS_path, "images", cam.image_name+last)
    # if not os.path.isfile(render_path):
    #     print(f"[跳过] 找不到渲染图：{render_path}")
    #     continue
    # gauss_bgr = cv2.imread(render_path, cv2.IMREAD_COLOR)  # BGR, HxWx3, uint8

    # # 3) 分辨率对齐（必要时 resize 投影和掩码）
    # Ht, Wt = gauss_bgr.shape[:2]
    # Hp, Wp = proj_img_bgr.shape[:2]
    # if (Ht != Hp) or (Wt != Wp):
    #     print("二者形状不一致!")
    #     proj_img_bgr = cv2.resize(proj_img_bgr, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
    #     wrote_mask = cv2.resize(wrote_mask.astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST).astype(bool)
    # # 生成“只允许下方 2/3 更新”的行掩码
    # cut = Ht // 3
    # keep_top_frac = 1/2.0     # 保留上方 2/5 的 3DGS
    # cut = int(round(Ht * keep_top_frac))
    # row_mask = np.zeros((Ht, Wt), dtype=bool)
    # row_mask[cut:, :] = True
    # final_mask = wrote_mask & row_mask
    # # 4) 合并：用掩码把投影覆盖到 3DGS
    # merged = gauss_bgr.copy()
    # merged[final_mask] = proj_img_bgr[final_mask]

    # # 5) 保存（保留 3DGS 的命名，另存 merged）
    # out_path = os.path.join(out_dir, cam.image_name+last)
    # cv2.imwrite(out_path, merged)
    # print(f"[OK] 写入：{out_path}")
    if "level1" or "level2" in path:
        pcd_down = downsample_pointcloud(scene_info.point_cloud, ratio=0.5)
        scene_info = SceneInfo(point_cloud=pcd_down,  # 重新构建 SceneInfo
                               train_cameras=scene_info.train_cameras,
                               test_cameras=scene_info.test_cameras,
                               nerf_normalization=scene_info.nerf_normalization,
                               ply_path=scene_info.ply_path, cam_frustum_aabb=scene_info.cam_frustum_aabb)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", is_train=True):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            sky_mask = np.ones_like(image)[:, :, 0].astype(np.uint8)

            if is_train:
                normal_path = image_path.replace("train", "normals")[:-4] + ".npy"
                normal = np.load(normal_path).astype(np.float32)
                normal = (normal - 0.5) * 2.0
                # normal[2, :, :] *= -1
            else:
                normal = np.zeros_like(image).transpose(2, 0, 1)

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1],
                                        K=np.array([1, 2, 3, 4]), sky_mask=sky_mask, normal=normal))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension,
                                               is_train=False)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}
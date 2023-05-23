import torch
import numpy as np


def rand_nm_bd(batch_size, variable_dim, region_a, init_l, init_r, to_torch=True, to_float=True, to_cuda=False,
               gpu_no=0, use_grad=True):
    # 生成诺依曼边界
    region_a = float(region_a)
    init_l = float(init_l)
    init_r = float(init_r)
    assert (int(variable_dim) == 2)
    x_left_bd = (init_r - init_l) * np.random.random([batch_size, 2]) + init_l
    x_left_bd[:, 0] = region_a

    if to_float:
        x_left_bd = x_left_bd.astype(np.float32)

    if to_torch:
        x_left_bd = torch.from_numpy(x_left_bd)
        if to_cuda:
            x_left_bd = x_left_bd(device='cuda:' + str(gpu_no))
        x_left_bd.requires_grad = use_grad

    return x_left_bd


def rand_bd_inside(batch_size, region_a, region_b, init_l, init_r, to_torch=True, to_float=True,
                   to_cuda=False, gpu_no=0, use_grad=True):
    # 生成二维矩阵的内部点
    region_a = float(region_a)
    region_b = float(region_b)
    init_l = float(init_l)
    init_r = float(init_r)
    inside_points = np.random.random([batch_size, 2])
    inside_points[:, 0] = region_a + inside_points[:, 0] * (region_b - region_a)
    inside_points[:, 1] = init_l + inside_points[:, 1] * (init_r - init_l)

    if to_float:
        inside_points = inside_points.astype(np.float32)

    if to_torch:
        inside_points = torch.from_numpy(inside_points)
        if to_cuda:
            inside_points = inside_points.cuda(device='cuda:' + str(gpu_no))
        inside_points.requires_grad = use_grad

    return inside_points


def rand_bd_inflow(batch_size, variable_dim, region_a, region_b, init_l, init_r, to_torch=True, to_float=True,
                   to_cuda=False, gpu_no=0,
                   use_grad=True):
    # 生成Dirhlet 边界的点
    region_a = float(region_a)
    region_b = float(region_b)
    init_l = float(init_l)
    init_r = float(init_r)
    assert (int(variable_dim) == 2)
    bd1 = int(batch_size * np.random.random(1))
    bd2 = batch_size - bd1
    x_right_bd = (init_r - init_l) * np.random.random([bd1, 2]) + init_l
    y_bottom_bd = (region_b - region_a) * np.random.random([bd2, 2]) + region_a
    x_right_bd[:, 0] = region_b
    y_bottom_bd[:, 1] = init_l
    inflow_boundary_points = np.concatenate([x_right_bd, y_bottom_bd], 0)
    if to_float:
        inflow_boundary_points = inflow_boundary_points.astype(np.float32)

    if to_torch:
        inflow_boundary_points = torch.from_numpy(inflow_boundary_points)
        if to_cuda:
            inflow_boundary_points = inflow_boundary_points(device='cuda:' + str(gpu_no))
        inflow_boundary_points.requires_grad = use_grad

    return inflow_boundary_points

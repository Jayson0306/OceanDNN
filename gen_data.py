import torch
import numpy as np


def rand_bd_inside(batch_size, variable_dim, region_a, region_b, init_l, init_r, to_torch=True, to_float=True,
                   to_cuda=False, gpu_no=0,use_grad=True):
    # 生成内部点
    region_a = float(region_a)
    region_b = float(region_b)
    init_l = float(init_l)
    init_r = float(init_r)
    assert (int(variable_dim) == 2)
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


def rand_bd(batch_size, variable_dim, region_a, region_b, init_l, init_r, to_torch=True, to_float=True, to_cuda=False,
            gpu_no=0,use_grad=True):
    # 全边界在Hard_PINN中没什么用
    region_a = float(region_a)
    region_b = float(region_b)
    init_l = float(init_l)
    init_r = float(init_r)
    assert (int(variable_dim) == 2)
    bd1 = int(batch_size * np.random.random(1))
    res = batch_size - bd1
    bd2 = int(res * np.random.random(1))
    res = res - bd2
    bd3 = int(res * np.random.random(1))
    bd4 = res - bd3
    x_left_bd = (init_r - init_l) * np.random.random([bd1, 2]) + init_l  # 浮点数都是从0-1中随机。
    x_right_bd = (init_r - init_l) * np.random.random([bd2, 2]) + init_l
    y_bottom_bd = (region_b - region_a) * np.random.random([bd3, 2]) + region_a
    y_top_bd = (region_b - region_a) * np.random.random([bd3, 2]) + region_a

    x_left_bd[:, 0] = region_a
    x_right_bd[:, 0] = region_b
    y_bottom_bd[:, 1] = init_l
    y_top_bd[:, 1] = init_r

    if to_float:
        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)
        y_bottom_bd = y_bottom_bd.astype(np.float32)
        y_top_bd = y_top_bd.astype(np.float32)
    if to_torch:
        x_left_bd = torch.from_numpy(x_left_bd)
        x_right_bd = torch.from_numpy(x_right_bd)
        y_bottom_bd = torch.from_numpy(y_bottom_bd)
        y_top_bd = torch.from_numpy(y_top_bd)
        if to_cuda:
            x_left_bd = x_left_bd.cuda(device='cuda:' + str(gpu_no))
            x_right_bd = x_right_bd.cuda(device='cuda:' + str(gpu_no))
            y_bottom_bd = y_bottom_bd.cuda(device='cuda:' + str(gpu_no))
            y_top_bd = y_top_bd.cuda(device='cuda:' + str(gpu_no))

        x_left_bd.requires_grad = use_grad
        x_right_bd.requires_grad = use_grad
        y_bottom_bd.requires_grad = use_grad
        y_top_bd.requires_grad = use_grad

    return x_left_bd, x_right_bd, y_bottom_bd, y_top_bd


def rand_bd_inflow(batch_size, variable_dim, region_a, region_b, init_l, init_r, to_torch=True, to_float=True,
                   to_cuda=False, gpu_no=0,use_grad=True):
    # 有Dirichlet的边界 叫做INFLOW边界
    region_a = float(region_a)
    region_b = float(region_b)
    init_l = float(init_l)
    init_r = float(init_r)
    assert (int(variable_dim) == 2)
    # bd1 = int(batch_size * np.random.random(1))
    # bd2 = batch_size - bd1
    x_right_bd = (init_r - init_l) * np.random.random([batch_size, 2]) + init_l
    y_bottom_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
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


def rand_nm_bd(batch_size, variable_dim, region_a, init_l, init_r, to_torch=True, to_float=True, to_cuda=False,
               gpu_no=0,
               use_grad=True):
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


def rand_bd_inflow_EX6(batch_size, region_a, region_b, init_l, init_r, to_torch=True, to_float=True, to_cuda=False,
                       gpu_no=0,
                       use_grad=True):
    # 生成Dirhlet 边界的点
    region_a = float(region_a)
    region_b = float(region_b)
    init_l = float(init_l)
    init_r = float(init_r)
    y_bottom_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
    y_bottom_bd[:, 1] = init_l
    inflow_boundary_points = y_bottom_bd
    if to_float:
        inflow_boundary_points = inflow_boundary_points.astype(np.float32)

    if to_torch:
        inflow_boundary_points = torch.from_numpy(inflow_boundary_points)
        if to_cuda:
            inflow_boundary_points = inflow_boundary_points(device='cuda:' + str(gpu_no))
        inflow_boundary_points.requires_grad = use_grad

    return inflow_boundary_points

def rand_bd_inside_EX6(batch_size, region_a, region_b, init_l, init_r, to_torch=True, to_float=True,
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
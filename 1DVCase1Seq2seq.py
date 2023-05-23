"""
@author: LXA
 Date: 2021 年 11 月 11 日
 Modifying on 2022 年 9月 2 日 ~~~~ 2022 年 9月 12 日
"""
import os
import sys
import torch
import torch.nn as tn
import numpy as np
import matplotlib
import platform
import shutil
import time
import DNN_base
import DNN_tools
import dataUtilizer2torch

import saveData
import plotData
import DNN_Log_Print
from Load_data2Mat import *
from scipy.special import erfc
import torch.nn.functional as tnf


def erfc1(x, GPUno=0, use_gpu=True):
    tmp = x.cpu().detach().numpy()
    z = erfc(tmp)
    z = torch.from_numpy(z)
    if use_gpu:
        z = z.cuda(device='cuda:' + str(GPUno))
    return z


class MscaleDNN(tn.Module):
    def __init__(self, input_dim=2, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='relu',
                 name2actHidden='relu', name2actOut='linear', opt2regular_WB='L2', type2numeric='float32',
                 factor2freq=None, sFourier=1.0, use_gpu=False, No2GPU=0, repeat_highFreq=True):
        super(MscaleDNN, self).__init__()
        if 'DNN' == str.upper(Model_name):
            self.DNN = DNN_base.Pure_DenseNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'SCALE_DNN' == str.upper(Model_name) or 'DNN_SCALE' == str.upper(Model_name):
            self.DNN = DNN_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'FOURIER_DNN' == str.upper(Model_name) or 'DNN_FOURIERBASE' == str.upper(Model_name):
            self.DNN = DNN_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                repeat_Highfreq=repeat_highFreq, to_gpu=use_gpu, gpu_no=No2GPU)
        elif 'FOURIER_SUBDNN' == str.upper(Model_name) or 'SUBDNN_FOURIERBASE' == str.upper(Model_name):
            self.DNN = DNN_base.Fourier_SubNets3D(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name,
                actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric,
                num2subnets=len(factor2freq), to_gpu=use_gpu, gpu_no=No2GPU)
        elif str.upper(Model_name) == 'MULTI_4FF_DNN':
            self.DNN = DNN_base.Multi_4FF_DNN(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer,
                                              name2Model=Model_name,
                                              actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut,
                                              scope2W='Weight', scope2B='Bias', repeat_Highfreq=True,
                                              type2float='float32',
                                              to_gpu=use_gpu, gpu_no=No2GPU, sigma1=1.0, sigma2=5.0, sigma3=10.0,
                                              sigma4=20.0,
                                              trainable2sigma=False)
        elif str.upper(Model_name) == 'MULTI_3FF_DNN':
            self.DNN = DNN_base.Multi_3FF_DNN(indim=input_dim, outdim=out_dim, hidden_units=hidden_layer,
                                              name2Model=Model_name,
                                              actName2in=name2actIn, actName=name2actHidden, actName2out=name2actOut,
                                              scope2W='Weight', scope2B='Bias', repeat_Highfreq=True,
                                              type2float='float32',
                                              to_gpu=use_gpu, gpu_no=No2GPU, sigma1=1.0, sigma2=10.0, sigma3=20.0,
                                              trainable2sigma=False)

        self.use_gpu = use_gpu
        if use_gpu:
            self.opt2device = 'cuda:' + str(No2GPU)
        else:
            self.opt2device = 'cpu'
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_layer = hidden_layer
        self.Model_name = Model_name
        self.name2actIn = name2actIn
        self.name2actHidden = name2actHidden
        self.name2actOut = name2actOut
        self.factor2freq = factor2freq
        self.sFourier = sFourier
        self.opt2regular_WB = opt2regular_WB

        if type2numeric == 'float32':
            self.float_type = torch.float32
        elif type2numeric == 'float64':
            self.float_type = torch.float64
        elif type2numeric == 'float16':
            self.float_type = torch.float16
        self.mat2X = torch.tensor([[1, 0]], dtype=self.float_type, device=self.opt2device)  # 1 行 2 列
        self.mat2U = torch.tensor([[0, 1]], dtype=self.float_type, device=self.opt2device)  # 1 行 2 列
        self.mat2T = torch.tensor([[0, 1]], dtype=self.float_type, device=self.opt2device)  # 1 行 3 列

    def loss_1DV(self, X_in=None, T_in=None, loss_type='l2_loss', scale2lncosh=0.5,
                 ws=0.001, ds=0.002):
        '''
        Args:
            X_in: 输入的内部深度（x or z)
            T_in: 输入的内部时间 (t)
            loss_type:
            scale2lncosh:
            ws: dx 的常系数
            ds: dxx 的常系数
        Returns:

        '''
        # 判断输入的内部点不为None
        assert (X_in is not None)
        assert (T_in is not None)

        # 判断 X_in 的形状
        shape2X = X_in.shape
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        # 判断 T_in 的形状
        shape2T = T_in.shape
        lenght2T_shape = len(shape2T)
        assert (lenght2T_shape == 2)
        assert (shape2T[-1] == 1)

        XT = torch.mul(X_in, self.mat2X) + torch.mul(T_in, self.mat2T)

        # 计算内部点损失 dx dxx
        UNN = self.DNN(XT, scale=self.factor2freq, sFourier=self.sFourier)
        grad2UNN = torch.autograd.grad(UNN, X_in, grad_outputs=torch.ones_like(X_in),
                                       create_graph=True, retain_graph=True)
        dUNN2x = grad2UNN[0]
        dUNN2x = -tnf.relu(-grad2UNN[0])#这样不行的话就考虑用绝对值 我们希望gradUNN2x 是负的 所以把为正的弄为0
        dUNNxx = torch.autograd.grad(dUNN2x, X_in, grad_outputs=torch.ones_like(X_in),
                                     create_graph=True, retain_graph=True)[0]
        # dt
        dUNN2t = torch.autograd.grad(UNN, T_in, grad_outputs=torch.ones_like(T_in),
                                     create_graph=True, retain_graph=True)[0]

        # loss
        res = dUNN2t + ws * dUNN2x - ds * dUNNxx
        if str.lower(loss_type) == 'l2_loss':
            loss_it = torch.mean(torch.square(res))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_it = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * res)))
        return UNN, loss_it

    def loss2bd_neumann(self, X_bd=None, T_bd=None, Ubd_exact=None, if_lambda2Ubd=True,
                        loss_type='l2_loss', scale2lncosh=0.5):
        assert (X_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = X_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 1)

        XT_bd = torch.mul(X_bd, self.mat2X) + torch.mul(T_bd, self.mat2T)

        if if_lambda2Ubd:
            U_bd = Ubd_exact(X_bd, T_bd)
        else:
            U_bd = Ubd_exact

        # 用神经网络求 dUNN2x 以及 UNN
        UNN = self.DNN(XT_bd, scale=self.factor2freq, sFourier=self.sFourier)
        grad2UNN = torch.autograd.grad(UNN, X_bd, grad_outputs=torch.ones_like(X_bd), create_graph=True,
                                       retain_graph=True, allow_unused=True)
        dUNN2x = grad2UNN[0]

        diff_bd = dUNN2x - U_bd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(diff_bd))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_bd = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * diff_bd)))
        return loss_bd

    def loss2bd_dirichlet(self, X_bd=None, T_bd=None, Ubd_exact=None, if_lambda2Ubd=True,
                          loss_type='l2_loss', scale2lncosh=0.5):
        assert (X_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = X_bd.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 1)

        XT_bd = torch.mul(X_bd, self.mat2X) + torch.mul(T_bd, self.mat2T)

        if if_lambda2Ubd:
            U_bd = Ubd_exact(X_bd, T_bd)
        else:
            U_bd = Ubd_exact

        UNN = self.DNN(XT_bd, scale=self.factor2freq, sFourier=self.sFourier)

        diff_bd = UNN - U_bd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(diff_bd))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_bd = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * diff_bd)))
        return loss_bd

    def loss2init(self, X_init=None, T_init=None, Uinit_exact=None, if_lambda2Uinit=True,
                  loss_type='l2_loss', scale2lncosh=0.5):
        assert (X_init is not None)
        assert (T_init is not None)
        assert (Uinit_exact is not None)

        shape2XY = X_init.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 1)

        XT_init = torch.mul(X_init, self.mat2X) + torch.mul(T_init, self.mat2T)

        if if_lambda2Uinit:
            U_init = Uinit_exact(X_init, T_init)
        else:
            U_init = Uinit_exact

        UNN = self.DNN(XT_init, scale=self.factor2freq, sFourier=self.sFourier)

        diff_init = UNN - U_init
        if str.lower(loss_type) == 'l2_loss':
            loss_init = torch.mean(torch.square(diff_init))
        elif str.lower(loss_type) == 'lncosh_loss':
            loss_init = (1 / scale2lncosh) * torch.mean(torch.log(torch.cosh(scale2lncosh * diff_init)))
        return loss_init

    def loss2bd_Neumann(self, X_bd=None, T_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss'):
        # 判断输入的内部点不为None
        assert (X_bd is not None)
        assert (T_bd is not None)

        # 判断 X_in 的形状
        shape2X = X_bd.shape
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        # 判断 T_in 的形状
        shape2T = T_bd.shape
        lenght2T_shape = len(shape2T)
        assert (lenght2T_shape == 2)
        assert (shape2T[-1] == 1)

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd, T_bd)
        else:
            Ubd = Ubd_exact

        XT_bd = torch.mul(X_bd, self.mat2X) + torch.mul(T_bd, self.mat2T)

        # 计算内部点损失 dx dxx
        UNN = self.DNN(XT_bd, scale=self.factor2freq, sFourier=self.sFourier)
        grad2UNN = torch.autograd.grad(UNN, X_bd, grad_outputs=torch.ones_like(X_bd),
                                       create_graph=True, retain_graph=True)
        dUNN2x = grad2UNN[0]

        res = dUNN2x - Ubd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(res))
        return loss_bd

    def loss2bd_Neumann_A4(self, X_bd=None, T_bd=None, Ubd_exact=None, if_lambda2Ubd=True, loss_type='l2_loss',ws=0.001, ds = 0.0002):
        # 判断输入的内部点不为None
        assert (X_bd is not None)
        assert (T_bd is not None)

        # 判断 X_in 的形状
        shape2X = X_bd.shape
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)
        assert (shape2X[-1] == 1)

        # 判断 T_in 的形状
        shape2T = T_bd.shape
        lenght2T_shape = len(shape2T)
        assert (lenght2T_shape == 2)
        assert (shape2T[-1] == 1)

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd, T_bd)
        else:
            Ubd = Ubd_exact

        XT_bd = torch.mul(X_bd, self.mat2X) + torch.mul(T_bd, self.mat2T)

        # 计算内部点损失 dx dxx
        UNN = self.DNN(XT_bd, scale=self.factor2freq, sFourier=self.sFourier)
        grad2UNN = torch.autograd.grad(UNN, X_bd, grad_outputs=torch.ones_like(X_bd),
                                       create_graph=True, retain_graph=True)
        dUNN2x = grad2UNN[0]

        res = -ds * dUNN2x + ws * UNN - Ubd
        if str.lower(loss_type) == 'l2_loss':
            loss_bd = torch.mean(torch.square(res))
        return loss_bd

    def get_regularSum2WB(self):
        sum2WB = self.DNN.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def evalue_MscaleDNN(self, XY_points=None):
        assert (XY_points is not None)
        shape2XY = XY_points.shape
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        UNN = self.DNN(XY_points, scale=self.factor2freq, sFourier=self.sFourier)
        return UNN


def solve_Multiscale_PDE(R):
    log_out_path = R['FolderName']  # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)  # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Log_Print.dictionary_out2file(R, log_fileout)

    # 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']
    batchsize_init = R['batch_size2init']

    bd_penalty_init = R['init_boundary_penalty']  # Regularization parameter for boundary conditions
    init_penalty_init = R['init_penalty2init']
    penalty2WB = R['penalty2weight_biases']  # Regularization parameter for weights and biases
    learning_rate = R['learning_rate']

    input_dim = R['input_dim']
    equation = 6

    # equation 1
    # 双诺依曼边界
    if equation == 1:
        # equation EX1
        # Analytical Solutions of One Dimensional Convective Dispersive Solute Transport Equations
        region_l = 0.0
        region_r = 10.0
        init_time = 0.0
        end_time = 5
        ds = 0.001
        ws = 0.01
        Ci = 0
        C0 = 0.1
        temp1 = lambda x, t: torch.div(x - ws * t, torch.sqrt(4 * ds * t))
        temp2 = lambda x, t: torch.div(x + ws * t, torch.sqrt(4 * ds * t))
        temp3 = lambda x, t: 0.5 * erfc1(temp1(x, t)) + 0.5 * torch.exp(ws * x / ds) * erfc1(temp2(x, t))
        u_init = lambda x, t: torch.ones_like(x) * Ci
        u_left = lambda x, t: torch.ones_like(t) * C0
        u_right = lambda x, t: torch.zeros_like(t)
        u_true = lambda x, t: Ci + (Ci - C0) * temp3(x, t)
    if equation == 2:
        # equation EX2
        # Derivation~....
        region_l = 0.0
        region_r = 2
        init_time = 0.0
        end_time = 50
        ds = 0.015
        ws = 0.03
        Ci = 0
        C0 = 0.001
        temp1 = lambda x, t: torch.div(x - ws * t, torch.sqrt(4 * ds * t))
        temp2 = lambda x, t: torch.div(x + ws * t, torch.sqrt(4 * ds * t))
        temp3 = lambda x, t: 0.5 * erfc1(temp1(x, t)) + 0.5 * torch.exp(ws * x / ds) * erfc1(temp2(x, t))
        u_init = lambda x, t: torch.ones_like(x) * Ci
        u_left = lambda x, t: torch.ones_like(t) * C0
        u_right = lambda x, t: torch.zeros_like(t)
        u_true = lambda x, t: Ci+(C0-Ci)*temp3(x, t)

    if equation == 3:
        # equation EX3  -- A3
        # Analytical Solutions of One Dimensional Convective Dispersive Solute Transport Equations
        region_l = 0.0
        region_r = 1
        init_time = 0.0
        end_time = 10
        ds = 0.0002
        ws = 0.0001
        Ci = 0
        C0 = 8
        pi = np.pi
        # A3的第一项
        temp1 = lambda x, t: torch.div(x - ws * t, torch.sqrt(4 * ds * t))
        temp2 = lambda x, t: torch.div(x + ws * t, torch.sqrt(4 * ds * t))
        temp3 = lambda x, t: 0.5 * erfc1(temp1(x, t)) + 0.5 * torch.exp(ws * x / ds) * erfc1(temp2(x, t))

        # A3的第二项
        temp4 = lambda x, t: 2 + (ws/ds) * (2*region_r-x) + (ws*ws/ds) * t
        temp5 = lambda x, t: np.exp(ws*region_r/ds) * erfc1(torch.div(2*region_r-x+ws*t,torch.sqrt(4*ds*t)))
        temp6 = lambda x, t: 0.5 * torch.mul(temp4(x, t), temp5(x, t))

        # A3的第三项
        temp7 = lambda x, t: ws * torch.sqrt(t/(pi*ds))
        temp8 = lambda x, t: torch.mul(1/(4*ds*t), torch.square(2*region_r- x + ws * t))
        temp9 = lambda x, t: torch.mul(temp7(x, t), torch.exp(ws*region_r/ds - temp8(x, t)))
        Axt = lambda x, t: temp3(x, t) + temp6(x, t) + temp9(x, t)

        u_init = lambda x, t: torch.ones_like(x) * Ci
        u_left = lambda x, t: torch.ones_like(t) * C0
        u_right = lambda x, t: torch.zeros_like(t)
        u_true = lambda x, t: Ci + (C0 - Ci) * Axt(x, t)

    if equation == 4:
        # equation EX4  -- A4
        # Analytical Solutions of One Dimensional Convective Dispersive Solute Transport Equations
        region_l = 0.0
        region_r = 1
        init_time = 0.0
        end_time = 1
        ds = 0.015
        ws = 0.03
        Ci = 0.01
        C0 = 0.1
        pi = np.pi
        # A4的第一项
        temp1 = lambda x, t: torch.div(x - ws * t, torch.sqrt(4 * ds * t))
        temp2 = lambda x, t: torch.div(x + ws * t, torch.sqrt(4 * ds * t))
        temp3 = lambda x, t: 0.5 * erfc1(temp1(x, t))

        # A4的第二项
        temp4 = lambda x, t: torch.sqrt(ws*ws * t/ds/pi)
        temp5 = lambda x, t: torch.mul(temp4(x, t), torch.exp(torch.square(temp1(x, t))))

        # A4的第三项
        temp6 = lambda x, t: 1 + ws * x /ds + ws*ws * t/ds
        temp7 = lambda x, t: torch.exp(ws*x/ds)
        temp8 = lambda x, t: 0.5 * torch.mul(torch.mul(temp6(x, t),temp7(x, t)), erfc1(temp2(x, t)))

        # A4的第四项
        temp9 = lambda x, t: 1 + (2*region_r-x+ws*t)*ws/ds/4
        temp91 = lambda x, t: torch.square(2*region_r- x + ws * t)
        temp10 = lambda x, t: ws*region_r/ds - torch.mul(1/(4*ds*t), temp91(x, t))
        temp11 = lambda x, t: 2 * torch.mul(torch.mul(temp4(x, t), temp9(x,t)), torch.exp(temp10(x, t)))

        # A4的第五项
        # v/D(2L-x + 3vt/2R + v/4D(2L-x+vt/R)^2)exp（vL/D）erfc(R(2L-x)+vt/sqrt(4DRt)))
        temp12 = lambda x, t: 2*region_r-x + 1.5 * ws *t + ws * temp91(x, t)/4/ds
        temp13 = lambda x, t: erfc1(torch.div(temp91(x, t), torch.sqrt(4*ds*t)))
        temp14 = lambda x, t: torch.mul(temp12(x, t), temp13(x, t)) * np.exp(ws*region_r/ds) * ws / ds
        Axt = lambda x, t: temp3(x, t) + temp5(x, t) + temp8(x, t) + temp11(x,t) +temp14(x, t)


        u_init = lambda x, t: torch.ones_like(x) * Ci
        u_left = lambda x, t: torch.ones_like(t) * C0 * ws
        u_right = lambda x, t: torch.zeros_like(t)
        u_true = lambda x, t: Ci + (C0-Ci) * Axt(x, t)

    if equation == 5:
        # equation EX4  -- A4
        # Performance Analysis of the Neural Network
        # Solution of Advection-Diffusion-Reaction Problem
        region_l = 0.0
        region_r = 1
        init_time = 0.0
        end_time = 1
        ds = 0.1
        ws = 0.1
        r = 2
        Ci = 0.001
        C0 = 0.001
        pi = np.pi
        # A4的第一项
        temp1 = lambda x, t: torch.div(x - ws * t, torch.sqrt(4 * ds * t))
        temp2 = lambda x, t: torch.div(x + ws * t, torch.sqrt(4 * ds * t))
        temp3 = lambda x, t: 0.5 * erfc1(temp1(x, t))

        # A4的第二项
        temp4 = lambda x, t: torch.sqrt(ws*ws * t/ds/pi)
        temp5 = lambda x, t: torch.mul(temp4(x, t), torch.exp(torch.square(temp1(x, t))))

        # A4的第三项
        temp6 = lambda x, t: 1 + ws * x /ds + ws*ws * t/ds
        temp7 = lambda x, t: torch.exp(ws*x/ds)
        temp8 = lambda x, t: 0.5 * torch.mul(torch.mul(temp6(x, t),temp7(x, t)), erfc1(temp2(x, t)))

        # A4的第四项
        temp9 = lambda x, t: 1 + (2*region_r-x+ws*t)*ws/ds/4
        temp91 = lambda x, t: torch.square(2*region_r- x + ws * t)
        temp10 = lambda x, t: ws*region_r/ds - torch.mul(1/(4*ds*t), temp91(x, t))
        temp11 = lambda x, t: 2 * torch.mul(torch.mul(temp4(x, t), temp9(x,t)), torch.exp(temp10(x, t)))

        # A4的第五项
        # v/D(2L-x + 3vt/2R + v/4D(2L-x+vt/R)^2)exp（vL/D）erfc(R(2L-x)+vt/sqrt(4DRt)))
        temp12 = lambda x, t: 2*region_r-x + 1.5 * ws *t + ws * temp91(x, t)/4/ds
        temp13 = lambda x, t: erfc1(torch.div(temp91(x, t), torch.sqrt(4*ds*t)))
        temp14 = lambda x, t: torch.mul(temp12(x, t), temp13(x, t)) * np.exp(ws*region_r/ds) * ws / ds
        Axt = lambda x, t: temp3(x, t) + temp5(x, t) + temp8(x, t) + temp11(x,t) +temp14(x, t)


        u_init = lambda x, t: torch.exp(-4*x)
        u_left = lambda x, t: torch.zeros_like(t)
        u_right = lambda x, t: torch.zeros_like(t)
        f_side = lambda x: r * torch.mul(x, 1-x)
        u_true = lambda x, t: torch.zeros_like(x)

    if equation == 6:
        #case 1
        region_l = 0.0
        region_r = 5
        init_time = 0.0
        end_time = 3600
        ds = 0.0002
        ws = 0.0001
        # ds = 0.2
        # ws = 0.1
        Ci = 0
        C0 = 1
        pi = np.pi
        # A3的第一项
        temp1 = lambda x, t: torch.div(x - ws * t, torch.sqrt(4 * ds * t))
        temp2 = lambda x, t: torch.div(x + ws * t, torch.sqrt(4 * ds * t))
        #公式16
        temp10 = lambda x, t: torch.exp(- ws * x /ds)
        temp11 = lambda x, t: 0.5 * C0 * (erfc1(temp2(x, t), GPUno=R['gpuNo'], use_gpu=R['use_gpu']) +
                                         torch.mul(temp10(x, t), erfc1(temp1(x, t), GPUno=R['gpuNo'], use_gpu=R['use_gpu'])))

        u_init = lambda x, t: torch.ones_like(x) * Ci
        u_left = lambda x, t: torch.ones_like(t) * C0
        u_right = lambda x, t: torch.zeros_like(t)
        # u_true = lambda x, t: Ci + (C0 - Ci) * Axt(x, t)
        u_true = lambda x, t: temp11(x, t)

    Seq2seq = True
    Seq2seq2x = False

    model = MscaleDNN(input_dim=R['input_dim'] + 1, out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                      Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                      name2actOut=R['name2act_out'], opt2regular_WB='L0', repeat_highFreq=R['repeat_highFreq'],
                      type2numeric='float32', factor2freq=R['freq'], use_gpu=R['use_gpu'], No2GPU=R['gpuNo'])

    if True == R['use_gpu']:
        model = model.cuda(device='cuda:' + str(R['gpuNo']))

    params2Net = model.DNN.parameters()

    # 定义优化方法，并给定初始学习率
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr)                     # SGD
    # optimizer = torch.optim.SGD(params2Net, lr=init_lr, momentum=0.8)       # momentum
    # optimizer = torch.optim.RMSprop(params2Net, lr=init_lr, alpha=0.95)     # RMSProp
    optimizer = torch.optim.Adam(params2Net, lr=learning_rate)  # Adam

    # 定义更新学习率的方法
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.99)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.975)


    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []
    loss_init_all = []

    if R['testData_model'] == 'random_generate':
        # 生成测试数据，用于测试训练后的网络
        test_bach_size = 100
        size2test = 100
        # test_bach_size = 6400
        # size2test = 80
        # test_bach_size = 10000
        # size2test = 100

        # ---------------------------------------------------------------------------------#
        test_xy_bach_x = np.linspace(region_l, region_r, test_bach_size).reshape(-1, 1)
        test_xy_bach_y = np.linspace(init_time, end_time, test_bach_size).reshape(-1, 1)
        x_repeat = np.repeat(test_xy_bach_x, test_bach_size).reshape(-1, 1)
        t2 = list(test_xy_bach_y)
        t1 = list(test_xy_bach_y)
        for i in range(test_bach_size - 1):
            t2.extend(t1)
        t_repeat = np.array(t2)
        test_xy_bach = np.concatenate([x_repeat, t_repeat], -1)

        # ------------------------------------------#
    # elif R['testData_model'] == 'load_dat':
    #     mat_data_path = 'D:/Matlab/bin'
    #     x_test, t_test = get_randomData2mat(dim=2, data_path=mat_data_path)
    #     x_test = x_test.reshape(-1, 1)
    #     t_test = t_test.reshape(-1, 1)
    #     test_xy_bach = np.concatenate([x_test, t_test], -1)

    saveData.save_testData_or_solus2mat(test_xy_bach, dataName='testXY', outPath=R['FolderName'])

    test_xy_bach = test_xy_bach.astype(np.float32)
    test_xy_torch = torch.from_numpy(test_xy_bach)

    # 生成左右边界
    xl_bd_batch = np.ones(shape=[batchsize_bd, 1], dtype=np.float32) * region_l
    xl_bd_batch = torch.from_numpy(xl_bd_batch)
    xr_bd_batch = np.ones(shape=[batchsize_bd, 1], dtype=np.float32) * region_r
    xr_bd_batch = torch.from_numpy(xr_bd_batch)
    t_init_batch = np.ones(shape=[batchsize_init, 1], dtype=np.float32) * init_time
    t_init_batch = torch.from_numpy(t_init_batch)
    if True == R['use_gpu']:
        test_xy_torch = test_xy_torch.cuda(device='cuda:' + str(R['gpuNo']))
        xl_bd_batch = xl_bd_batch.cuda(device='cuda:' + str(R['gpuNo']))
        xr_bd_batch = xr_bd_batch.cuda(device='cuda:' + str(R['gpuNo']))
        t_init_batch = t_init_batch.cuda(device='cuda:' + str(R['gpuNo']))

    xl_bd_batch.requires_grad_(True)
    xr_bd_batch.requires_grad_(True)
    Utrue2test1 = u_true(torch.reshape(test_xy_torch[:, 0], shape=[-1, 1]),
                         torch.reshape(test_xy_torch[:, 1], shape=[-1, 1]))
    Utrue2test = torch.nan_to_num(input=Utrue2test1, nan=0.0)
    Utrue2test[0] = C0
    t0 = time.time()
    for i_epoch in range(R['max_epoch'] + 1):
        # 内部点
        delta_t = 0.1
        delta_t = end_time/R['max_epoch']
        num_stairs = end_time/delta_t
        stair = R['max_epoch']/num_stairs
        if Seq2seq:
            epsi_t = 0.0000000000001
            n = min(i_epoch+1, R['max_epoch'])
            # t_in_batch = dataUtilizer2torch.rand_it(batchsize_it, 1, region_a=init_time, region_b=init_time+delta_t * (1 + int(stair)),
            #                                     to_float=True, to_cuda=R['use_gpu'],
            #                                     gpu_no=R['gpuNo'], use_grad2x=True)
            # # 边界点 左右边界时间点取相同
            # t_bd_batch = dataUtilizer2torch.rand_it(batchsize_bd, 1, region_a=init_time, region_b=init_time+delta_t * (1 + int(stair)),
            #                                     to_float=True, to_cuda=R['use_gpu'],
            #                                     gpu_no=R['gpuNo'], use_grad2x=False)
            t_in_batch = dataUtilizer2torch.rand_it(batchsize_it, 1, region_a=init_time, region_b=init_time + delta_t * n,
                                                to_float=True, to_cuda=R['use_gpu'],
                                                gpu_no=R['gpuNo'], use_grad2x=True, lhs_sampling=True)
            # 边界点 左右边界时间点取相同
            # t_bd_batch = dataUtilizer2torch.rand_it(batchsize_bd, 1, region_a=init_time, region_b=init_time + delta_t * n,
            #                                     to_float=True, to_cuda=R['use_gpu'],
            #                                     gpu_no=R['gpuNo'], use_grad2x=False, lhs_sampling=True)
            t_bd_batch = dataUtilizer2torch.rand_it(batchsize_bd, 1, region_a=init_time,
                                                    region_b=init_time + delta_t * n,
                                                    to_float=True, to_cuda=R['use_gpu'],
                                                    gpu_no=R['gpuNo'], use_grad2x=False, lhs_sampling=True)
        else:
            t_in_batch = dataUtilizer2torch.rand_it(batchsize_it, 1, region_a=init_time,
                                                    region_b=end_time,
                                                    to_float=True, to_cuda=R['use_gpu'],
                                                    gpu_no=R['gpuNo'], use_grad2x=True, lhs_sampling=True)
            t_bd_batch = dataUtilizer2torch.rand_it(batchsize_bd, 1, region_a=init_time,
                                                region_b=end_time,
                                                to_float=True, to_cuda=R['use_gpu'],
                                                gpu_no=R['gpuNo'], use_grad2x=False, lhs_sampling=True)
        if Seq2seq2x:
            # 内部点
            delta_x = 0.1
            delta_x = region_r / R['max_epoch']
            num_stairs = end_time / delta_x
            stair_x = R['max_epoch'] / num_stairs
            x_init_batch = dataUtilizer2torch.rand_it(batchsize_init, 1, region_a=region_l + 0.000000000001,
                                                      region_b=region_l + i_epoch*delta_x,
                                                      to_float=True, to_cuda=R['use_gpu'],
                                                      gpu_no=R['gpuNo'], use_grad2x=False, lhs_sampling=True)
            x_in_batch = dataUtilizer2torch.rand_it(batchsize_it, input_dim, region_a=region_l, region_b=region_l + i_epoch*delta_x,
                                                    to_float=True, to_cuda=R['use_gpu'],
                                                    gpu_no=R['gpuNo'], use_grad2x=True, lhs_sampling=True)
        else:
            x_init_batch = dataUtilizer2torch.rand_it(batchsize_init, 1, region_a=region_l, region_b=region_r,
                                                  to_float=True, to_cuda=R['use_gpu'],
                                                  gpu_no=R['gpuNo'], use_grad2x=False, lhs_sampling=True)
            x_in_batch = dataUtilizer2torch.rand_it(batchsize_it, input_dim, region_a=region_l, region_b=region_r,
                                                to_float=True, to_cuda=R['use_gpu'],
                                                gpu_no=R['gpuNo'], use_grad2x=True, lhs_sampling=True)
        # 计算损失函数
        if R['activate_penalty2bd_increase'] == 1:
            if i_epoch < int(R['max_epoch'] / 10):
                temp_penalty_bd = bd_penalty_init
            elif i_epoch < int(R['max_epoch'] / 5):
                temp_penalty_bd = 10 * bd_penalty_init
            elif i_epoch < int(R['max_epoch'] / 4):
                temp_penalty_bd = 50 * bd_penalty_init
            elif i_epoch < int(R['max_epoch'] / 2):
                temp_penalty_bd = 100 * bd_penalty_init
            elif i_epoch < int(3 * R['max_epoch'] / 4):
                temp_penalty_bd = 200 * bd_penalty_init
            else:
                temp_penalty_bd = 500 * bd_penalty_init
        else:
            temp_penalty_bd = bd_penalty_init

        if R['activate_penalty2bd_increase'] == 1:
            if i_epoch < int(R['max_epoch'] / 10):
                temp_penalty_init = init_penalty_init
            elif i_epoch < int(R['max_epoch'] / 5):
                temp_penalty_init = 10 * init_penalty_init
            elif i_epoch < int(R['max_epoch'] / 4):
                temp_penalty_init = 50 * init_penalty_init
            elif i_epoch < int(R['max_epoch'] / 2):
                temp_penalty_init = 100 * init_penalty_init
            elif i_epoch < int(3 * R['max_epoch'] / 4):
                temp_penalty_init = 200 * init_penalty_init
            else:
                temp_penalty_init = 500 * init_penalty_init
        else:
            temp_penalty_init = init_penalty_init

        # 内部点损失 用pinn就没有初始点的选取
        UNN2train, loss_it = model.loss_1DV(X_in=x_in_batch, T_in=t_in_batch, loss_type=R['loss_type'], ws=ws, ds=ds)
        # UNN2train, loss_it = model.loss_1DV_EX5(X_in=x_in_batch, T_in=t_in_batch, loss_type=R['loss_type'],
        #                                         ws=ws, ds=ds, f_side=f_side)
        loss_bd2left = model.loss2bd_dirichlet(X_bd=xl_bd_batch, T_bd=t_bd_batch, Ubd_exact=u_left,
                                               if_lambda2Ubd=True, loss_type=R['loss_type'])
        # loss_bd2left = model.loss2bd_Neumann_A4(X_bd=xl_bd_batch, T_bd=t_bd_batch, Ubd_exact=u_left, ws=ws, ds=ds,
        #                                        if_lambda2Ubd=True, loss_type=R['loss_type'])
        # loss_bd2right = model.loss2bd_dirichlet(X_bd=xr_bd_batch, T_bd=t_bd_batch, Ubd_exact=u_right,
        #                                         if_lambda2Ubd=True, loss_type=R['loss_type'])
        # loss_bd2right2 = model.loss2bd_Neumann(X_bd=xr_bd_batch, T_bd=t_bd_batch, Ubd_exact=u_right,
        #                                         if_lambda2Ubd=True, loss_type=R['loss_type'])
        loss_bd2right = model.loss2bd_dirichlet(X_bd=xr_bd_batch, T_bd=t_bd_batch, Ubd_exact=u_right,
                                                if_lambda2Ubd=True, loss_type=R['loss_type'])
        #变化率为0

        loss_init = model.loss2init(X_init=x_init_batch, T_init=t_init_batch, Uinit_exact=u_init,
                                    if_lambda2Uinit=True, loss_type=R['loss_type'])

        loss_bd = loss_bd2left + loss_bd2right
        # PWB = penalty2WB * model.get_regularSum2WB()
        loss = loss_it + loss_bd * temp_penalty_bd + loss_init * temp_penalty_init

        loss_it_all.append(loss_it.item())
        loss_all.append(loss.item())

        optimizer.zero_grad()  # 求导前先清零, 只要在下一次求导前清零即可
        loss.backward()  # 对loss关于Ws和Bs求偏导
        optimizer.step()  # 更新参数Ws和Bs
        scheduler.step()

        if i_epoch % 1000 == 0:
            run_times = time.time() - t0
            PWB = 0.0
            tmp_lr = optimizer.param_groups[0]['lr']
            Uexact2train = u_true(x_in_batch, t_in_batch)
            # 把为nan的转化为0
            Uexact2train = torch.nan_to_num(input=Uexact2train, nan=0.0)
            # Uexact2train = torch.clamp(Uexact2train, min=0.0)
            train_mse = torch.mean(torch.square(UNN2train - Uexact2train))
            train_rel = train_mse / torch.mean(torch.square(Uexact2train))
            train_mse_all.append(train_mse.item())
            train_rel_all.append(train_rel.item())
            DNN_Log_Print.print_and_log_train_one_epoch2Ocean(
                i_epoch, run_times, tmp_lr, temp_penalty_bd, temp_penalty_init, PWB, loss_it.item(), loss_bd.item(),
                loss_init.item(), loss.item(),
                train_mse.item(), train_rel.item(), log_out=log_fileout)

            test_epoch.append(i_epoch / 1000)
            UNN2test = model.evalue_MscaleDNN(XY_points=test_xy_torch)
            # UNN2test = torch.nan_to_num(input=UNN2test, nan=0.0)
            # Utrue2test = torch.clamp(Utrue2test, min=0.0)
            point_square_error = torch.square(Utrue2test - UNN2test)
            test_mse = torch.mean(point_square_error)
            test_rel = test_mse / torch.mean(torch.square(Utrue2test))
            test_mse_all.append(test_mse.item())
            test_rel_all.append(test_rel.item())
            DNN_tools.print_and_log_test_one_epoch(test_mse.item(), test_rel.item(), log_out=log_fileout)

    # ------------------- save the training results into mat file and plot them -------------------------
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=R['activate_func'],
                                         outPath=R['FolderName'])
    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['activate_func'], outPath=R['FolderName'])
    # #
    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['activate_func'], outPath=R['FolderName'])
    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=R['activate_func'], seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)
    #
    # # ----------------------  save testing results to mat files, then plot them --------------------------------
    if True == R['use_gpu']:
        utrue2test_numpy = Utrue2test.cpu().detach().numpy()
        unn2test_numpy = UNN2test.cpu().detach().numpy()
        unn2test_numpy = unn2test_numpy.reshape((size2test, size2test))
    else:
        utrue2test_numpy = Utrue2test.detach().numpy()
        unn2test_numpy = UNN2test.detach().numpy()

    saveData.save_2testSolus2mat(utrue2test_numpy, unn2test_numpy, actName='utrue',
                                 actName1='sin', outPath=R['FolderName'])
    #
    # plotData.plot_Hot_solution2test(utrue2test_numpy, size_vec2mat=size2test**2, actName='Utrue',
    #                                 seedNo=R['seed'], outPath=R['FolderName'])
    # plotData.plot_Hot_solution2test(unn2test_numpy, size_vec2mat=size2test, actName=R['name2act_hidden'],
    #                                 seedNo=R['seed'], outPath=R['FolderName'])

    # saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=R['name2act_hidden'], outPath=R['FolderName'])
    # plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=R['activate_func'],
    #                           seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)
    #
    # saveData.save_test_point_wise_err2mat(point_square_error_numpy, actName=R['name2act_hidden'],
    #                                       outPath=R['FolderName'])

    # plotData.plot_Hot_point_wise_err(point_square_error_numpy, size_vec2mat=size2test,
    #                                  actName=R['activate_func'], seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R = {}
    R['gpuNo'] = 0
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')
    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model2NN'] = 'DNN'
    # R['model2NN'] = 'Scale_DNN'
    # R['model2NN'] = 'Fourier_DNN'
    R['model2NN'] = 'Fourier_SubDNN'
    # R['model2NN'] = 'MULTI_3FF_DNN'  # 第一层隐藏层都设置成偶数
    R['max_epoch'] = 100000

    # 文件保存路径设置
    # if R['model2NN'] == 'Fourier_SubDNN':
    #     store_file = 'soft_sub_Fourier_Ocean'
    # elif R['model2NN'] == 'DNN':
    #     store_file = 'baseline_EX1'
    # else:
    store_file = 'OceanDNN_Seq2seq_0519'

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])  # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of multi-scale problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    R['input_dim'] = 1  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数

    R['PDE_type'] = 'PINN'
    R['equa_name'] = 'PINN'

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    R['batch_size2interior'] = 15000  # 内部训练数据的批大小
    R['batch_size2boundary'] = 5000  # 边界训练数据的批大小
    R['batch_size2init'] = 1000
    R['batch_size2test'] = 100

    # 装载测试数据模式
    # R['testData_model'] = 'loadData'
    R['testData_model'] = 'random_generate'

    R['loss_type'] = 'L2_loss'  # loss类型:L2 loss
    # R['loss_type'] = 'variational_loss'                      # loss类型:PDE变分
    # R['loss_type'] = 'lncosh_loss'
    R['lambda2lncosh'] = 0.5

    R['optimizer_name'] = 'Adam'  # 优化器
    R['learning_rate'] = 0.01  # 学习率
    # R['learning_rate'] = 0.001              # 学习率
    R['learning_rate_decay'] = 5e-5  # 学习率 decay
    R['train_model'] = 'union_training'

    # 正则化权重和偏置的模式
    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    # R['penalty2weight_biases'] = 0.000                    # Regularization parameter for weights
    R['penalty2weight_biases'] = 0.001  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                 # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    # R['activate_penalty2bd_increase'] = 1
    R['activate_penalty2bd_increase'] = 0
    R['init_boundary_penalty'] = 20

    # R['activate_penalty2init_increase'] = 1
    R['activate_penalty2init_increase'] = 0
    R['init_penalty2init'] = 1

    # 网络的频率范围设置
    R['freq'] = np.concatenate(([1], np.arange(1, 40 - 1)), axis=0)
    R['repeat_highFreq'] = True

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'Fourier_SubDNN':
        R['hidden_layers'] = (10, 25, 20, 20, 15)  # （1*10+250+500+400+300+15）* 20 = 1475 *20 (subnet个数) = 29500
        # R['hidden_layers'] = (20, 35, 25, 25, 15)# 调大一些
        R['freq'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.float32)
        # R['freq'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
        # R['hidden_layers'] = (125, 200, 100, 100, 80)  # 1*125+250*200+200*200+200*100+100*100+100*50+50*1=128205
        # R['hidden_layers'] = (50, 10, 10, 10)
        # R['hidden_layers'] = (50, 80, 60, 60, 40)
    elif R['model2NN'] == 'MULTI_4FF_DNN' or R['model2NN'] == 'MULTI_3FF_DNN':
        R['hidden_layers'] = (120, 100, 80, 80, 30)
    else:
        R['hidden_layers'] = (100, 150, 80, 80, 50)  # 1*100+100*150+150*80+80*50+50*1 = 31150
        # R['hidden_layers'] = (200, 100, 80, 50, 30)
        # R['hidden_layers'] = (250, 200, 100, 100, 80)  # 1*250+250*200+200*200+200*100+100*100+100*50+50*1=128330
        # R['hidden_layers'] = (500, 400, 300, 200, 100)
        # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['name2act_in'] = 'elu'
    # R['name2act_hidden'] = 'elu'
    # R['name2act_in'] = 'relu'
    # R['name2act_hidden'] = 'relu'
    R['name2act_in'] = 'sin'
    R['name2act_hidden'] = 'sin'
    R['name2act_out'] = 'linear'
    R['activate_func'] = R['name2act_hidden']
    R['sfourier'] = 1.0

    R['use_gpu'] = True
    # R['use_gpu'] = False

    solve_Multiscale_PDE(R)

    # R['name2act_in'] = 'relu'
    # R['name2act_in'] = 'sin'
    # R['name2act_in'] = 'tanh'
    # R['name2act_in'] = 's2relu'
    # R['name2act_in'] = 'Enh_tanh'

    # R['name2act_hidden'] = 'relu'
    # R['name2act_hidden'] = 'tanh'
    # R['name2act_hidden']' = leaky_relu'
    # R['name2act_hidden'] = 'srelu'
    # R['name2act_hidden'] = 's2relu'
    # R['name2act_hidden'] = 'sin'
    # R['name2act_hidden'] = 'sinAddcos'
    # R['name2act_hidden'] = 'Enh_tanh'
    # R['name2act_hidden'] = 'elu'
    # R['name2act_hidden'] = 'phi'

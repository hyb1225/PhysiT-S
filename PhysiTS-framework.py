import sys

sys.path.insert(0, '../Utilities/')
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as io
import numpy as np
import math
import warnings
import weight_adjustment

warnings.filterwarnings('ignore')

np.random.seed(1234)

# NOTE: parameter setting
Tair = 15
bx = 1
ET = 0.00022
# boundary conditions
x_r = 6.5
x_l = 0
# initial condition
t0 = 0
i0 = 7
v0 = 0
time_steps = 2400
# Physics Prior
lambda1 = math.sqrt(0.03954194)
lambda2 = 0
PhysPrior = [lambda1, lambda2]

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda:0')


# the deep neural network
class DNN1(torch.nn.Module):
    def __init__(self, inC, outC):
        super(DNN1, self).__init__()
        self.fc1 = nn.Linear(inC, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 80)
        self.fc4 = nn.Linear(80, 80)
        self.fc5 = nn.Linear(80, 80)
        self.fc6 = nn.Linear(80, 80)
        # NOTE: Narrowing Structure
        self.fc7 = nn.Linear(80, 60)
        self.fc8 = nn.Linear(60, outC)

    def forward(self, x):
        # NOTE: PhyiS/T TEST
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        # NOTE: Tanh function
        x = F.tanh(self.fc7(x))
        # NOTE: last layer with fully-connected: fc(Wx+b)
        out = F.tanh(self.fc8(x))
        return out


class PhysEnhancedUnit(torch.nn.Module):
    # TODO: design the spatial-temporal enhanced unit
    def __init__(self, inC, N):
        super(PhysEnhancedUnit, self).__init__()
        self.fcx = nn.Linear(inC, N)
        self.fct = nn.Linear(inC, N)

    def SpatialBFs(self, x):
        # NOTE: design the SBF form
        # NOTE: set the BFs coefficient
        c1, c2 = PhysPrior
        hidden = self.fcx(x)
        return torch.cos(c1 * hidden) * lambda1 + torch.sin(c2 * hidden)

    def TimeVariables(self, x):
        # NOTE: design the TV form
        x_normalized = x / time_steps
        hidden = self.fct(x_normalized)
        return F.tanh(hidden)

    def forward(self, x1, x2):
        out = torch.mul(self.SpatialBFs(x1), self.TimeVariables(x2))
        return out


class PhysicsEnhancedTS():
    def __init__(self, X, u, Eoc):
        # NOTE: input initialization
        self.eoc = torch.tensor(Eoc).float().to(device)
        self.previous_loss = None
        # data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.i = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)
        self.v = torch.tensor(X[:, 3:4], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.dim = X.shape[0]

        # set boundary conditions
        xbc_r = np.ones(self.dim).reshape(self.dim, 1) * x_r
        xbc_l = np.ones(self.dim).reshape(self.dim, 1) * x_l
        self.x_bc_r = torch.Tensor(xbc_r).requires_grad_(True).to(device)
        self.x_bc_l = torch.Tensor(xbc_l).requires_grad_(True).to(device)

        # set initial condition
        self.t_ic = torch.Tensor(t0).requires_grad_(True).to(device)
        self.i_ic = torch.Tensor(i0).requires_grad_(True).to(device)
        self.v_ic = torch.Tensor(v0).requires_grad_(True).to(device)

        # variable parameter settings
        self.rohCp = torch.tensor([30.927], requires_grad=True).to(device)
        self.Kx = torch.tensor([0.80], requires_grad=True).to(device)
        self.hc = torch.tensor([0.12], requires_grad=True).to(device)

        # neural networks
        self.PEUnit = PhysEnhancedUnit(1, 60).to(device)
        self.dnn1 = DNN1(4, 60).to(device)
        params = list(self.PEUnit.parameters()) + list(self.dnn1.parameters())

        # NOTE: Optimizer
        self.optimizer1 = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"  # can be "strong_wolfe"
        )
        self.optimizer_Adam1 = torch.optim.Adam(
            params,
            lr=0.001,
            betas=(0.9, 0.99),
            eps=1e-08,
            weight_decay=0
        )
        self.optimizer_Adam11 = torch.optim.Adam(
            params,
            lr=0.0001,
            betas=(0.9, 0.99),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False)

        self.scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_Adam1, milestones=[4000, 8000],
                                                                gamma=0.5)
        self.scheduler_11 = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_Adam11, milestones=[6000, 10000],
                                                                 gamma=0.1)

        self.iter = 0

    def net_A(self, x, t, i, v):
        x1 = x
        t1 = t / time_steps
        i1 = i
        v1 = v
        a = self.dnn1(torch.cat([x1, t1, i1, v1], dim=1))
        return a

    def net_u(self, x, t, i, v):
        a = self.net_A(x, t, i, v)
        sp = self.PEUnit(x, t)
        u = torch.mul(a, sp)
        return u

    def net_f(self, x, t, i, v, eoc):
        """ The pytorch autograd version of calculating residual """
        rohCp = self.rohCp
        Kx = self.Kx
        u = self.net_u(x, t, i, v)
        q = bx * i * (eoc - v - ET * u)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        f = rohCp * u_t - Kx * u_xx - q
        return f

    def net_bc_l(self, x_bc_l, t, i, v):
        Kx = self.Kx
        hv = self.hc
        u_bc = self.net_u(x_bc_l, t, i, v)
        u_bc_x = torch.autograd.grad(
            u_bc, x_bc_l,
            grad_outputs=torch.ones_like(u_bc),
            retain_graph=True,
            create_graph=True
        )[0]
        bc_l = -Kx * u_bc_x + hv * (u_bc - Tair)
        return bc_l

    def net_bc_r(self, x_bc_r, t, i, v):
        Kx = self.Kx
        hv = self.hc
        u_bc = self.net_u(x_bc_r, t, i, v)
        u_bc_x = torch.autograd.grad(
            u_bc, x_bc_r,
            grad_outputs=torch.ones_like(u_bc),
            retain_graph=True,
            create_graph=True
        )[0]
        bc_r = -Kx * u_bc_x - hv * (u_bc - Tair)
        return bc_r

    def net_ic(self, x, t_ic, i_ic, v_ic):
        u_t0 = 15
        u_ic = self.net_u(x, t_ic, i_ic, v_ic)
        ic_loss = u_ic - u_t0
        return ic_loss

    def loss_func(self):
        u_pred = self.net_u(self.xh, self.th, self.ih, self.vh)
        f_pred = self.net_f(self.xh, self.th, self.ih, self.vh, self.eoch)
        bc_l = self.net_bc_l(self.xh_bc_l, self.th, self.ih, self.vh)
        bc_r = self.net_bc_r(self.xh_bc_r, self.th, self.ih, self.vh)
        ic = self.net_ic(self.xh, self.th_ic, self.ih_ic, self.vh_ic)
        loss_bcic = np.linalg.norm(bc_l) + np.linalg.norm(ic) + np.linalg.norm(bc_r)
        loss_u = torch.mean((self.uh - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)

        # WAA
        current_loss = [loss_u, loss_f, loss_bcic]
        if epoch > 10:
            u_w, f_w, bc_w = weight_adjustment.wad(current_loss, self.previous_loss)
        else:
            u_w, f_w, bc_w = [1, 1, 1]

        loss = loss_u * u_w + loss_f * f_w + loss_bcic * bc_w

        self.optimizer1.zero_grad()
        loss.backward()
        lossu = loss_u.item()
        lossf = loss_f.item()
        lossib = loss_bcic.item()
        self.previous_loss = [lossu, lossf, lossib]

        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Loss: %e' %  # , rohCp: %.3f, Kx: %.6f, hc: %.6f
                (
                    loss.item(),
                )
            )
        return loss

    def pretrain(self, nIter):

        print('pretrain')

        self.dnn1.train()
        for epoch in range(nIter):
            self.iter += 1
            u_pred = self.net_u(self.x, self.t, self.i, self.v)
            f_pred = self.net_f(self.x, self.t, self.i, self.v, self.eoc)
            bc_l = self.net_bc_l(self.x_bc_l, self.t, self.i, self.v)
            bc_r = self.net_bc_r(self.x_bc_r, self.t, self.i, self.v)
            ic = self.net_ic(self.x, self.t_ic, self.i_ic, self.v_ic)
            loss_bcic = np.linalg.norm(bc_l) + np.linalg.norm(ic) + np.linalg.norm(bc_r)
            loss_u = torch.mean((self.u - u_pred) ** 2)
            loss_f = torch.mean(f_pred ** 2)

            # WAA
            current_loss = [loss_u, loss_f, loss_bcic]
            if epoch > 10:
                u_w, f_w, bc_w = weight_adjustment.wad(current_loss, self.previous_loss)
            else:
                u_w, f_w, bc_w = [1, 1, 1]

            loss = loss_u * u_w + loss_f * f_w + loss_bcic * bc_w

            # Backward and optimize
            self.optimizer_Adam1.zero_grad()
            loss.backward(retain_graph=True)
            lossu = loss_u.item()
            lossf = loss_f.item()
            lossib = loss_bcic.item()
            self.previous_loss = [lossu, lossf, lossib]

            self.optimizer_Adam1.step()
            self.scheduler_1.step()

            if epoch % 100 == 0:
                print(
                    'It: %d, Loss: %.3e,u:%.3e,f:%.3e,bcic:%.3e' %  # , hc: %.6f
                    (
                        epoch,
                        loss.item(),
                        loss_u.item(),
                        loss_f.item(),
                        loss_bcic.item(),
                    )
                )

        # Backward and optimize

    def update(self, nIter, Xh, uh, Eoch):

        self.eoch = torch.tensor(Eoch).float().to(device)
        self.xh = torch.tensor(Xh[:, 0:1], requires_grad=True).float().to(device)
        self.th = torch.tensor(Xh[:, 1:2], requires_grad=True).float().to(device)
        self.ih = torch.tensor(Xh[:, 2:3], requires_grad=True).float().to(device)
        self.vh = torch.tensor(Xh[:, 3:4], requires_grad=True).float().to(device)
        self.uh = torch.tensor(uh).float().to(device)

        x_max = max(self.xh).detach().cpu().numpy()
        v_max = max(self.vh).detach().cpu().numpy()
        self.dim_h = Xh.shape[0]
        xhbc_r = np.ones(self.dim_h).reshape(self.dim_h, 1) * x_max
        self.xh_bc_r = torch.Tensor(xhbc_r).requires_grad_(True).to(device)
        self.xh_bc_l = torch.zeros(self.dim_h, 1).requires_grad_(True).to(device)
        self.th_ic = torch.zeros(self.dim_h, 1).requires_grad_(True).to(device)

        vh_ic = np.ones(self.dim_h).reshape(self.dim_h, 1) * v_max
        self.ih_ic = torch.zeros(self.dim_h, 1).requires_grad_(True).to(device)
        self.vh_ic = torch.Tensor(vh_ic).requires_grad_(True).to(device)

        print('train')

        freeze_layers = ("fc1", "fc2", "fc3", "fc4", "fc5", "fc6")
        for name, param in self.dnn1.named_parameters():
            print(name, param.shape)
            if name.split(".")[0] in freeze_layers:
                param.requires_grad = False

        self.dnn1.train()
        for epoch in range(nIter):
            self.iter += 1
            u_pred = self.net_u(self.xh, self.th, self.ih, self.vh)
            f_pred = self.net_f(self.xh, self.th, self.ih, self.vh, self.eoch)
            bc_l = self.net_bc_l(self.xh_bc_l, self.th, self.ih, self.vh)
            bc_r = self.net_bc_r(self.xh_bc_r, self.th, self.ih, self.vh)
            ic = self.net_ic(self.xh, self.th_ic, self.ih_ic, self.vh_ic)
            loss_bcic = np.linalg.norm(bc_l) + np.linalg.norm(ic) + np.linalg.norm(bc_r)
            loss_u = torch.mean((self.uh - u_pred) ** 2)
            loss_f = torch.mean(f_pred ** 2)

            # WAA
            current_loss = [loss_u, loss_f, loss_bcic]
            if epoch > 10:
                u_w, f_w, bc_w = weight_adjustment.wad(current_loss, self.previous_loss)
            else:
                u_w, f_w, bc_w = [1, 1, 1]

            loss = loss_u * u_w + loss_f * f_w + loss_bcic * bc_w

            # Backward and optimize
            self.optimizer_Adam11.zero_grad()

            loss.backward()  # retain_graph=True
            lossu = loss_u.item()
            lossf = loss_f.item()
            lossib = loss_bcic.item()
            self.previous_loss = [lossu, lossf, lossib]

            self.optimizer_Adam11.step()
            self.scheduler_11.step()

            if epoch % 100 == 0:
                print(
                    'It: %d, Loss: %.3e,u:%.3e,f:%.3e,bcic:%.3e' %  # , hc: %.6f
                    (
                        epoch,
                        loss.item(),
                        loss_u.item(),
                        loss_f.item(),
                        loss_bcic.item(),
                    )
                )

        # Backward and optimize
        self.optimizer1.step(self.loss_func)

    def predict(self, X, Eoc):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        i = torch.tensor(X[:, 2:3], requires_grad=True).float().to(device)
        v = torch.tensor(X[:, 3:4], requires_grad=True).float().to(device)
        eoc = torch.tensor(Eoc).float().to(device)

        # self.dnn.eval()
        self.dnn1.eval()
        u = self.net_u(x, t, i, v)
        f = self.net_f(x, t, i, v, eoc)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()

        return u, f

    def state_dict(self):
        pass





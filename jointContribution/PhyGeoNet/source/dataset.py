import numpy as np
import paddle


class VaryGeoDataset(paddle.io.Dataset):
    def __init__(self, mesh_list):
        self.mesh_list = mesh_list

    def __len__(self):
        return len(self.mesh_list)

    def __getitem__(self, idx):
        mesh = self.mesh_list[idx]
        x = mesh.x
        y = mesh.y
        xi = mesh.xi
        eta = mesh.eta
        j = mesh.j_ho
        j_inv = mesh.j_inv_ho
        dxdxi = mesh.dxdxi_ho
        dydxi = mesh.dydxi_ho
        dxdeta = mesh.dxdeta_ho
        dydeta = mesh.dydeta_ho
        cord = np.zeros([2, x.shape[0], x.shape[1]])
        cord[0, :, :] = x
        cord[1, :, :] = y
        invariant_input = np.zeros([2, j.shape[0], j.shape[1]])
        invariant_input[0, :, :] = j
        invariant_input[1, :, :] = j_inv
        return [invariant_input, cord, xi, eta, j, j_inv, dxdxi, dydxi, dxdeta, dydeta]


class FixGeoDataset(paddle.io.Dataset):
    def __init__(self, para_list, mesh, of_solution_list):
        self.para_list = para_list
        self.mesh = mesh
        self.of_solution_list = of_solution_list

    def __len__(self):
        return len(self.para_list)

    def __getitem__(self, idx):
        mesh = self.mesh
        x = mesh.x
        y = mesh.y
        xi = mesh.xi
        eta = mesh.eta
        j = mesh.j_ho
        j_inv = mesh.j_inv_ho
        dxdxi = mesh.dxdxi_ho
        dydxi = mesh.dydxi_ho
        dxdeta = mesh.dxdeta_ho
        dydeta = mesh.dydeta_ho
        cord = np.zeros([2, x.shape[0], x.shape[1]])
        cord[0, :, :] = x
        cord[1, :, :] = y
        ParaStart = np.ones(x.shape[0]) * self.para_list[idx]
        ParaEnd = np.zeros(x.shape[0])
        Para = np.linspace(ParaStart, ParaEnd, x.shape[1]).T
        return [
            Para,
            cord,
            xi,
            eta,
            j,
            j_inv,
            dxdxi,
            dydxi,
            dxdeta,
            dydeta,
            self.of_solution_list[idx],
        ]


class VaryGeoDataset_PairedSolution(paddle.io.Dataset):
    def __init__(self, mesh_list, solution_list):
        self.mesh_list = mesh_list
        self.solution_list = solution_list

    def __len__(self):
        return len(self.mesh_list)

    def __getitem__(self, idx):
        mesh = self.mesh_list[idx]
        x = mesh.x
        y = mesh.y
        xi = mesh.xi
        eta = mesh.eta
        j = mesh.j_ho
        j_inv = mesh.j_inv_ho
        dxdxi = mesh.dxdxi_ho
        dydxi = mesh.dydxi_ho
        dxdeta = mesh.dxdeta_ho
        dydeta = mesh.dydeta_ho
        cord = np.zeros([2, x.shape[0], x.shape[1]])
        cord[0, :, :] = x
        cord[1, :, :] = y
        invariant_input = np.zeros([2, j.shape[0], j.shape[1]])
        invariant_input[0, :, :] = j
        invariant_input[1, :, :] = j_inv
        return [
            invariant_input,
            cord,
            xi,
            eta,
            j,
            j_inv,
            dxdxi,
            dydxi,
            dxdeta,
            dydeta,
            self.solution_list[idx][:, :, 0],
            self.solution_list[idx][:, :, 1],
            self.solution_list[idx][:, :, 2],
        ]

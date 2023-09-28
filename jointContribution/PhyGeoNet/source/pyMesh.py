import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paddle
from matplotlib import collections
from matplotlib import patches

errMessageJoint = "The geometry is not closed!"
errMessageParallel = "The parallel sides do not have the same number of node!"
errMessageXYShape = "The x y shapes do not have match each other!"
errMessageDomainType = "domainType can only be physical domain or reference domain!"
arrow = "====>"
clow = "green"
cup = "blue"
cright = "red"
cleft = "orange"
cinternal = "black"


def np2cuda(mylist):
    my_list = []
    for item in mylist:
        my_list.append(item)
    return my_list


def to4DTensor(mylist):
    my_list = []
    for item in mylist:
        if len(item.shape) == 3:
            item = paddle.to_tensor(
                item.reshape([item.shape[0], 1, item.shape[1], item.shape[2]])
            )
            my_list.append(item.astype(paddle.get_default_dtype()))
        else:
            item = paddle.to_tensor(item)
            my_list.append(item.astype(paddle.get_default_dtype()))
    return my_list


def checkGeo(left_x, left_y, right_x, right_y, low_x, low_y, up_x, up_y, tol_joint):
    print(arrow + "Check bc nodes!")
    assert (
        len(left_x.shape)
        == len(left_y.shape)
        == len(right_x.shape)
        == len(right_y.shape)
        == len(low_x.shape)
        == len(low_y.shape)
        == len(up_x.shape)
        == len(up_y.shape)
        == 1
    ), "all left(right)X(Y) must be 1d vector!"
    assert np.abs(left_x[0] - low_x[0]) < tol_joint, errMessageJoint
    assert np.abs(left_x[-1] - up_x[0]) < tol_joint, errMessageJoint
    assert np.abs(right_x[0] - low_x[-1]) < tol_joint, errMessageJoint
    assert np.abs(right_x[-1] - up_x[-1]) < tol_joint, errMessageJoint
    assert np.abs(left_y[0] - low_y[0]) < tol_joint, errMessageJoint
    assert np.abs(left_y[-1] - up_y[0]) < tol_joint, errMessageJoint
    assert np.abs(right_y[0] - low_y[-1]) < tol_joint, errMessageJoint
    assert np.abs(right_y[-1] - up_y[-1]) < tol_joint, errMessageJoint
    assert (
        left_x.shape == left_y.shape == right_x.shape == right_y.shape
    ), errMessageParallel
    assert up_x.shape == up_y.shape == low_x.shape == low_y.shape, errMessageParallel
    print(arrow + "BC nodes pass!")


def plotBC(ax, x, y):
    ax.plot(x[:, 0], y[:, 0], "-o", color=cleft)  # left BC
    ax.plot(x[:, -1], y[:, -1], "-o", color=cright)  # right BC
    ax.plot(x[0, :], y[0, :], "-o", color=clow)  # low BC
    ax.plot(x[-1, :], y[-1, :], "-o", color=cup)  # up BC
    return ax


def plotMesh(ax, x, y, width=0.05):
    [ny, nx] = x.shape
    for j in range(0, nx):
        ax.plot(x[:, j], y[:, j], color=cinternal, linewidth=width)
    for i in range(0, ny):
        ax.plot(x[i, :], y[i, :], color=cinternal, linewidth=width)
    return ax


def setAxisLabel(ax, type):
    if type == "p":
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
    elif type == "r":
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
    else:
        raise ValueError("The axis type only can be reference or physical")


def ellipticMap(x, y, h, tol):
    eps = 2.2e-16
    assert x.shape == y.shape, errMessageXYShape
    [ny, nx] = x.shape
    ite = 1
    A = np.ones([ny - 2, nx - 2])
    B = A
    C = A
    err_list = []
    while True:
        X = (
            (
                A * (x[2:, 1:-1] + x[0:-2, 1:-1])
                + C * (x[1:-1, 2:] + x[1:-1, 0:-2])
                - B / 2 * (x[2:, 2:] + x[0:-2, 0:-2] - x[2:, 0:-2] - x[0:-2, 2:])
            )
            / 2
            / (A + C)
        )
        Y = (
            (
                A * (y[2:, 1:-1] + y[0:-2, 1:-1])
                + C * (y[1:-1, 2:] + y[1:-1, 0:-2])
                - B / 2 * (y[2:, 2:] + y[0:-2, 0:-2] - y[2:, 0:-2] - y[0:-2, 2:])
            )
            / 2
            / (A + C)
        )
        err = np.max(np.max(np.abs(x[1:-1, 1:-1] - X))) + np.max(
            np.max(np.abs(y[1:-1, 1:-1] - Y))
        )
        err_list.append(err)
        x[1:-1, 1:-1] = X
        y[1:-1, 1:-1] = Y
        A = (
            ((x[1:-1, 2:] - x[1:-1, 0:-2]) / 2 / h) ** 2
            + ((y[1:-1, 2:] - y[1:-1, 0:-2]) / 2 / h) ** 2
            + eps
        )
        B = (
            (x[2:, 1:-1] - x[0:-2, 1:-1])
            / 2
            / h
            * (x[1:-1, 2:] - x[1:-1, 0:-2])
            / 2
            / h
            + (y[2:, 1:-1] - y[0:-2, 1:-1])
            / 2
            / h
            * (y[1:-1, 2:] - y[1:-1, 0:-2])
            / 2
            / h
            + eps
        )
        C = (
            ((x[2:, 1:-1] - x[0:-2, 1:-1]) / 2 / h) ** 2
            + ((y[2:, 1:-1] - y[0:-2, 1:-1]) / 2 / h) ** 2
            + eps
        )
        if err < tol:
            print("The mesh generation reaches covergence!")
            break
            pass
        if ite > 50000:
            print(
                "The mesh generation not reaches covergence "
                + "within 50000 iterations! The current resdiual is "
            )
            print(err)
            break
            pass
        ite = ite + 1
    return x, y


def gen_e2vcg(x):
    nelem = (x.shape[0] - 1) * (x.shape[1] - 1)
    nelemx = x.shape[1] - 1
    nelemy = x.shape[0] - 1
    nelem = nelemx * nelemy
    nnx = x.shape[1]
    e2vcg = np.zeros([4, nelem])
    for j in range(nelemy):
        for i in range(nelemx):
            e2vcg[:, j * nelemx + i] = np.asarray(
                [j * nnx + i, j * nnx + i + 1, (j + 1) * nnx + i, (j + 1) * nnx + i + 1]
            )
    return e2vcg.astype("int")


def visualize2D(ax, x, y, u, colorbarPosition="vertical", colorlimit=None):
    xdg0 = np.vstack([x.flatten(order="C"), y.flatten(order="C")])
    udg0 = u.flatten(order="C")
    idx = np.asarray([0, 1, 3, 2])
    nelem = (x.shape[0] - 1) * (x.shape[1] - 1)
    nelemx = x.shape[1] - 1
    nelemy = x.shape[0] - 1
    nelem = nelemx * nelemy
    e2vcg0 = gen_e2vcg(x)
    udg_ref = udg0[e2vcg0]
    cmap = matplotlib.cm.coolwarm
    polygon_list = []
    for i in range(nelem):
        polygon_ = patches.Polygon(xdg0[:, e2vcg0[idx, i]].T)
        polygon_list.append(polygon_)
    polygon_ensemble = collections.PatchCollection(polygon_list, cmap=cmap, alpha=1)
    polygon_ensemble.set_edgecolor("face")
    polygon_ensemble.set_array(np.mean(udg_ref, axis=0))
    if colorlimit is None:
        pass
    else:
        polygon_ensemble.set_clim(colorlimit)
    ax.add_collection(polygon_ensemble)
    ax.set_xlim(np.min(xdg0[0, :]), np.max(xdg0[0, :]))
    ax.set_ylim(np.min(xdg0[1, :]), np.max(xdg0[1, :]))
    cbar = plt.colorbar(polygon_ensemble, orientation=colorbarPosition)
    return ax, cbar


class hcubeMesh(object):
    """docstring for hcubeMesh"""

    def __init__(
        self,
        left_x,
        left_y,
        right_x,
        right_y,
        low_x,
        low_y,
        up_x,
        up_y,
        h,
        plot_flag=False,
        save_flag=False,
        saveDir="./output/mesh.pdf",
        tol_mesh=1e-8,
        tol_joint=1e-6,
    ):
        self.h = h
        self.tol_mesh = tol_mesh
        self.tol_joint = tol_joint
        self.plot_flag = plot_flag
        self.save_flag = save_flag
        checkGeo(left_x, left_y, right_x, right_y, low_x, low_y, up_x, up_y, tol_joint)
        # Extract discretization info
        self.ny = left_x.shape[0]
        self.nx = up_x.shape[0]
        # Prellocate the physical domain
        # Left->Right->Low->Up
        self.x = np.zeros([self.ny, self.nx])
        self.y = np.zeros([self.ny, self.nx])
        self.x[:, 0] = left_x
        self.y[:, 0] = left_y
        self.x[:, -1] = right_x
        self.y[:, -1] = right_y
        self.x[0, :] = low_x
        self.y[0, :] = low_y
        self.x[-1, :] = up_x
        self.y[-1, :] = up_y
        self.x, self.y = ellipticMap(self.x, self.y, self.h, self.tol_mesh)
        # Define the ref domain
        eta, xi = np.meshgrid(
            np.linspace(0, self.ny - 1, self.ny),
            np.linspace(0, self.nx - 1, self.nx),
            sparse=False,
            indexing="ij",
        )
        self.xi = xi * h
        self.eta = eta * h
        fig = plt.figure()
        ax = plt.subplot(1, 2, 1)
        plotBC(ax, self.x, self.y)
        plotMesh(ax, self.x, self.y)
        setAxisLabel(ax, "p")
        ax.set_aspect("equal")
        ax.set_title("Physics Domain Mesh")
        ax = plt.subplot(1, 2, 2)
        plotBC(ax, self.xi, self.eta)
        plotMesh(ax, self.xi, self.eta)
        setAxisLabel(ax, "r")
        ax.set_aspect("equal")
        ax.set_title("Reference Domain Mesh")
        fig.tight_layout(pad=1)
        if save_flag:
            plt.savefig(saveDir, bbox_inches="tight")
        if plot_flag:
            plt.show()
        plt.close(fig)
        self.dxdxi = (self.x[1:-1, 2:] - self.x[1:-1, 0:-2]) / 2 / self.h
        self.dydxi = (self.y[1:-1, 2:] - self.y[1:-1, 0:-2]) / 2 / self.h
        self.dxdeta = (self.x[2:, 1:-1] - self.x[0:-2, 1:-1]) / 2 / self.h
        self.dydeta = (self.y[2:, 1:-1] - self.y[0:-2, 1:-1]) / 2 / self.h
        self.j = self.dxdxi * self.dydeta - self.dxdeta * self.dydxi
        self.j_inv = 1 / self.j

        dxdxi_ho_internal = (
            (
                -self.x[:, 4:]
                + 8 * self.x[:, 3:-1]
                - 8 * self.x[:, 1:-3]
                + self.x[:, 0:-4]
            )
            / 12
            / self.h
        )
        dydxi_ho_internal = (
            (
                -self.y[:, 4:]
                + 8 * self.y[:, 3:-1]
                - 8 * self.y[:, 1:-3]
                + self.y[:, 0:-4]
            )
            / 12
            / self.h
        )
        dxdeta_ho_internal = (
            (
                -self.x[4:, :]
                + 8 * self.x[3:-1, :]
                - 8 * self.x[1:-3, :]
                + self.x[0:-4, :]
            )
            / 12
            / self.h
        )
        dydeta_ho_internal = (
            (
                -self.y[4:, :]
                + 8 * self.y[3:-1, :]
                - 8 * self.y[1:-3, :]
                + self.y[0:-4, :]
            )
            / 12
            / self.h
        )

        dxdxi_ho_left = (
            (
                -11 * self.x[:, 0:-3]
                + 18 * self.x[:, 1:-2]
                - 9 * self.x[:, 2:-1]
                + 2 * self.x[:, 3:]
            )
            / 6
            / self.h
        )
        dxdxi_ho_right = (
            (
                11 * self.x[:, 3:]
                - 18 * self.x[:, 2:-1]
                + 9 * self.x[:, 1:-2]
                - 2 * self.x[:, 0:-3]
            )
            / 6
            / self.h
        )
        dydxi_ho_left = (
            (
                -11 * self.y[:, 0:-3]
                + 18 * self.y[:, 1:-2]
                - 9 * self.y[:, 2:-1]
                + 2 * self.y[:, 3:]
            )
            / 6
            / self.h
        )
        dydxi_ho_right = (
            (
                11 * self.y[:, 3:]
                - 18 * self.y[:, 2:-1]
                + 9 * self.y[:, 1:-2]
                - 2 * self.y[:, 0:-3]
            )
            / 6
            / self.h
        )

        dxdeta_ho_low = (
            (
                -11 * self.x[0:-3, :]
                + 18 * self.x[1:-2, :]
                - 9 * self.x[2:-1, :]
                + 2 * self.x[3:, :]
            )
            / 6
            / self.h
        )
        dxdeta_ho_up = (
            (
                11 * self.x[3:, :]
                - 18 * self.x[2:-1, :]
                + 9 * self.x[1:-2, :]
                - 2 * self.x[0:-3, :]
            )
            / 6
            / self.h
        )
        dydeta_ho_low = (
            (
                -11 * self.y[0:-3, :]
                + 18 * self.y[1:-2, :]
                - 9 * self.y[2:-1, :]
                + 2 * self.y[3:, :]
            )
            / 6
            / self.h
        )
        dydeta_ho_up = (
            (
                11 * self.y[3:, :]
                - 18 * self.y[2:-1, :]
                + 9 * self.y[1:-2, :]
                - 2 * self.y[0:-3, :]
            )
            / 6
            / self.h
        )

        self.dxdxi_ho = np.zeros(self.x.shape)
        self.dxdxi_ho[:, 2:-2] = dxdxi_ho_internal
        self.dxdxi_ho[:, 0:2] = dxdxi_ho_left[:, 0:2]
        self.dxdxi_ho[:, -2:] = dxdxi_ho_right[:, -2:]

        self.dydxi_ho = np.zeros(self.y.shape)
        self.dydxi_ho[:, 2:-2] = dydxi_ho_internal
        self.dydxi_ho[:, 0:2] = dydxi_ho_left[:, 0:2]
        self.dydxi_ho[:, -2:] = dydxi_ho_right[:, -2:]

        self.dxdeta_ho = np.zeros(self.x.shape)
        self.dxdeta_ho[2:-2, :] = dxdeta_ho_internal
        self.dxdeta_ho[0:2, :] = dxdeta_ho_low[0:2, :]
        self.dxdeta_ho[-2:, :] = dxdeta_ho_up[-2:, :]

        self.dydeta_ho = np.zeros(self.y.shape)
        self.dydeta_ho[2:-2, :] = dydeta_ho_internal
        self.dydeta_ho[0:2, :] = dydeta_ho_low[0:2, :]
        self.dydeta_ho[-2:, :] = dydeta_ho_up[-2:, :]

        self.j_ho = self.dxdxi_ho * self.dydeta_ho - self.dxdeta_ho * self.dydxi_ho
        self.j_inv_ho = 1 / self.j_ho

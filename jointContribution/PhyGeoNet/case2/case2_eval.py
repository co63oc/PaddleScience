import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import Ofpp
import tikzplotlib

os.environ["FLAGS_set_to_1d"] = "False"
import paddle  # noqa: E402
import paddle.nn as nn  # noqa: E402

sys.path.insert(0, "../")
from source import dataset  # noqa: E402
from source import model as source_model  # noqa: E402
from source import pyMesh  # noqa: E402
from source import readOF  # noqa: E402

h = 0.01
r = 0.5
R = 1
dtheta = 0
of_bcc_oord = Ofpp.parse_boundary_field("TemplateCase/30/C")
of_left_c = of_bcc_oord[b"left"][b"value"]
of_right_c = of_bcc_oord[b"right"][b"value"]

left_x = r * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
left_y = r * np.sin(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
right_x = R * np.cos(np.linspace(dtheta, 2 * np.pi - dtheta, 276))
right_y = R * np.sin(np.linspace(dtheta, 2 * np.pi - dtheta, 276))

low_x = np.linspace(left_x[0], right_x[0], 49)
low_y = low_x * 0 + np.sin(dtheta)
up_x = np.linspace(left_x[-1], right_x[-1], 49)
up_y = up_x * 0 - np.sin(dtheta)


ny = len(left_x)
nx = len(low_x)


my_mesh = pyMesh.hcubeMesh(
    left_x,
    left_y,
    right_x,
    right_y,
    low_x,
    low_y,
    up_x,
    up_y,
    h,
    False,
    True,
    tol_mesh=1e-10,
    tol_joint=0.01,
)

batch_size = 1
n_var_input = 1
n_var_output = 1
n_epochs = 1
lr = 0.001
ns = 1
nu = 0.01
model = source_model.USCNN(h, nx, ny, n_var_input, n_var_output)
state_dict = paddle.load("./output/1000.pth")
model.set_state_dict(state_dict)
criterion = nn.MSELoss()
optimizer = paddle.optimizer.Adam(
    parameters=model.parameters(), learning_rate=lr, weight_decay=0.0
)
pad_single_side = 1
udfpad = nn.Pad2D(
    [pad_single_side, pad_single_side, pad_single_side, pad_single_side], value=0.0
)

para_list = [1, 2, 3, 4, 5, 6, 7]
case_name = [
    "TemplateCase0",
    "TemplateCase1",
    "TemplateCase2",
    "TemplateCase3",
    "TemplateCase4",
    "TemplateCase5",
    "TemplateCase6",
]
of_v_sb = []
for name in case_name:
    of_pic = readOF.convertOFMeshToImage_StructuredMesh(
        nx, ny, name + "/30/C", [name + "/30/T"], [0, 1, 0, 1], 0.0, False
    )
    of_x = of_pic[:, :, 0]
    of_y = of_pic[:, :, 1]
    of_v = of_pic[:, :, 2]

    of_v_sb_Temp = np.zeros(of_v.shape)

    for i in range(nx):
        for j in range(ny):
            dist = (my_mesh.x[j, i] - of_x) ** 2 + (my_mesh.y[j, i] - of_y) ** 2
            idx_min = np.where(dist == dist.min())
            of_v_sb_Temp[j, i] = of_v[idx_min]
    of_v_sb.append(of_v_sb_Temp)


test_set = dataset.FixGeoDataset(para_list, my_mesh, of_v_sb)


def dfdx(f, dydeta, dydxi, j_inv):
    dfdxi_internal = (
        (
            -f[:, :, :, 4:]
            + 8 * f[:, :, :, 3:-1]
            - 8 * f[:, :, :, 1:-3]
            + f[:, :, :, 0:-4]
        )
        / 12
        / h
    )
    dfdxi_left = (
        (
            -11 * f[:, :, :, 0:-3]
            + 18 * f[:, :, :, 1:-2]
            - 9 * f[:, :, :, 2:-1]
            + 2 * f[:, :, :, 3:]
        )
        / 6
        / h
    )
    dfdxi_right = (
        (
            11 * f[:, :, :, 3:]
            - 18 * f[:, :, :, 2:-1]
            + 9 * f[:, :, :, 1:-2]
            - 2 * f[:, :, :, 0:-3]
        )
        / 6
        / h
    )
    dfdxi = paddle.concat(
        (dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3
    )

    dfdeta_internal = (
        (
            -f[:, :, 4:, :]
            + 8 * f[:, :, 3:-1, :]
            - 8 * f[:, :, 1:-3, :]
            + f[:, :, 0:-4, :]
        )
        / 12
        / h
    )
    dfdeta_low = (
        (
            -11 * f[:, :, 0:-3, :]
            + 18 * f[:, :, 1:-2, :]
            - 9 * f[:, :, 2:-1, :]
            + 2 * f[:, :, 3:, :]
        )
        / 6
        / h
    )
    dfdeta_up = (
        (
            11 * f[:, :, 3:, :]
            - 18 * f[:, :, 2:-1, :]
            + 9 * f[:, :, 1:-2, :]
            - 2 * f[:, :, 0:-3, :]
        )
        / 6
        / h
    )
    dfdeta = paddle.concat(
        (dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2
    )
    dfdx = j_inv * (dfdxi * dydeta - dfdeta * dydxi)
    return dfdx


def dfdy(f, dxdxi, dxdeta, j_inv):
    dfdxi_internal = (
        (
            -f[:, :, :, 4:]
            + 8 * f[:, :, :, 3:-1]
            - 8 * f[:, :, :, 1:-3]
            + f[:, :, :, 0:-4]
        )
        / 12
        / h
    )
    dfdxi_left = (
        (
            -11 * f[:, :, :, 0:-3]
            + 18 * f[:, :, :, 1:-2]
            - 9 * f[:, :, :, 2:-1]
            + 2 * f[:, :, :, 3:]
        )
        / 6
        / h
    )
    dfdxi_right = (
        (
            11 * f[:, :, :, 3:]
            - 18 * f[:, :, :, 2:-1]
            + 9 * f[:, :, :, 1:-2]
            - 2 * f[:, :, :, 0:-3]
        )
        / 6
        / h
    )
    dfdxi = paddle.concat(
        (dfdxi_left[:, :, :, 0:2], dfdxi_internal, dfdxi_right[:, :, :, -2:]), 3
    )

    dfdeta_internal = (
        (
            -f[:, :, 4:, :]
            + 8 * f[:, :, 3:-1, :]
            - 8 * f[:, :, 1:-3, :]
            + f[:, :, 0:-4, :]
        )
        / 12
        / h
    )
    dfdeta_low = (
        (
            -11 * f[:, :, 0:-3, :]
            + 18 * f[:, :, 1:-2, :]
            - 9 * f[:, :, 2:-1, :]
            + 2 * f[:, :, 3:, :]
        )
        / 6
        / h
    )
    dfdeta_up = (
        (
            11 * f[:, :, 3:, :]
            - 18 * f[:, :, 2:-1, :]
            + 9 * f[:, :, 1:-2, :]
            - 2 * f[:, :, 0:-3, :]
        )
        / 6
        / h
    )
    dfdeta = paddle.concat(
        (dfdeta_low[:, :, 0:2, :], dfdeta_internal, dfdeta_up[:, :, -2:, :]), 2
    )
    dfdy = j_inv * (dfdeta * dxdxi - dfdxi * dxdeta)
    return dfdy


velocity_magnitude_error_record = []
for i in range(len(para_list)):
    [
        para,
        coord,
        xi,
        eta,
        j,
        j_inv,
        dxdxi,
        dydxi,
        dxdeta,
        dydeta,
        truth,
    ] = pyMesh.to4DTensor(test_set[i])
    para = para.reshape((1, 1, para.shape[0], para.shape[1]))
    truth = truth.reshape((1, 1, truth.shape[0], truth.shape[1]))
    coord = coord.reshape((1, 2, coord.shape[2], coord.shape[3]))
    print("i=", str(i))
    output = model(para)
    output_pad = udfpad(output)
    output_v = output_pad[:, 0, :, :].reshape(
        (output_pad.shape[0], 1, output_pad.shape[2], output_pad.shape[3])
    )
    # Impose BC
    output_v[0, 0, -pad_single_side:, pad_single_side:-pad_single_side] = output_v[
        0, 0, 1:2, pad_single_side:-pad_single_side
    ]  # up outlet bc zero gradient
    output_v[0, 0, :pad_single_side, pad_single_side:-pad_single_side] = output_v[
        0, 0, -2:-1, pad_single_side:-pad_single_side
    ]  # down inlet bc
    output_v[0, 0, :, -pad_single_side:] = 0  # right wall bc
    output_v[0, 0, :, 0:pad_single_side] = para[0, 0, 0, 0]  # left  wall bc

    dvdx = dfdx(output_v, dydeta, dydxi, j_inv)
    d2vdx2 = dfdx(dvdx, dydeta, dydxi, j_inv)

    dvdy = dfdy(output_v, dxdxi, dxdeta, j_inv)
    d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, j_inv)
    # Calculate PDE Residual
    continuity = d2vdy2 + d2vdx2
    loss = criterion(continuity, continuity * 0)
    velocity_magnitude_error_record.append(
        paddle.sqrt(criterion(truth, output_v) / criterion(truth, truth * 0))
    )
    fig1 = plt.figure()
    xylabelsize = 20
    xytickssize = 20
    titlesize = 20
    ax = plt.subplot(1, 2, 1)
    _, cbar = pyMesh.visualize2D(
        ax,
        coord[0, 0, :, :].cpu().detach().numpy(),
        coord[0, 1, :, :].cpu().detach().numpy(),
        output_v[0, 0, :, :].cpu().detach().numpy(),
        "horizontal",
        [0, max(para_list)],
    )
    ax.set_aspect("equal")
    pyMesh.setAxisLabel(ax, "p")
    ax.set_title("PhyGeoNet " + r"$T$", fontsize=titlesize)
    ax.set_xlabel(xlabel=r"$x$", fontsize=xylabelsize)
    ax.set_ylabel(ylabel=r"$y$", fontsize=xylabelsize)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis="x", labelsize=xytickssize)
    ax.tick_params(axis="y", labelsize=xytickssize)
    cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7])
    cbar.ax.tick_params(labelsize=xytickssize)
    ax = plt.subplot(1, 2, 2)
    _, cbar = pyMesh.visualize2D(
        ax,
        coord[0, 0, :, :].cpu().detach().numpy(),
        coord[0, 1, :, :].cpu().detach().numpy(),
        truth[0, 0, :, :].cpu().detach().numpy(),
        "horizontal",
        [0, max(para_list)],
    )
    ax.set_aspect("equal")
    pyMesh.setAxisLabel(ax, "p")
    ax.set_title("FV " + r"$T$", fontsize=titlesize)
    ax.set_xlabel(xlabel=r"$x$", fontsize=xylabelsize)
    ax.set_ylabel(ylabel=r"$y$", fontsize=xylabelsize)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(axis="x", labelsize=xytickssize)
    ax.tick_params(axis="y", labelsize=xytickssize)
    cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7])
    cbar.ax.tick_params(labelsize=xytickssize)
    fig1.tight_layout(pad=1)
    fig1.savefig(os.path.join("output", "para" + str(i) + "T.pdf"), bbox_inches="tight")
    fig1.savefig(os.path.join("output", "para" + str(i) + "T.png"), bbox_inches="tight")
    plt.close(fig1)


v_error_numpy = np.asarray(
    [i.cpu().detach().numpy() for i in velocity_magnitude_error_record]
)
plt.figure()
plt.plot(np.asarray(para_list), v_error_numpy, "-x", label="Temperature Error")
plt.legend()
plt.xlabel("Inner circle temprature")
plt.ylabel("Error")
plt.savefig("./output/Error.pdf", bbox_inches="tight")
tikzplotlib.save("./output/Error.tikz")

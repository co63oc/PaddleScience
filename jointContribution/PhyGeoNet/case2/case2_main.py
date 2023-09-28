import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import Ofpp

os.environ["FLAGS_set_to_1d"] = "False"
import paddle  # noqa: E402
import paddle.nn as nn  # noqa: E402

sys.path.insert(0, "../")
from source import dataset  # noqa: E402
from source import model as source_model  # noqa: E402
from source import pyMesh  # noqa: E402
from source import readOF  # noqa: E402

os.makedirs("output", exist_ok=True)
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
    True,
    True,
    tol_mesh=1e-10,
    tol_joint=0.01,
)
batch_size = 2
n_var_input = 1
n_var_output = 1
n_epochs = 1000
lr = 0.001
ns = 1
nu = 0.01
model = source_model.USCNN(h, nx, ny, n_var_input, n_var_output)
criterion = nn.MSELoss()
optimizer = paddle.optimizer.Adam(
    parameters=model.parameters(), learning_rate=lr, weight_decay=0.0
)
pad_single_side = 1
udfpad = nn.Pad2D(
    [pad_single_side, pad_single_side, pad_single_side, pad_single_side], value=0
)
para_list = [1, 7]
case_name = ["TemplateCase0", "TemplateCase6"]
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
train_set = dataset.FixGeoDataset(para_list, my_mesh, of_v_sb)
training_data_loader = paddle.io.DataLoader(dataset=train_set, batch_size=batch_size)


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


def train(epoch):
    m_res = 0
    e_v = 0
    for _, batch in enumerate(training_data_loader):
        [
            para,
            coord,
            _,
            _,
            _,
            j_inv,
            dxdxi,
            dydxi,
            dxdeta,
            dydeta,
            truth,
        ] = pyMesh.to4DTensor(batch)
        optimizer.clear_grad()
        output = model(para)
        output_pad = udfpad(output)
        output_v = output_pad[:, 0, :, :].reshape(
            (output_pad.shape[0], 1, output_pad.shape[2], output_pad.shape[3])
        )
        for j in range(batch_size):
            # Impose BC
            output_v[
                j, 0, -pad_single_side:, pad_single_side:-pad_single_side
            ] = output_v[j, 0, 1:2, pad_single_side:-pad_single_side]
            output_v[
                j, 0, :pad_single_side, pad_single_side:-pad_single_side
            ] = output_v[j, 0, -2:-1, pad_single_side:-pad_single_side]
            output_v[j, 0, :, -pad_single_side:] = 0
            output_v[j, 0, :, 0:pad_single_side] = para[j, 0, 0, 0]
        dvdx = dfdx(output_v, dydeta, dydxi, j_inv)
        d2vdx2 = dfdx(dvdx, dydeta, dydxi, j_inv)
        dvdy = dfdy(output_v, dxdxi, dxdeta, j_inv)
        d2vdy2 = dfdy(dvdy, dxdxi, dxdeta, j_inv)
        continuity = d2vdy2 + d2vdx2
        loss = criterion(continuity, continuity * 0)
        loss.backward()
        optimizer.step()
        loss_mass = criterion(continuity, continuity * 0)
        m_res += loss_mass.item()
        e_v = e_v + paddle.sqrt(
            criterion(truth, output_v) / criterion(truth, truth * 0)
        )
        if epoch % 1000 == 0 or epoch % n_epochs == 0:
            paddle.save(model.state_dict(), os.path.join("output", str(epoch) + ".pth"))
            for j in range(batch_size):
                fig1 = plt.figure()
                ax = plt.subplot(1, 2, 1)
                pyMesh.visualize2D(
                    ax,
                    coord[0, 0, :, :].cpu().detach().numpy(),
                    coord[0, 1, :, :].cpu().detach().numpy(),
                    output_v[j, 0, :, :].cpu().detach().numpy(),
                )
                ax.set_aspect("equal")
                pyMesh.setAxisLabel(ax, "p")
                ax.set_title("CNN " + r"$T$")
                ax = plt.subplot(1, 2, 2)
                pyMesh.visualize2D(
                    ax,
                    coord[0, 0, :, :].cpu().detach().numpy(),
                    coord[0, 1, :, :].cpu().detach().numpy(),
                    truth[j, 0, :, :].cpu().detach().numpy(),
                )
                ax.set_aspect("equal")
                pyMesh.setAxisLabel(ax, "p")
                ax.set_title("FV " + r"$T$")
                fig1.tight_layout(pad=1)
                fig1.savefig(
                    os.path.join(
                        "output", "Epoch" + str(epoch) + "para" + str(j) + "T.pdf"
                    ),
                    bbox_inches="tight",
                )
                plt.close(fig1)
    print("Epoch is ", epoch)
    print("m_res Loss is", (m_res / len(training_data_loader)))
    print("e_v Loss is", (e_v / len(training_data_loader)))
    return (m_res / len(training_data_loader)), (e_v / len(training_data_loader))


m_res = []
ev_list = []
total_start_time = time.time()
for epoch in range(1, n_epochs + 1):
    mres, ev = train(epoch)
    m_res.append(mres)
    ev_list.append(ev)
time_spent = time.time() - total_start_time
plt.figure()
plt.plot(m_res, "-*", label="Equation Residual")
plt.xlabel("Epoch")
plt.ylabel("Residual")
plt.legend()
plt.yscale("log")
plt.savefig("./output/convergence.pdf", bbox_inches="tight")
plt.figure()
plt.plot(ev_list, "-x", label=r"$e_v$")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()
plt.yscale("log")
plt.savefig("./output/error.pdf", bbox_inches="tight")
ev_list = np.asarray(ev_list)
m_res = np.asarray(m_res)
np.savetxt("./output/ev_list.txt", ev_list)
np.savetxt("./output/m_res.txt", m_res)
np.savetxt("./output/time_spent.txt", np.zeros([2, 2]) + time_spent)

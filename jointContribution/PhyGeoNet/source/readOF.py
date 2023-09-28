"""
This is for cases read openfoam result on a squared mesh 2D
"""

import matplotlib.pyplot as plt
import numpy as np

from .foamFileOperation import readScalarFromFile
from .foamFileOperation import readVectorFromFile


def convertOFMeshToImage(mesh_file, filename, ext, mri_level=0, plot_flag=True):
    title = ["x", "y"]
    of_vector = None
    of_scalar = None
    for i in range(len(filename)):
        if filename[i][-1] == "U":
            of_vector = readVectorFromFile(filename[i])
            title.append("u")
            title.append("v")
        elif filename[i][-1] == "p":
            of_scalar = readScalarFromFile(filename[i])
            title.append("p")
        elif filename[i][-1] == "T":
            of_scalar = readScalarFromFile(filename[i])
            title.append("T")
        elif filename[i][-1] == "f":
            of_scalar = readScalarFromFile(filename[i])
            title.append("f")
        else:
            print("Variable name is not clear")
            exit()
    n_var = len(title)
    of_mesh = readVectorFromFile(mesh_file)
    ng = of_mesh.shape[0]
    of_case = np.zeros([ng, n_var])
    of_case[:, 0:2] = np.copy(of_mesh[:, 0:2])
    if of_vector is not None and of_scalar is not None:
        # TODO: Undefined name `foamFileAddNoise`
        # if mri_level > 1e-16:
        #     of_vector = foamFileAddNoise.addMRINoise(of_vector, mri_level)
        of_case[:, 2:4] = np.copy(of_vector[:, 0:2])
        of_case[:, 4] = np.copy(of_scalar)
    elif of_scalar is not None:
        of_case[:, 2] = np.copy(of_scalar)
    row = int(np.sqrt(ng))
    of_pic = np.reshape(of_case, (row, row, n_var), order="C")
    if plot_flag:
        for i in range(len(title)):
            fig, ax = plt.subplots()
            im = ax.imshow(
                of_pic[:, :, i],
                interpolation="bicubic",
                cmap="coolwarm",  # cm.RdYlGn,
                origin="lower",
                extent=ext,
                vmax=of_pic[:, :, i].max(),
                vmin=of_pic[:, :, i].min(),
            )
            plt.xlabel("x")
            plt.ylabel("y")
            fig.colorbar(im)
            plt.title(title[i])
            plt.savefig(title[i] + ".pdf", bbox_inches="tight")
    return of_pic


def convertOFMeshToImage_StructuredMesh(
    nx, ny, mesh_file, filename, ext, mri_level=0, plot_flag=True
):
    title = ["x", "y"]
    of_vector = None
    of_scalar = None
    for i in range(len(filename)):
        if filename[i][-1] == "U":
            of_vector = readVectorFromFile(filename[i])
            title.append("u")
            title.append("v")
        elif filename[i][-1] == "p":
            of_scalar = readScalarFromFile(filename[i])
            title.append("p")
        elif filename[i][-1] == "T":
            of_scalar = readScalarFromFile(filename[i])
            title.append("T")
        elif filename[i][-1] == "f":
            of_scalar = readScalarFromFile(filename[i])
            title.append("f")
        else:
            print("Variable name is not clear")
            exit()
    n_var = len(title)
    of_mesh = readVectorFromFile(mesh_file)
    ng = of_mesh.shape[0]
    of_case = np.zeros([ng, n_var])
    of_case[:, 0:2] = np.copy(of_mesh[:, 0:2])
    if of_vector is not None and of_scalar is not None:
        # TODO: Undefined name `foamFileAddNoise`
        # if mri_level > 1e-16:
        #     of_vector = foamFileAddNoise.addMRINoise(of_vector, mri_level)
        of_case[:, 2:4] = np.copy(of_vector[:, 0:2])
        of_case[:, 4] = np.copy(of_scalar)
    elif of_scalar is not None:
        of_case[:, 2] = np.copy(of_scalar)
    of_pic = np.reshape(of_case, (ny, nx, n_var), order="F")
    if plot_flag:
        pass
    return of_pic


if __name__ == "__main__":
    convertOFMeshToImage(
        "./NS10000/0/C", "./NS10000/65/U", "./NS10000/65/p", [0, 1, 0, 1], 0.0, False
    )

"""
    convertOFMeshToImage('./result/preProcessing/highFidelityCases/TemplateCase-tmp_1.0/0/C',
                         './result/preProcessing/highFidelityCases/TemplateCase-tmp_1.0/100/U',
                         './result/preProcessing/highFidelityCases/TemplateCase-tmp_1.0/100/p',
                         [0,1,0,1],True)
                         """

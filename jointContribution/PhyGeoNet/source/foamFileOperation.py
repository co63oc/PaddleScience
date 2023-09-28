import re

import numpy as np

global unitTest
unitTest = False


def readVectorFromFile(u_file):
    """readVectorFromFile.

    Args:
        u_file (str): The directory path of OpenFOAM vector file (e.g., velocity).

    Returns:
        np.array: Matrix of vector.
    """
    res_mid = extractVector(u_file)
    fout = open("./output/Utemp", "w")
    glob_pattern = res_mid.group()
    glob_pattern = re.sub(r"\(", "", glob_pattern)
    glob_pattern = re.sub(r"\)", "", glob_pattern)
    fout.write(glob_pattern)
    fout.close()
    vector = np.loadtxt("./output/Utemp")
    return vector


def readScalarFromFile(file_name):
    """readScalarFromFile.

    Args:
        file_name (str): The file name of OpenFOAM scalar field.

    Returns:
        np.array: a vector of scalar field.
    """
    res_mid = extractScalar(file_name)

    # write it in Tautemp
    fout = open("./output/temp.txt", "w")
    glob_patternx = res_mid.group()
    glob_patternx = re.sub(r"\(", "", glob_patternx)
    glob_patternx = re.sub(r"\)", "", glob_patternx)
    fout.write(glob_patternx)
    fout.close()
    scalarVec = np.loadtxt("./output/temp.txt")
    return scalarVec


################################################ Regular Expression #####################################################


def extractVector(vector_file):
    """Function is using regular expression select Vector value out.

    Args:
        vector_file (str): The directory path of file.

    Returns:
        re.Pattern: the U as (Ux1,Uy1,Uz1);(Ux2,Uy2,Uz2);........
    """
    fin = open(vector_file, "r")  # need consider directory
    line = fin.read()  # line is U file to read
    fin.close()
    ### select U as (X X X)pattern (Using regular expression)
    pattern_mid = re.compile(
        r"""
	(
	\(                                                   # match(
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	\)                                                   # match )
	\n                                                   # match next line
	)+                                                   # search greedly
	""",
        re.DOTALL | re.VERBOSE,
    )
    res_mid = pattern_mid.search(line)
    return res_mid


def extractScalar(scalar_file):
    """subFunction of readTurbStressFromFile,
    Using regular expression to select scalar value out.

    Args:
        scalar_file (str): The directory path of file of scalar.

    Returns:
        re.Pattern: scalar selected, you need use res_mid.group() to see the content.
    """
    fin = open(scalar_file, "r")  # need consider directory
    line = fin.read()  # line is k file to read
    fin.close()
    ### select k as ()pattern (Using regular expression)
    pattern_mid = re.compile(
        r"""
		\(                                                   # match"("
		\n                                                   # match next line
		(
		[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
		\n                                                   # match next line
		)+                                                   # search greedly
		\)                                                   # match")"
	""",
        re.DOTALL | re.VERBOSE,
    )
    res_mid = pattern_mid.search(line)
    return res_mid

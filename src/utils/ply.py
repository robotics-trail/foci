import plyfile as ply
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.spatial.transform import Rotation as R


def extract_splat_data(ply_file):

    plydata = ply.PlyData.read(ply_file)
    splat = plydata['vertex']
    
    means = np.vstack([splat['x'], splat['y'], splat['z']]).T

    # covarianza por defecto (identidad)
    covs = np.array([np.eye(3) for _ in range(len(splat))])

    # colores normalizados a [0,1]
    colors = np.vstack([splat['red'], splat['green'], splat['blue']]).T / 255.0

    # opacidades por defecto (todas 1)
    opacities = np.ones((len(splat), 1))

    return means, covs, colors, opacities
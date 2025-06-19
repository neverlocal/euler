"""
Conversion between SU2 rotations (single-qubit unitaries) and the coresponding
SO3 rotations (actions of those unitaries on the Bloch sphere).
"""
from typing import Any, TypeAlias
from euler import angles_xzx
import numpy as np
from .utils import Float, RotMatrix

# tuple[int, ...] used for all shapes to fix regression in Numpy 2.2.6 typing.
# Seems to be fine in Numpy 2.3, but that's currently not supported by Numba.

SU2Matrix: TypeAlias = np.ndarray[
    tuple[int, ...], np.dtype[np.complexfloating[Any]]
]
"""Type alias for 2-by-2 SU2 matrices."""

def su2_to_so3(mat: SU2Matrix, tol: Float = 1e-8) -> RotMatrix:
    """
    Converts an SU2 rotation to the corresponding SO3 rotation.
    """
    r, s  = abs(mat[0,0]), abs(mat[0,1])
    r2, s2 = r*r, s*s
    if r2 < tol:
        global_phase = np.sqrt(complex(-mat[1,0]*mat[0,1]/s2))
    else:
        global_phase = np.sqrt(complex(mat[0,0]*mat[1,1]/r2))
    mat = mat/global_phase
    r2s2 = r2*s2
    if abs(mat[1,1]) < tol:
        exp_i2theta = 1.0
    else:
        exp_i2theta  = mat[0,0]/mat[1,1]
    cos_2theta = (exp_i2theta+1/exp_i2theta)/2
    sin_2theta = (exp_i2theta-1/exp_i2theta)/2j
    if abs(mat[0,1]) < tol:
        exp_i2phi = 1.0
    else:
        exp_i2phi = -mat[1,0]/mat[0,1]
    cos_2phi = (exp_i2phi+1/exp_i2phi)/2
    sin_2phi = (exp_i2phi-1/exp_i2phi)/2j
    rs_exp_add = -mat[0,0]*mat[1,0]
    if abs(rs_exp_add) < tol:
        rs_cos_add = 0.0
        rs_sin_add = 0.0
    else:
        rs_cos_add = (rs_exp_add+r2s2/rs_exp_add)/2
        rs_sin_add = (rs_exp_add-r2s2/rs_exp_add)/2j
    rs_exp_sub = mat[0,0]*mat[0,1]
    if abs(rs_exp_sub) < tol:
        rs_cos_sub = 0.0
        rs_sin_sub = 0.0
    else:
        rs_cos_sub = (rs_exp_sub+r2s2/rs_exp_sub)/2
        rs_sin_sub = (rs_exp_sub-r2s2/rs_exp_sub)/2j
    return np.array([
        [r2*cos_2theta-s2*cos_2phi, r2*sin_2theta-s2*sin_2phi, -2*rs_cos_sub],
        [-r2*sin_2theta-s2*sin_2phi, r2*cos_2theta+s2*cos_2phi, 2*rs_sin_sub],
        [2*rs_cos_add, 2*rs_sin_add, r2-s2],
    ]).real.astype(np.float64)

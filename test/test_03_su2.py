from itertools import product
from typing import Any
import pytest
import numpy as np
from euler.utils import Float, RotMatrix, AxisTriple, AXIS_TRIPLES
from euler.matrix import matrix
from euler.su2 import so3_to_su2, su2_to_so3, SU2Matrix

def normalise_phase(
    array: np.ndarray[tuple[int, ...], np.dtype[np.complexfloating[Any]]],
    *,
    tol: float = 1e-8,
) -> None:
    """Normalises the phase of a complex array (in place)."""
    assert tol > 0
    idx = np.argmax(abs(array) >= tol)
    val = array.flatten()[idx]
    if (aval := abs(val)) >= tol:
        array *= aval / val


def rot_x(a: Float) -> RotMatrix:
    sa, ca = np.sin(a), np.cos(a)
    return np.array([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa, ca],
    ], dtype=np.float64)

def rot_y(a: Float) -> RotMatrix:
    sa, ca = np.sin(a), np.cos(a)
    return np.array([
        [ca, 0, sa],
        [0, 1, 0],
        [-sa, 0, ca],
    ], dtype=np.float64)

def rot_z(a: Float) -> RotMatrix:
    sa, ca = np.sin(a), np.cos(a)
    return np.array([
        [ca, -sa, 0],
        [sa, ca, 0],
        [0, 0, 1],
    ], dtype=np.float64)

ROT = {
    "x": rot_x,
    "y": rot_y,
    "z": rot_z
}

def su2_rot_x(a: Float) -> SU2Matrix:
    sa, ca = np.sin(a/2), np.cos(a/2)
    return np.array([
        [ca, -1j*sa],
        [-1j*sa, ca],
    ], dtype=np.complex128)

def su2_rot_y(a: Float) -> SU2Matrix:
    sa, ca = np.sin(a/2), np.cos(a/2)
    return np.array([
        [ca, -sa],
        [sa, ca],
    ], dtype=np.complex128)

def su2_rot_z(a: Float) -> SU2Matrix:
    return np.array([
        [1, 0],
        [0, np.exp(1j*a)]
    ], dtype=np.complex128)

SU2_ROT = {
    "x": su2_rot_x,
    "y": su2_rot_y,
    "z": su2_rot_z
}

NUM_ANGLES = 17

assert (NUM_ANGLES-1)%4 == 0, "Tested angles must include 0, pi/2, pi and -pi/2."

@pytest.mark.parametrize("p", AXIS_TRIPLES)
def test_su2_to_so3(p: AxisTriple) -> None:
    for a, b, c in product(np.linspace(0, 2*np.pi, NUM_ANGLES), repeat=3):
        su2_mat = SU2_ROT[p[0]](a) @ SU2_ROT[p[1]](b) @ SU2_ROT[p[2]](c)
        so3_mat = matrix(p, a, b, c)
        converted = su2_to_so3(su2_mat)
        assert np.allclose(so3_mat, converted)

@pytest.mark.parametrize("p", AXIS_TRIPLES)
def test_so3_to_su2(p: AxisTriple) -> None:
    for a, b, c in product(np.linspace(0, 2*np.pi, NUM_ANGLES), repeat=3):
        su2_mat = SU2_ROT[p[0]](a) @ SU2_ROT[p[1]](b) @ SU2_ROT[p[2]](c)
        so3_mat = matrix(p, a, b, c)
        converted = so3_to_su2(so3_mat)
        normalise_phase(su2_mat)
        normalise_phase(converted)
        assert np.allclose(su2_mat, converted)

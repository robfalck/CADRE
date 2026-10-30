"""
Smoothed interpolation tables for CADRE: solar exposed area, solar cell voltage and antenna gain.

The legacy model did not interpolate its raw tables. It fit each with MBI, a least-squares
tensor-product B-spline with a few control points per axis, and evaluated the fit; the fits
differ from the raw tables by up to tens of percent (the raw ray-traced areas are noisy). This
module reproduces those fits with SciPy (``MBIFit``, which matches the stored legacy MBI
outputs to MBI's own solver tolerance) and samples each on a grid aligned with the fit's knots,
so a grid interpolator such as om4's ``StructuredMetaModelComp`` with method ``'cubic'``
reproduces the fit (ADL-007).

Each ``*_table()`` returns ``(grid, values)``: a dict of strictly increasing axes keyed by input
name, and the values on that grid, with any value shape trailing. The tables are built on first
use (about a second in all) and cached; the arrays are read-only.
"""
import functools
import os

import numpy as np
from scipy.interpolate import BSpline, NdBSpline
from scipy.optimize import brentq

DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class MBIFit:
    """
    The least-squares tensor-product B-spline fit of MBI, which the legacy model evaluated.

    MBI fits each axis parametrically. Along an axis of ``n`` points with ``m`` control points
    and order ``k`` (degree ``k - 1``), the knots are open and uniform on ``t`` in [0, 1], and
    data point ``l`` sits at the ``t`` where the spline with uniformly spaced control points
    equals ``l / (n - 1)``. The values are fit in least squares at those parameters, and so is
    the axis itself, ``x(t)``, with its end control points pinned to the end of the axis. A query
    ``x`` is evaluated at ``t = x^-1(x)``.

    MBI solved the normal equations by conjugate gradients; this solves the separable least
    squares problem directly, so the two differ by CG's tolerance (about 1e-4 of the range).

    Parameters
    ----------
    values : ndarray
        Values on the grid, one leading dimension per axis followed by any value shape.
    axes : list of ndarray
        The grid axes, each strictly increasing.
    num_cp : list of int
        Control points per axis (MBI's ``ms``).
    order : list of int
        B-spline order per axis (MBI's ``ks``); the degree is one less.
    """

    def __init__(self, values, axes, num_cp, order):
        self.value_shape = values.shape[len(axes):]
        self.knots = []
        self.x_of_t = []
        bases = []
        for x, m, k in zip(axes, num_cp, order):
            degree = k - 1
            knots = np.r_[np.zeros(k), np.arange(1, m - k + 1) / (m - k + 1), np.ones(k)]
            uniform = BSpline(knots, np.linspace(0.0, 1.0, m), degree)
            t = np.array([brentq(lambda s: uniform(s) - p, 0.0, 1.0) if 0.0 < p < 1.0 else p
                          for p in np.linspace(0.0, 1.0, x.size)])
            basis = BSpline.design_matrix(t, knots, degree).toarray()
            cx = np.linalg.lstsq(basis, x, rcond=None)[0]
            cx[0], cx[-1] = x[0], x[-1]
            self.knots.append(knots)
            self.x_of_t.append(BSpline(knots, cx, degree))
            bases.append(basis)

        coef = values.reshape(*[x.size for x in axes], -1)
        for axis, basis in enumerate(bases):
            coef = np.moveaxis(np.tensordot(np.linalg.pinv(basis), np.moveaxis(coef, axis, 0), axes=1),
                               0, axis)
        self.spline = NdBSpline(tuple(self.knots), coef, tuple(k - 1 for k in order))

    def _t(self, axis, x):
        """Invert the monotone ``x(t)`` of one axis: a table lookup, then Newton steps."""
        x_of_t = self.x_of_t[axis]
        dx_dt = x_of_t.derivative()
        samples = np.linspace(0.0, 1.0, 4001)
        t = np.interp(x, x_of_t(samples), samples)
        for _ in range(4):
            t = np.clip(t - (x_of_t(t) - x) / dx_dt(t), 0.0, 1.0)
        return t

    def __call__(self, points):
        """
        Evaluate the fit at ``points``, shape ``(num_points, num_axes)``.

        Returns an array of shape ``(num_points, *value_shape)``.
        """
        points = np.asarray(points, dtype=float)
        t = np.column_stack([self._t(axis, points[:, axis]) for axis in range(points.shape[1])])
        return self.spline(t).reshape(points.shape[0], *self.value_shape)

    def knot_grid(self, axis, per_span):
        """
        Grid along ``axis`` with a point at every knot and ``per_span`` intervals per knot span.

        The fit is smooth within a knot span but its highest derivative jumps at a knot, so
        a grid interpolator is accurate only if the knots are grid points.
        """
        knots = np.unique(self.knots[axis])
        t = np.concatenate([np.linspace(a, b, per_span + 1)[:-1]
                            for a, b in zip(knots[:-1], knots[1:])] + [knots[-1:]])
        return self.x_of_t[axis](t)

    def sample(self, names, per_span):
        """
        Sample the fit on a knot-aligned grid.

        Parameters
        ----------
        names : list of str
            Input names for the axes, in order.
        per_span : list of int
            Intervals per knot span along each axis.

        Returns
        -------
        dict, ndarray
            The grid axes keyed by name, and the values on the grid.
        """
        axes = [self.knot_grid(axis, s) for axis, s in enumerate(per_span)]
        mesh = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, len(axes))
        values = self(mesh).reshape(*[x.size for x in axes], *self.value_shape)
        return dict(zip(names, axes)), values


def _read_only(grid, values):
    for array in (*grid.values(), values):
        array.flags.writeable = False
    return grid, values


def raw_exposed_area_table():
    """
    The raw exposed area of each solar cell, as ``Solar_ExposedArea`` read it.

    Returns
    -------
    dict, ndarray
        Axes ``finAngle`` (10, rad), ``azimuth`` (73, rad) and ``elevation`` (37, rad), and the
        exposed area in m**2 of shape ``(10, 73, 37, 7, 12)``: cell, then panel.
    """
    raw1 = np.genfromtxt(os.path.join(DATA_DIR, 'Solar', 'Area10.txt'))
    raw2 = np.loadtxt(os.path.join(DATA_DIR, 'Solar', 'Area_all.txt'))
    na, nz, ne = 10, 73, 37

    angle = raw1[:na].copy()
    azimuth = raw1[na:na + nz].copy()
    # The last azimuth is overwritten, and the elevations start in its place.
    elevation = raw1[na + nz - 1:na + nz - 1 + ne].copy()
    angle[0], angle[-1] = 0.0, np.pi / 2.0
    azimuth[0], azimuth[-1] = 0.0, 2.0 * np.pi
    elevation[0], elevation[-1] = 0.0, np.pi

    area = np.zeros((na, nz, ne, 7, 12))
    size = na * nz * ne
    for p in range(12):
        for c in range(7):
            area[:, :, :, c, p] = raw2[7 * p + c][119:119 + size].reshape((na, nz, ne))
    return {'finAngle': angle, 'azimuth': azimuth, 'elevation': elevation}, area


def raw_cell_voltage_table():
    """
    The raw solar cell voltage, as ``Power_CellVoltage`` read it.

    Returns
    -------
    dict, ndarray
        Axes ``T`` (20, degK), ``A`` (20, the illuminated area in m**2) and ``I`` (200, A), and
        the cell voltage in V of shape ``(20, 20, 200)``.
    """
    dat = np.genfromtxt(os.path.join(DATA_DIR, 'Power', 'curve.dat'))
    nT, nA, nI = (int(n) for n in dat[:3])
    T = dat[3:3 + nT]
    A = dat[3 + nT:3 + nT + nA]
    I = dat[3 + nT + nA:3 + nT + nA + nI]
    V = dat[3 + nT + nA + nI:].reshape((nT, nA, nI), order='F')
    return {'T': T, 'A': A, 'I': I}, V


def raw_antenna_gain_table():
    """
    The raw transmitter gain, as ``Comm_GainPattern`` read it (converted from dB).

    Returns
    -------
    dict, ndarray
        Axes ``azimuth`` and ``elevation`` (361 each, rad, on [0, 2 pi]), and the gain of shape
        ``(361, 361)``.
    """
    gain_db = np.genfromtxt(os.path.join(DATA_DIR, 'Comm', 'Gain.txt'))
    gain = (10 ** (gain_db / 10.0)).reshape((361, 361), order='F')
    angles = np.linspace(0, 2 * np.pi, 361)
    return {'azimuth': angles, 'elevation': angles.copy()}, gain


@functools.cache
def exposed_area_fit():
    """The legacy fit of the exposed area: control points [4, 10, 8], cubic."""
    grid, area = raw_exposed_area_table()
    return MBIFit(area, list(grid.values()), [4, 10, 8], [4, 4, 4])


@functools.cache
def cell_voltage_fit():
    """The legacy fit of the cell voltage: control points [6, 6, 15], quadratic."""
    grid, V = raw_cell_voltage_table()
    return MBIFit(V, list(grid.values()), [6, 6, 15], [3, 3, 3])


@functools.cache
def antenna_gain_fit():
    """The legacy fit of the antenna gain: control points [15, 15], cubic."""
    grid, gain = raw_antenna_gain_table()
    return MBIFit(gain, list(grid.values()), [15, 15], [4, 4])


@functools.cache
def exposed_area_table():
    """
    The exposed area fit, sampled for cubic interpolation.

    Returns
    -------
    dict, ndarray
        Axes ``finAngle`` (9), ``azimuth`` (57) and ``elevation`` (41), in rad, and the exposed
        area in m**2 of shape ``(9, 57, 41, 7, 12)``. Cubic interpolation of it matches the fit
        to about 1e-4 of the range.
    """
    # The fin angle axis has a single knot span, so it takes enough intervals for a cubic.
    return _read_only(*exposed_area_fit().sample(['finAngle', 'azimuth', 'elevation'], [8, 8, 8]))


@functools.cache
def cell_voltage_table():
    """
    The cell voltage fit, sampled for cubic interpolation.

    Returns
    -------
    dict, ndarray
        Axes ``T`` (65, degK), ``A`` (65, m**2) and ``I`` (209, A), and the cell voltage in V of
        shape ``(65, 65, 209)``. The fit is quadratic, so its second derivative jumps at the
        knots and the grid is finer than the others; cubic interpolation matches the fit to
        about 4e-4 of the range.
    """
    return _read_only(*cell_voltage_fit().sample(['T', 'A', 'I'], [16, 16, 16]))


@functools.cache
def antenna_gain_table():
    """
    The antenna gain fit, sampled for cubic interpolation.

    Returns
    -------
    dict, ndarray
        Axes ``azimuth`` and ``elevation`` (97 each, rad, on [0, 2 pi]), and the gain of shape
        ``(97, 97)``. Cubic interpolation of it matches the fit to about 1e-4 of the range.
    """
    return _read_only(*antenna_gain_fit().sample(['azimuth', 'elevation'], [8, 8]))

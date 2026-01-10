# butterfly/thermal_bf.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False

import sys
import time
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange

cnp.import_array()

cdef extern from "thrmlLib.h":
    void conductionQ(int nz, double z[], double dt, double Qn, double Qnp1,
                     double T[], double ti[], double rhoc[],
                     double emiss, double Fgeotherm, double *Fsurf) nogil

    void conductionT(int nz, double z[], double dt, double T[], double Tsurf,
                     double Tsurfp1, double ti[], double rhoc[],
                     double Fgeotherm, double *Fsurf) nogil

    double flux_noatm(double R, double decl, double latitude, double HA,
                      double surfaceSlope, double azFac) nogil

    void heatflux_from_temperature(int nz, double z[], double T[],
                                   double k[], double H[]) nogil

    void tridag(double a[], double b[], double c[], double r[],
                double T[], unsigned long nz) nogil


cdef double SIGSB = 5.6704e-8


cdef class BfPccThermalModel1D:
    cdef:
        int _num_layers
        int _num_faces
        int _num_layers_interior

        double _current_time

        double[::1] _emiss
        double[::1] _z
        double[::1] _ti
        double[::1] _rhoc
        double[::1] _Fgeotherm
        double[::1] _Qprev
        double[::1] _Fsurf
        double[:, ::1] _T

        str _bcond

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def num_faces(self):
        return self._num_faces

    @property
    def num_layers_interior(self):
        return self._num_layers_interior

    @property
    def current_time(self):
        return self._current_time

    @property
    def emiss(self):
        return np.asarray(self._emiss)

    @property
    def z(self):
        return np.asarray(self._z)

    @property
    def ti(self):
        return np.asarray(self._ti)

    @property
    def rhoc(self):
        return np.asarray(self._rhoc)

    @property
    def Fgeotherm(self):
        return np.asarray(self._Fgeotherm)

    @property
    def Qprev(self):
        return np.asarray(self._Qprev)

    @property
    def Fsurf(self):
        return np.asarray(self._Fsurf)

    @property
    def T(self):
        return np.asarray(self._T)

    @property
    def bcond(self):
        return self._bcond

    def __cinit__(self,
                  double[::1] z,
                  object T0_in,
                  double[::1] ti,
                  double[::1] rhoc,
                  double[::1] emiss,
                  double[::1] Fgeotherm,
                  double[::1] Q0,
                  double[::1] Tsurfprev,
                  bcond='Q'):
        if bcond not in {'Q', 'T'}:
            raise ValueError('bcond should be "Q" or "T"')
        self._bcond = bcond

        cdef cnp.ndarray[cnp.double_t, ndim=2] T0_arr = np.ascontiguousarray(T0_in, dtype=np.float64)
        cdef double[:, ::1] T0 = T0_arr

        self._num_faces = T0.shape[0]
        self._num_layers = z.size
        self._num_layers_interior = self._num_layers - 1
        self._current_time = 0

        for i in range(self._num_faces):
            if emiss[i] < 0.0 or emiss[i] > 1.0:
                raise ValueError('emissivity values ("emiss") should be in range [0, 1]')
        if emiss.ndim != 1 or emiss.size != self.num_faces:
            raise ValueError('"emiss" should be a 1D array with length == T0.shape[0]')
        self._emiss = emiss

        if z[0] != 0:
            raise RuntimeError('z[0] is required to be zero')
        self._z = z

        self._ti = ti
        for i in range(self._num_layers):
            if ti[i] <= 0.0:
                raise ValueError('thermal inertia ("ti") values should be positive')
        if self.ti.ndim != 1 or self.ti.size != self.num_layers:
            raise ValueError('"ti" should be a 1D array with length == z.size')

        self._rhoc = rhoc
        for i in range(self._num_layers):
            if rhoc[i] <= 0.0:
                raise ValueError('volumetric heat capacity ("rhoc") values should be positive')
        if self.rhoc.ndim != 1 or self.rhoc.size != self.num_layers:
            raise ValueError('"rhoc" should be a 1D array with length == z.size')

        self._Fgeotherm = Fgeotherm
        for i in range(self._num_faces):
            if Fgeotherm[i] < 0.0:
                raise ValueError('geothermal fluxes ("Fgeotherm") should be positive')
        if self.Fgeotherm.ndim != 1 or self.Fgeotherm.size != self.num_faces:
            raise ValueError('"Fgeotherm" should be a 1D array with length == T0.shape[0]')

        for i in range(self._num_faces):
            if Tsurfprev[i] < 0.0:
                raise ValueError('surface temperatures ("Tsurfprev") should be nonnegative')

        if Tsurfprev.ndim != 1 or Tsurfprev.size != self.num_faces:
            raise ValueError('"Tsurfprev" should be a 1D array with length == T0.shape[0]')
        # NOTE: For 'Q' BC we don’t actually use Tsurfprev in conductionQ,
        # but we keep the param for parity with your original interface.

        self._Qprev = Q0
        for i in range(self._num_faces):
            if Q0[i] < 0.0:
                raise ValueError('initial irradiance ("Q0") should be nonnegative')
        if self.Qprev.ndim != 1 or self.Qprev.size != self.num_faces:
            raise ValueError('"Q0" should be a 1D array with length == T0.shape[0]')

        self._Fsurf = np.empty(self._num_faces, dtype=np.float64)
        self._Fsurf[:] = np.nan

        self._T = T0_arr   # assigns memoryview from ndarray (keeps base alive)
        cdef Py_ssize_t j
        for i in range(self._num_faces):
            for j in range(self._num_layers):
                if T0[i, j] < 0.0:
                    raise ValueError('initial temperature ("T0") should be nonnegative')
        if self.T.ndim != 2 or self.T.shape[1] != self.num_layers:
            raise ValueError('"T0" should be a 2D array with T0.shape[1] == z.size')

    cpdef step(self, double dt, X):
        """
        Step conduction by dt using either flux ('Q') or temperature ('T') BC.
        X has length num_faces.
        """
        cdef int i

        # Move all cdefs here (Cython requirement)
        cdef double[::1] z = self._z
        cdef double[::1] ti = self._ti
        cdef double[::1] rhoc = self._rhoc
        cdef double[::1] Qprev = self._Qprev
        cdef double[::1] emiss = self._emiss
        cdef double[::1] Fgeotherm = self._Fgeotherm
        cdef double[:, ::1] T = self._T
        cdef double[::1] Fsurf = self._Fsurf

        cdef cnp.ndarray[cnp.double_t, ndim=1] X_arr
        cdef double[::1] Xv

        # Ensure we have a float64 1D view (no copy if already float64)
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 1:
            raise RuntimeError("X must be 1D")
        if X_arr.shape[0] != self._num_faces:
            raise RuntimeError(f'input of wrong size: X.size == {X_arr.shape[0]}')

        if not np.isfinite(X_arr).all():
            raise RuntimeError('trying to step with nonfinite inputs')

        Xv = X_arr  # typed view

        if self._bcond == 'Q':
            if (X_arr < 0).any():
                raise RuntimeError('trying to step with negative fluxes')

            # Parallel over faces (each face touches disjoint T[i,:], Fsurf[i], Qprev[i])
            with nogil:
                for i in prange(self._num_faces, schedule='static'):
                    conductionQ(
                        self._num_layers_interior,
                        &z[0],
                        dt,
                        Qprev[i],
                        Xv[i],
                        &T[i, 0],
                        &ti[0],
                        &rhoc[0],
                        emiss[i],
                        Fgeotherm[i],
                        &Fsurf[i])

                    # update Qprev per-face safely in same loop
                    Qprev[i] = Xv[i]

        elif self._bcond == 'T':
            if (X_arr < 0).any():
                raise RuntimeError('trying to step with negative input temps')

            with nogil:
                for i in prange(self._num_faces, schedule='static'):
                    conductionT(
                        self._num_layers_interior,
                        &z[0],
                        dt,
                        &T[i, 0],
                        T[i, 0],   # previous surface T
                        Xv[i],     # current surface T
                        &ti[0],
                        &rhoc[0],
                        Fgeotherm[i],
                        &Fsurf[i])

        else:
            raise RuntimeError(f'got unexpected BC mode: "{self._bcond}"')

        if not np.isfinite(np.asarray(T)).all():
            raise RuntimeError('computed nonfinite temperatures while stepping')

        # quick nonneg check (cheap enough vs conduction loop)
        if (np.asarray(T) < 0).any():
            raise RuntimeError('computed negative temperatures while stepping')

        self._current_time += dt


cpdef cnp.ndarray setgrid_uniform(int nz, double zmax):
    """
    Simple uniform depth grid [0, zmax] split in nz layers.
    The centers are at (i + 0.5)*dz; z[0] is > 0 so we prepend 0 in Pcc.
    """
    cdef int i
    cdef double dz = zmax / (nz - 1)
    cdef cnp.ndarray[cnp.double_t, ndim=1] z = np.empty(nz, dtype=np.float64)
    z[0] = 0.0
    for i in range(1, nz):
        z[i] = i * dz
    return z


cpdef cnp.ndarray setgrid(int nz, double zmax, double zfac):
    """
    Construct depth grid compatible with conductionQ.

    Returns z of length nz (NOT nz+1):
        z[0] = 0 (surface)
        z[1..nz-1] subsurface depths

    In conductionQ you should pass nz_interior = z.size - 1.
    Example: --nz 60 -> len(z)=60, conductionQ nz=59.
    """
    cdef int i
    cdef int N
    cdef double dz
    cdef double denom
    cdef cnp.ndarray[cnp.double_t, ndim=1] z

    if nz < 3:
        # conductionQ uses k[2] etc, so need at least z[0], z[1], z[2]
        raise ValueError("nz must be >= 3 (includes surface node)")
    if zmax <= 0.0:
        raise ValueError("zmax must be > 0")
    if zfac < 1.0:
        raise ValueError("zfac must be >= 1.0")

    z = np.empty(nz, dtype=np.float64)
    z[0] = 0.0

    # N = number of subsurface grid points used by conductionQ
    # (what conductionQ calls 'nz')
    N = nz - 1

    if zfac > 1.0:
        # Same geometric-grid formula, but with N (= nz-1) subsurface points.
        denom = 3.0 + 2.0*zfac*(zfac**(N - 2) - 1.0)/(zfac - 1.0)
        dz = zmax / denom
        z[1] = dz
        z[2] = 3.0*dz
        for i in range(3, N + 1):  # i = 3..N
            z[i] = (1.0 + zfac)*z[i - 1] - zfac*z[i - 2]
    else:
        # Uniform-ish grid with "half top layer":
        # want z[2] = 3*z[1] and constant spacing after that.
        # If z[i] = (i-0.5)*dz for i>=1, then z[2]=1.5dz and z[1]=0.5dz => ratio 3.
        # Set last point z[N] = zmax => (N - 0.5)*dz = zmax.
        dz = zmax / (N - 0.5)
        for i in range(1, N + 1):  # i = 1..N
            z[i] = (i - 0.5) * dz

    # sanity checks
    if not np.isfinite(z).all():
        raise RuntimeError("setgrid produced non-finite depths")
    if z[0] != 0.0:
        raise RuntimeError("setgrid did not set z[0]=0")
    if (z[1:] <= 0).any():
        raise RuntimeError("setgrid produced non-positive subsurface depths")
    if not np.all(np.diff(z) > 0):
        raise RuntimeError("setgrid produced non-monotone depths")

    return z


cdef inline int _zero(double[::1] v) nogil:
    cdef Py_ssize_t k
    for k in range(v.shape[0]):
        v[k] = 0.0
    return 0

cdef inline int _zero2(double[::1, :] A) nogil:
    cdef Py_ssize_t i, j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = 0.0
    return 0


cpdef object run_thermal_series_with_bf(
    object ff_op,
    cnp.ndarray[cnp.double_t, ndim=2] E_series,
    object dt_in,
    cnp.ndarray[cnp.double_t, ndim=1] z,
    cnp.ndarray[cnp.double_t, ndim=2] T0,
    cnp.ndarray[cnp.double_t, ndim=1] ti,
    cnp.ndarray[cnp.double_t, ndim=1] rhoc,
    cnp.ndarray[cnp.double_t, ndim=1] emiss,
    cnp.ndarray[cnp.double_t, ndim=1] Fgeotherm,
    double rho,
    bint clamp=True,
    int log_every=10,
    int num_reps=1,
    bint return_last_cycle_only=True,
    bint return_diagnostics=False,
):
    """
    Time-dependent thermal model driven by a BF FF operator.

    Parameters
    ----------
    ff_op : butterfly.VfHier
        Has method apply_inplace(x, y) performing y = F@x.
    E_series : (nt, Nfaces)
        Direct irradiance at each time.
    dt : float
        Constant time step between samples [s].
    z : (nz,)
        Depth grid. First entry must be 0 (Pcc requirement).
    T0 : (Nfaces, nz)
        Initial temperature profile [K].
    ti : (nz,)
        Thermal inertia profile.
    rhoc : (nz,)
        Volumetric heat capacity.
    emiss : (Nfaces,)
        Emissivity per face.
    Fgeotherm : (Nfaces,)
        Geothermal heat flux [W/m^2].
    rho : float
        Surface albedo.
    clamp : bool
        Clamp fluxes to >= 0.
    log_every : int
        Print a progress line every this many steps.

    Returns
    -------
    T_all : (nt, Nfaces, nz)
        Temperature at all depths and times.
    Tsurf_all : (nt, Nfaces)
        Surface temperature at each time.
    """
    import os
    print("affinity:", len(os.sched_getaffinity(0)), sorted(os.sched_getaffinity(0))[:32])

    cdef Py_ssize_t nt = E_series.shape[0]
    cdef Py_ssize_t Nfaces = E_series.shape[1]
    cdef Py_ssize_t nz = z.shape[0]

    if num_reps < 1:
        raise ValueError("num_reps must be >= 1")
    if T0.shape[0] != Nfaces or T0.shape[1] != nz:
        raise ValueError("T0 must be (Nfaces, nz)")
    if ti.shape[0] != nz or rhoc.shape[0] != nz:
        raise ValueError("ti, rhoc must be (nz,)")
    if emiss.shape[0] != Nfaces or Fgeotherm.shape[0] != Nfaces:
        raise ValueError("emiss, Fgeotherm must be (Nfaces,)")

    # views
    cdef double[:, ::1] E_view = E_series
    cdef double[::1] z_view = z
    cdef double[:, ::1] T0_view = T0
    cdef double[::1] ti_view = ti
    cdef double[::1] rhoc_view = rhoc
    cdef double[::1] emiss_view = emiss
    cdef double[::1] Fgeo_view = Fgeotherm

    cdef Py_ssize_t i, j, t_idx

    # --- initial direct flux and net flux Q0 for steady-state Tsurf0 ---
    cdef cnp.ndarray[cnp.double_t, ndim=1] E0 = E_series[0].copy()
    cdef cnp.ndarray[cnp.double_t, ndim=1] Q0 = np.empty(Nfaces, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] Tsurf0 = np.empty(Nfaces, dtype=np.float64)

    cdef double[::1] E0_view = E0
    cdef double[::1] Q0_view = Q0
    cdef double[::1] Tsurf0_view = Tsurf0
    cdef double denom, val

    cdef bint dt_is_scalar = False
    cdef double dt_scalar = 0.0
    cdef cnp.ndarray[cnp.double_t, ndim=1] dt_arr = None
    cdef double[::1] dt_view
    cdef double sum_dt_cycle = 0.0
    cdef double dt_step = 0.0

    # dt_in can be scalar or array (nt-1)
    try:
        dt_scalar = float(dt_in)
        if dt_scalar <= 0:
            raise ValueError("dt must be positive")
        dt_is_scalar = True
        sum_dt_cycle = dt_scalar * (<double>(nt - 1))
    except Exception:
        dt_arr = np.asarray(dt_in, dtype=np.float64)
        if dt_arr.ndim != 1 or dt_arr.shape[0] != nt - 1:
            raise ValueError("dt must be scalar or dt_series with length nt-1")
        dt_view = dt_arr
        # sum_dt_cycle = sum(dt_view)
        for t_idx in range(nt - 1):
            if dt_view[t_idx] <= 0.0:
                raise ValueError("dt_series contains non-positive entries")
            sum_dt_cycle += dt_view[t_idx]


    # for i in range(Nfaces):
    #     Q0_view[i] = (1.0 - rho) * E0_view[i] + Fgeo_view[i]
    #     if Q0_view[i] < 0.0 and clamp:
    #         Q0_view[i] = 0.0
    #     denom = emiss_view[i] * SIGSB
    #     if denom <= 0.0:
    #         Tsurf0_view[i] = 0.0
    #     else:
    #         Tsurf0_view[i] = (Q0_view[i] / denom)**0.25

    for i in range(Nfaces):
        Q0_view[i] = (1.0 - rho) * E0_view[i]
        if Q0_view[i] < 0.0 and clamp:
            Q0_view[i] = 0.0
        Tsurf0_view[i] = T0_view[i, 0]  # bcond='Q' doesn't use it anyway

    # --- construct Pcc model bundle ---
    cdef BfPccThermalModel1D model = BfPccThermalModel1D(
        z_view,
        T0_view,
        ti_view,
        rhoc_view,
        emiss_view,
        Fgeo_view,
        Q0_view,
        Tsurf0_view,
        bcond="Q",
    )

    cdef double[::1] Qprev_model = model._Qprev

    cdef Py_ssize_t nt_out
    if return_last_cycle_only:
        nt_out = nt
    else:
        nt_out = nt * num_reps

    # --- allocate output arrays ---
    cdef cnp.ndarray[cnp.double_t, ndim=3] T_all = np.empty(
        (nt_out, Nfaces, nz), dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=2] Tsurf_all = np.empty(
        (nt_out, Nfaces), dtype=np.float64)

    cdef double[:, :, ::1] T_all_view = T_all
    cdef double[:, ::1] Tsurf_all_view = Tsurf_all

    # --- diagnostics (small) ---
    cdef cnp.ndarray[cnp.double_t, ndim=2] mu_z = None
    cdef cnp.ndarray[cnp.double_t, ndim=2] amp_z = None
    cdef cnp.ndarray[cnp.double_t, ndim=1] drift_max = None
    cdef cnp.ndarray[cnp.double_t, ndim=2] Tbar_last = None

    cdef double[:, ::1] mu_z_view
    cdef double[:, ::1] amp_z_view
    cdef double[::1] drift_max_view
    cdef double[:, ::1] Tbar_last_view

    if return_diagnostics:
        mu_z = np.empty((num_reps, nz), dtype=np.float64)
        amp_z = np.empty((num_reps, nz), dtype=np.float64)
        drift_max = np.empty(num_reps, dtype=np.float64)
        drift_max[:] = np.nan
        Tbar_last = np.empty((nt, nz), dtype=np.float64)  # only last cycle’s Tbar(t,z)

        mu_z_view = mu_z
        amp_z_view = amp_z
        drift_max_view = drift_max
        Tbar_last_view = Tbar_last

    # --- working arrays (all length Nfaces) ---
    cdef cnp.ndarray[cnp.double_t, ndim=1] Qrefl = np.zeros(Nfaces, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] QIR = np.zeros(Nfaces, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] Qrefl_next = np.zeros(Nfaces, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] QIR_next = np.zeros(Nfaces, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] Tsurf_prev = np.asarray(T0[:, 0]).copy()
    cdef cnp.ndarray[cnp.double_t, ndim=1] tmp_short = np.empty(Nfaces, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] tmp_long = np.empty(Nfaces, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] res_short = np.empty(Nfaces, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] res_long = np.empty(Nfaces, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] Q = np.empty(Nfaces, dtype=np.float64)

    cdef double[::1] Qrefl_view = Qrefl
    cdef double[::1] QIR_view = QIR
    cdef double[::1] Qrefl_next_view = Qrefl_next
    cdef double[::1] QIR_next_view = QIR_next
    cdef double[::1] Tsurf_prev_view = Tsurf_prev
    cdef double[::1] tmp_short_view = tmp_short
    cdef double[::1] tmp_long_view = tmp_long
    cdef double[::1] res_short_view = res_short
    cdef double[::1] res_long_view = res_long
    cdef double[::1] Q_view = Q

    # --- many-RHS scratch (optional fast path) ---
    cdef bint use_many = False
    use_many = hasattr(ff_op, "apply_mat_inplace")

    cdef cnp.ndarray[cnp.double_t, ndim=2] X2 = None
    cdef cnp.ndarray[cnp.double_t, ndim=2] Y2 = None
    cdef double[::1, :] X2_view
    cdef double[::1, :] Y2_view

    if use_many:
        # Fortran order: (Nfaces, 2) with columns = RHS
        X2 = np.empty((Nfaces, 2), dtype=np.float64, order='F')
        Y2 = np.empty((Nfaces, 2), dtype=np.float64, order='F')
        X2_view = X2
        Y2_view = Y2

    cdef double[:, ::1] Tmat

    cdef double[::1] E_t_view

    # ---- cycle stats work arrays (nz) ----
    cdef cnp.ndarray[cnp.double_t, ndim=1] sum_z = np.empty(nz, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] tbar_z = np.empty(nz, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] tbar_sum = np.empty(nz, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] tbar_min = np.empty(nz, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] tbar_max = np.empty(nz, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] mu_prev = np.empty(nz, dtype=np.float64)

    cdef double[::1] sum_z_view = sum_z
    cdef double[::1] tbar_z_view = tbar_z
    cdef double[::1] tbar_sum_view = tbar_sum
    cdef double[::1] tbar_min_view = tbar_min
    cdef double[::1] tbar_max_view = tbar_max
    cdef double[::1] mu_prev_view = mu_prev

    cdef double invN = 1.0 / (<double>Nfaces)

    for j in range(nz):
        mu_prev_view[j] = 0.0

    # --- time loop ---
    cdef Py_ssize_t rep, out_t
    cdef double mu_j, dtmp, dmax

    for rep in range(num_reps):

        # init cycle accumulators
        if return_diagnostics:
            for j in range(nz):
                tbar_sum_view[j] = 0.0
                tbar_min_view[j] = 1.0e300
                tbar_max_view[j] = -1.0e300

        # -------------------------
        # t_idx = 0 : compute radiative states, set Qprev, STORE ONLY (no conduction step)
        # -------------------------
        t_idx = 0
        E_t_view = E_view[t_idx, :]

        if use_many:
            # Build 2 RHS columns:
            #   col0 = shortwave input  = E + Qrefl
            #   col1 = longwave input   = eps*sigma*T^4 + (1-eps)*QIR
            for i in range(Nfaces):
                X2_view[i, 0] = E_t_view[i] + Qrefl_view[i]

                val = emiss_view[i]*SIGSB*Tsurf_prev_view[i]*Tsurf_prev_view[i]* \
                      Tsurf_prev_view[i]*Tsurf_prev_view[i] + \
                      (1.0 - emiss_view[i]) * QIR_view[i]
                X2_view[i, 1] = val

            with nogil:
                _zero2(Y2_view)
            ff_op.apply_mat_inplace(X2, Y2)

            if not np.isfinite(Y2).all():
                raise RuntimeError("non-finite Y2 from FF apply_mat_inplace")

            # Unpack and clamp:
            for i in range(Nfaces):
                Qrefl_next_view[i] = rho * Y2_view[i, 0]
                if clamp and Qrefl_next_view[i] < 0.0:
                    Qrefl_next_view[i] = 0.0

                QIR_next_view[i] = Y2_view[i, 1]
                if clamp and QIR_next_view[i] < 0.0:
                    QIR_next_view[i] = 0.0

        else:
            # shortwave reflected
            for i in range(Nfaces):
                tmp_short_view[i] = E_t_view[i] + Qrefl_view[i]
            _zero(res_short_view)
            ff_op.apply_inplace(tmp_short, res_short)
            if not np.isfinite(res_short).all():
                raise RuntimeError("non-finite res_short from FF apply")

            for i in range(Nfaces):
                Qrefl_next_view[i] = rho * res_short_view[i]
                if clamp and Qrefl_next_view[i] < 0.0:
                    Qrefl_next_view[i] = 0.0

            # longwave
            for i in range(Nfaces):
                val = emiss_view[i]*SIGSB*Tsurf_prev_view[i]*Tsurf_prev_view[i]* \
                      Tsurf_prev_view[i]*Tsurf_prev_view[i] + \
                      (1.0 - emiss_view[i]) * QIR_view[i]
                tmp_long_view[i] = val

            _zero(res_long_view)
            ff_op.apply_inplace(tmp_long, res_long)
            if not np.isfinite(res_long).all():
                raise RuntimeError("non-finite res_long from FF apply")

            for i in range(Nfaces):
                QIR_next_view[i] = res_long_view[i]
                if clamp and QIR_next_view[i] < 0.0:
                    QIR_next_view[i] = 0.0

        # net flux at t0
        for i in range(Nfaces):
            val = (1.0 - rho)*(E_t_view[i] + Qrefl_next_view[i]) \
                  + emiss_view[i]*QIR_next_view[i]
            if clamp and val < 0.0:
                val = 0.0
            Q_view[i] = val
            # Make model's "previous" flux consistent for the first real step
            Qprev_model[i] = val

        # shift radiative states for next step
        for i in range(Nfaces):
            Qrefl_view[i] = Qrefl_next_view[i]
            QIR_view[i] = QIR_next_view[i]

        # decide output index and STORE state at t0 (no conduction applied yet)
        if return_last_cycle_only:
            out_t = 0 if rep == num_reps - 1 else -1
        else:
            out_t = rep * nt + 0

        Tmat = model._T
        if return_diagnostics:
            for j in range(nz):
                sum_z_view[j] = 0.0

        for i in range(Nfaces):
            # IMPORTANT: keep Tsurf_prev consistent with the actual state stored in T
            Tsurf_prev_view[i] = Tmat[i, 0]

            if out_t >= 0:
                Tsurf_all_view[out_t, i] = Tmat[i, 0]
                for j in range(nz):
                    T_all_view[out_t, i, j] = Tmat[i, j]

            if return_diagnostics:
                for j in range(nz):
                    sum_z_view[j] += Tmat[i, j]

        if return_diagnostics:
            for j in range(nz):
                tbar_z_view[j] = sum_z_view[j] * invN
                if rep == num_reps - 1:
                    Tbar_last_view[0, j] = tbar_z_view[j]
                # amplitude tracking includes t0
                if tbar_z_view[j] < tbar_min_view[j]:
                    tbar_min_view[j] = tbar_z_view[j]
                if tbar_z_view[j] > tbar_max_view[j]:
                    tbar_max_view[j] = tbar_z_view[j]
                # NOTE: we do NOT add to tbar_sum here (no interval weight at t0)

        # -------------------------
        # t_idx = 1..nt-1 : compute flux at t_idx, step over dt[t_idx-1], STORE
        # -------------------------
        for t_idx in range(1, nt):

            if log_every > 0 and (t_idx + 1) % log_every == 0:
                if num_reps == 1:
                    print(f"[bf-thermal] step {t_idx+1}/{nt}")
                else:
                    print(f"[bf-thermal] rep {rep+1}/{num_reps} step {t_idx+1}/{nt}")
                sys.stdout.flush()

            E_t_view = E_view[t_idx, :]
            t0 = time.perf_counter()

            if use_many:
                # Build 2 RHS columns:
                #   col0 = shortwave input  = E + Qrefl
                #   col1 = longwave input   = eps*sigma*T^4 + (1-eps)*QIR
                for i in range(Nfaces):
                    X2_view[i, 0] = E_t_view[i] + Qrefl_view[i]

                    val = emiss_view[i] * SIGSB * Tsurf_prev_view[i] * Tsurf_prev_view[i] * \
                          Tsurf_prev_view[i] * Tsurf_prev_view[i] + \
                          (1.0 - emiss_view[i]) * QIR_view[i]
                    X2_view[i, 1] = val
                tA = time.perf_counter()

                with nogil:
                    _zero2(Y2_view)
                ff_op.apply_mat_inplace(X2, Y2)

                if not np.isfinite(Y2).all():
                    raise RuntimeError("non-finite Y2 from FF apply_mat_inplace")
                tB = time.perf_counter()

                # Unpack and clamp:
                for i in range(Nfaces):
                    Qrefl_next_view[i] = rho * Y2_view[i, 0]
                    if clamp and Qrefl_next_view[i] < 0.0:
                        Qrefl_next_view[i] = 0.0

                    QIR_next_view[i] = Y2_view[i, 1]
                    if clamp and QIR_next_view[i] < 0.0:
                        QIR_next_view[i] = 0.0
                tC = time.perf_counter()
                tD = time.perf_counter()
                tE = time.perf_counter()
            else:
                # shortwave reflected
                for i in range(Nfaces):
                    tmp_short_view[i] = E_t_view[i] + Qrefl_view[i]
                tA = time.perf_counter()
                _zero(res_short_view)
                ff_op.apply_inplace(tmp_short, res_short)
                if not np.isfinite(res_short).all():
                    raise RuntimeError("non-finite res_short from FF apply")
                tB = time.perf_counter()

                for i in range(Nfaces):
                    Qrefl_next_view[i] = rho * res_short_view[i]
                    if clamp and Qrefl_next_view[i] < 0.0:
                        Qrefl_next_view[i] = 0.0
                tC = time.perf_counter()

                # longwave
                for i in range(Nfaces):
                    val = emiss_view[i]*SIGSB*Tsurf_prev_view[i]*Tsurf_prev_view[i]* \
                          Tsurf_prev_view[i]*Tsurf_prev_view[i] + \
                          (1.0 - emiss_view[i]) * QIR_view[i]
                    tmp_long_view[i] = val
                tD = time.perf_counter()

                _zero(res_long_view)
                ff_op.apply_inplace(tmp_long, res_long)
                tE = time.perf_counter()
                if not np.isfinite(res_long).all():
                    raise RuntimeError("non-finite res_long from FF apply")

                for i in range(Nfaces):
                    QIR_next_view[i] = res_long_view[i]
                    if clamp and QIR_next_view[i] < 0.0:
                        QIR_next_view[i] = 0.0

            # net flux at t_idx
            for i in range(Nfaces):
                val = (1.0 - rho)*(E_t_view[i] + Qrefl_next_view[i]) \
                      + emiss_view[i]*QIR_next_view[i]
                if clamp and val < 0.0:
                    val = 0.0
                Q_view[i] = val

            # dt for interval [t_idx-1, t_idx]
            if dt_is_scalar:
                dt_step = dt_scalar
            else:
                dt_step = dt_view[t_idx - 1]
            tF = time.perf_counter()

            # conduction step advances from previous endpoint flux (stored in model._Qprev) to current Q
            model.step(dt_step, Q)
            tG = time.perf_counter()

            # shift radiative states for next step
            for i in range(Nfaces):
                Qrefl_view[i] = Qrefl_next_view[i]
                QIR_view[i] = QIR_next_view[i]

            # output index
            if return_last_cycle_only:
                out_t = t_idx if rep == num_reps - 1 else -1
            else:
                out_t = rep * nt + t_idx

            # store and diagnostics
            Tmat = model._T
            tH = time.perf_counter()

            if return_diagnostics:
                for j in range(nz):
                    sum_z_view[j] = 0.0

            for i in range(Nfaces):
                Tsurf_prev_view[i] = Tmat[i, 0]

                if out_t >= 0:
                    Tsurf_all_view[out_t, i] = Tmat[i, 0]
                    for j in range(nz):
                        T_all_view[out_t, i, j] = Tmat[i, j]

                if return_diagnostics:
                    for j in range(nz):
                        sum_z_view[j] += Tmat[i, j]

            if return_diagnostics:
                for j in range(nz):
                    tbar_z_view[j] = sum_z_view[j] * invN

                    if rep == num_reps - 1:
                        Tbar_last_view[t_idx, j] = tbar_z_view[j]

                    # time-weighted mean accumulator
                    tbar_sum_view[j] += tbar_z_view[j] * dt_step

                    # amplitude tracking
                    if tbar_z_view[j] < tbar_min_view[j]:
                        tbar_min_view[j] = tbar_z_view[j]
                    if tbar_z_view[j] > tbar_max_view[j]:
                        tbar_max_view[j] = tbar_z_view[j]

            print(f"[timing] prepS={tA - t0:.4f} applyS={tB - tA:.4f} postS={tC - tB:.4f} "
                  f"prepL={tD - tC:.4f} applyL={tE - tD:.4f} postL={tF - tE:.4f} "
                  f"cond={tG - tF:.4f} store={tH - tG:.4f} total={tH - t0:.4f}")

        # end of cycle: write mu_z, amp_z, drift
        if return_diagnostics:
            dmax = 0.0
            for j in range(nz):
                mu_j = tbar_sum_view[j] / sum_dt_cycle
                mu_z_view[rep, j] = mu_j
                amp_z_view[rep, j] = tbar_max_view[j] - tbar_min_view[j]

                if rep > 0:
                    dtmp = mu_j - mu_prev_view[j]
                    if dtmp < 0:
                        dtmp = -dtmp
                    if dtmp > dmax:
                        dmax = dtmp

                mu_prev_view[j] = mu_j

            if rep > 0:
                drift_max_view[rep] = dmax

    # ---- AFTER finishing all reps ----
    if not return_diagnostics:
        return T_all, Tsurf_all

    diag = {
        "mu_z": mu_z,
        "amp_z": amp_z,
        "drift_max": drift_max,
        "Tbar_last": Tbar_last,
    }
    return T_all, Tsurf_all, diag


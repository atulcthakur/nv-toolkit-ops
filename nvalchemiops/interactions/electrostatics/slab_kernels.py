# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Yeh-Berkowitz Slab Correction Kernels
=====================================

This module provides Warp kernels for the Yeh-Berkowitz slab
correction, enabling accurate electrostatics for 2D periodic (slab) systems.

The slab correction removes spurious interactions between periodic images
along the non-periodic direction when using 3D Ewald methods for systems that
are only periodic in two dimensions. For triclinic cells, positions are
projected onto the normal of the two periodic cell vectors.

MATHEMATICAL FORMULATION
========================

Let :math:`\mathbf{n}` be the unit normal to the periodic plane,
:math:`z_i = \mathbf{r}_i \cdot \mathbf{n}`, and
:math:`L = |\mathbf{h}_k \cdot \mathbf{n}|`, where :math:`\mathbf{h}_k`
is the non-periodic cell vector selected by pbc. Per-atom energy:

.. math::

    E_{\\text{slab},i} = \\frac{2\\pi}{V} q_i
    \\left[ z_i M - \\frac{1}{2}(M_2 + Q z_i^2) - \\frac{Q}{12} L^2 \\right]

Per-atom force:

.. math::

    \\mathbf{F}_{\\text{slab},i} = -\\frac{4\\pi}{V} q_i (M - Q z_i) \\mathbf{n}

Per-atom charge gradient:

.. math::

    \\frac{\\partial E_{\\text{slab}}}{\\partial q_i} = \\frac{4\\pi}{V}
    \\left[ z_i M - \\frac{1}{2}(M_2 + Q z_i^2) - \\frac{Q}{12} L^2 \\right]

Per-atom virial contribution under the normal-following affine strain
convention :math:`\mathbf{r}' = \mathbf{F}\mathbf{r}`,
:math:`\mathbf{h}' = \mathbf{F}\mathbf{h}`:

.. math::

    \\mathbf{W}_{\\text{slab},i} =
    E_{\\text{slab},i}(\\mathbf{I} - 2\\mathbf{n}\\mathbf{n}^{T})

where :math:`M = \\sum_j q_j z_j`, :math:`M_2 = \\sum_j q_j z_j^2`,
:math:`Q = \\sum_j q_j`, :math:`V = |\\det(\\mathbf{h})|`,
:math:`L = |\\mathbf{h}_k \\cdot \mathbf{n}|`.

CELL GEOMETRY
=============

Orthorhombic and triclinic cells are supported. The pbc tensor selects the
non-periodic cell vector. The slab normal is recomputed from the two periodic
cell vectors for each system, so tilted periodic planes use the correct
normal-following geometry.

NON-NEUTRAL SYSTEMS
===================

For systems with net charge :math:`Q \\ne 0`, the slab correction follows
the Ballenegger et al. (2009) Eq. 29 convention: a uniform-volume
neutralizing background charge density :math:`\\rho_b = -Q/V` (the same
convention used by standard 3D Ewald). Other conventions (uniform plane,
pinned dipole) yield different additive constants.

PER-SYSTEM PBC
==============

Each batch system carries its own pbc tensor of shape (3,) with True for
periodic directions and False for the non-periodic direction. The kernels
inspect pbc[system_id] to determine the non-periodic axis without any
host/device synchronization. Systems with pbc patterns other than exactly
one False entry (e.g., fully 3D periodic [T, T, T] or 1D periodic) yield
zero contribution.

KERNEL ORGANIZATION
===================

Moment Reduction:
    _slab_reduce_moments_kernel: Accumulate projected M, M2, Q_total per system

Per-Atom Correction:
    _slab_correction_kernel: Energy, forces, charge gradients, and virial

Both kernels handle single-system and batched calculations via batch_idx.
For single systems, pass batch_idx = zeros(N, dtype=int32).

REFERENCES
==========

- Yeh, I.-C. & Berkowitz, M. L. (1999). J. Chem. Phys. 111, 3155-3162.
  (Original slab correction for neutral systems)
- Ballenegger, V., Arnold, A. & Cerdà, J. J. (2009). J. Chem. Phys. 131, 094107.
  (Extension to non-neutral systems via background charge correction, Eq. 29)
"""

import math
from typing import Any

import warp as wp

# Mathematical constants
PI = wp.constant(wp.float64(math.pi))
TWOPI = wp.constant(wp.float64(2.0 * math.pi))
FOURPI = wp.constant(wp.float64(4.0 * math.pi))


###########################################################################################
########################### Moment Reduction Kernel #######################################
###########################################################################################


@wp.kernel
def _slab_reduce_moments_kernel(
    positions: wp.array(dtype=Any),  # (N,) vec3
    charges: wp.array(dtype=Any),  # (N,)
    batch_idx: wp.array(dtype=wp.int32),  # (N,)
    pbc: wp.array2d(dtype=wp.bool),  # (B, 3) per-system pbc
    cell: wp.array(dtype=Any),  # (B,) mat33 -- slab normal computed inside
    mz: wp.array2d(dtype=wp.float64),  # (B, 3) OUTPUT -- projected M in slab-axis slot
    mz2: wp.array2d(
        dtype=wp.float64
    ),  # (B, 3) OUTPUT -- projected M2 in slab-axis slot
    qtotal: wp.array(dtype=wp.float64),  # (B,) OUTPUT -- total charge per system
):
    """Accumulate charge moments along each system's non-periodic axis.

    Each thread processes one atom and accumulates its contributions to its
    system's moments using atomic additions. The non-periodic axis is
    determined per-system from pbc[system_id] entirely on-device.

    Launch Grid
    -----------
    dim = [N_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom (0 to B-1). For single systems, all zeros.
    pbc : wp.array2d, shape (B, 3), dtype=wp.bool
        Per-system periodic boundary conditions. True for periodic directions,
        False for the non-periodic (slab) direction. Systems with patterns
        other than exactly one False entry contribute zero.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system cell matrices. The slab normal is computed from the two
        periodic cell vectors.
    mz : wp.array2d, shape (B, 3), dtype=wp.float64
        OUTPUT: Per-system projected dipole M = sum_i q_i (r_i dot n),
        stored in the non-periodic axis slot.
        Must be zero-initialized before launch.
    mz2 : wp.array2d, shape (B, 3), dtype=wp.float64
        OUTPUT: Per-system projected moment M2 = sum_i q_i (r_i dot n)^2,
        stored in the non-periodic axis slot.
        Must be zero-initialized before launch.
    qtotal : wp.array, shape (B,), dtype=wp.float64
        OUTPUT: Per-system total charge.
        Must be zero-initialized before launch.

    Notes
    -----
    - All accumulations use float64 for numerical stability.
    - Output arrays must be zero-initialized before kernel launch.
    - Atoms in non-slab systems contribute nothing; the kernel determines
      slab geometry per-system from pbc without any host sync.
    """
    atom_idx = wp.tid()

    system_id = batch_idx[atom_idx]
    p0 = pbc[system_id, 0]
    p1 = pbc[system_id, 1]
    p2 = pbc[system_id, 2]

    q = charges[atom_idx]
    pos = positions[atom_idx]

    # Determine the non-periodic axis (the index where pbc is False).
    # Slab geometry has exactly one False entry. Other patterns
    # (fully 3D periodic, 1D periodic, no periodicity) contribute zero.
    axis_idx = wp.int32(2)
    is_slab = False

    if (not p0) and p1 and p2:
        axis_idx = wp.int32(0)
        is_slab = True
    elif p0 and (not p1) and p2:
        axis_idx = wp.int32(1)
        is_slab = True
    elif p0 and p1 and (not p2):
        axis_idx = wp.int32(2)
        is_slab = True

    if is_slab:
        cell_b = cell[system_id]

        # Pick periodic cell vectors by the cyclic convention:
        # axis 0 -> cross(h1, h2), axis 1 -> cross(h2, h0),
        # axis 2 -> cross(h0, h1). This reduces to +x/+y/+z for
        # right-handed axis-aligned cells.
        periodic_a = cell_b[0]
        periodic_b = cell_b[1]
        if axis_idx == wp.int32(0):
            periodic_a = cell_b[1]
            periodic_b = cell_b[2]
        elif axis_idx == wp.int32(1):
            periodic_a = cell_b[2]
            periodic_b = cell_b[0]

        normal_raw = wp.cross(periodic_a, periodic_b)
        normal = normal_raw / wp.length(normal_raw)
        z = wp.dot(pos, normal)

        q_f64 = wp.float64(q)
        z_f64 = wp.float64(z)
        m_contrib = q_f64 * z_f64
        m2_contrib = m_contrib * z_f64

        if axis_idx == wp.int32(0):
            wp.atomic_add(mz, system_id, 0, m_contrib)
            wp.atomic_add(mz2, system_id, 0, m2_contrib)
        elif axis_idx == wp.int32(1):
            wp.atomic_add(mz, system_id, 1, m_contrib)
            wp.atomic_add(mz2, system_id, 1, m2_contrib)
        else:
            wp.atomic_add(mz, system_id, 2, m_contrib)
            wp.atomic_add(mz2, system_id, 2, m2_contrib)

        wp.atomic_add(qtotal, system_id, wp.float64(q))


###########################################################################################
########################### Per-Atom Slab Correction Kernel ###############################
###########################################################################################


@wp.kernel
def _slab_correction_kernel(
    positions: wp.array(dtype=Any),  # (N,) vec3
    charges: wp.array(dtype=Any),  # (N,)
    batch_idx: wp.array(dtype=wp.int32),  # (N,)
    pbc: wp.array2d(dtype=wp.bool),  # (B, 3) per-system pbc
    cell: wp.array(dtype=Any),  # (B,) mat33 -- volume/normal/height computed inside
    mz: wp.array2d(dtype=wp.float64),  # (B, 3) projected M in slab-axis slot
    mz2: wp.array2d(dtype=wp.float64),  # (B, 3) projected M2 in slab-axis slot
    qtotal: wp.array(dtype=wp.float64),  # (B,) precomputed total charge
    energy_in: wp.array(dtype=wp.float64),  # (N,) input energies
    energy_out: wp.array(dtype=wp.float64),  # (N,) OUTPUT: energy_in + slab correction
    forces: wp.array(dtype=Any),  # (N,) vec3 -- OUTPUT: accumulated forces
    charge_grads: wp.array(
        dtype=wp.float64
    ),  # (N,) OUTPUT: accumulated charge gradients
    virial: wp.array(dtype=Any),  # (B,) mat33 -- OUTPUT: accumulated virial tensor
):
    """Apply Yeh-Berkowitz slab correction (with Ballenegger non-neutral
    extension) to per-atom energies, forces, charge gradients, and virial.

    Each thread processes one atom independently. Reads precomputed moments
    from the reduction kernel and computes all correction terms. The
    non-periodic axis is determined per-system from pbc[system_id].

    Launch Grid
    -----------
    dim = [N_atoms]

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom (0 to B-1).
    pbc : wp.array2d, shape (B, 3), dtype=wp.bool
        Per-system periodic boundary conditions.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system cell matrices. Volume, periodic-plane normal, and projected
        slab height are computed inside the kernel so the Warp tape captures
        the chain for autograd backpropagation to cell.
    mz : wp.array, shape (B, 3), dtype=wp.float64
        Per-system projected dipole moment from reduction kernel, stored in
        the non-periodic axis slot.
    mz2 : wp.array, shape (B, 3), dtype=wp.float64
        Per-system projected second moment from reduction kernel, stored in
        the non-periodic axis slot.
    qtotal : wp.array, shape (B,), dtype=wp.float64
        Per-system total charge from reduction kernel.
    energy_in : wp.array, shape (N,), dtype=wp.float64
        Input per-atom energies (from Ewald).
    energy_out : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Corrected per-atom energies (energy_in + slab correction).
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Slab force contribution accumulated onto existing forces.
    charge_grads : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Slab charge gradient contribution accumulated onto existing gradients.
    virial : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Slab virial contribution accumulated onto existing virial.

    Notes
    -----
    - Uses separate energy_in/energy_out arrays for Warp autodiff compatibility.
    - Forces and charge_grads are accumulated via atomic_add (not overwritten).
    - Virial follows the normal-following affine strain convention:
      W_i = E_i * (I - 2 n n^T).
    - Cell is accepted in user dtype; geometry is computed inside the Warp
      kernel so the tape captures the full cell -> correction chain for
      autograd backpropagation.
    - Forces/virial are cast back to input dtype for output.
    - Energy prefactor is 2*pi/V (toolkit-ops returns per-atom energies
      directly; no final /2 division).
    - Force and charge gradient prefactors are 4*pi/V due to the double-sum
      structure of the total energy.
    - Systems whose pbc is not slab-like (not exactly one False entry) get
      energy_out[i] = energy_in[i] (passthrough) and zero contribution to
      forces, charge_grads, virial.
    """
    atom_idx = wp.tid()

    system_id = batch_idx[atom_idx]
    p0 = pbc[system_id, 0]
    p1 = pbc[system_id, 1]
    p2 = pbc[system_id, 2]

    q = charges[atom_idx]
    pos = positions[atom_idx]

    # Determine non-periodic axis index (0, 1, or 2).
    axis_idx = wp.int32(2)
    is_slab = False

    if (not p0) and p1 and p2:
        axis_idx = wp.int32(0)
        is_slab = True
    elif p0 and (not p1) and p2:
        axis_idx = wp.int32(1)
        is_slab = True
    elif p0 and p1 and (not p2):
        axis_idx = wp.int32(2)
        is_slab = True

    if not is_slab:
        # Passthrough; no slab contribution
        energy_out[atom_idx] = energy_in[atom_idx]
        return

    # Compute geometry inside the kernel so Warp autodiff tracks cell dependence.
    cell_b = cell[system_id]
    vol = wp.float64(wp.abs(wp.determinant(cell_b)))

    periodic_a = cell_b[0]
    periodic_b = cell_b[1]
    nonperiodic_c = cell_b[2]
    if axis_idx == wp.int32(0):
        periodic_a = cell_b[1]
        periodic_b = cell_b[2]
        nonperiodic_c = cell_b[0]
    elif axis_idx == wp.int32(1):
        periodic_a = cell_b[2]
        periodic_b = cell_b[0]
        nonperiodic_c = cell_b[1]

    normal_raw = wp.cross(periodic_a, periodic_b)
    normal = normal_raw / wp.length(normal_raw)
    z = wp.dot(pos, normal)
    c_dot_n = wp.dot(nonperiodic_c, normal)
    height_sq = wp.float64(c_dot_n) * wp.float64(c_dot_n)

    # Load projected per-system moments (float64). The reduction kernel stores
    # M and M2 in the slot corresponding to the non-periodic pbc axis.
    mz_val = mz[system_id, 2]
    mz2_val = mz2[system_id, 2]
    if axis_idx == wp.int32(0):
        mz_val = mz[system_id, 0]
        mz2_val = mz2[system_id, 0]
    elif axis_idx == wp.int32(1):
        mz_val = mz[system_id, 1]
        mz2_val = mz2[system_id, 1]
    qtot = qtotal[system_id]

    # Cast to float64 for bracket computation (mixes with float64 moments)
    z_f64 = wp.float64(z)
    q_f64 = wp.float64(q)

    # Common subexpression: bracket term
    # [z_i * M - 0.5 * (M2 + Q * z_i^2) - Q/12 * L^2]
    bracket = (
        z_f64 * mz_val
        - wp.float64(0.5) * (mz2_val + qtot * z_f64 * z_f64)
        - qtot / wp.float64(12.0) * height_sq
    )

    # E_slab_i = (2*pi/V) * q_i * bracket
    twopi_over_v = TWOPI / vol
    e_slab = twopi_over_v * q_f64 * bracket
    energy_out[atom_idx] = energy_in[atom_idx] + e_slab

    # F_slab_i = -(4*pi/V) * q_i * (M - Q * z_i) * n
    fourpi_over_v = FOURPI / vol
    f_slab_mag = -fourpi_over_v * q_f64 * (mz_val - qtot * z_f64)

    f_slab = type(pos)(
        type(pos[0])(f_slab_mag * wp.float64(normal[0])),
        type(pos[0])(f_slab_mag * wp.float64(normal[1])),
        type(pos[0])(f_slab_mag * wp.float64(normal[2])),
    )

    wp.atomic_add(forces, atom_idx, f_slab)

    # dE_slab/dq_i = (4*pi/V) * bracket
    cg_slab = fourpi_over_v * bracket
    wp.atomic_add(charge_grads, atom_idx, cg_slab)

    # Under affine strain of positions and cell, n follows the periodic plane:
    # W_i = E_i * (I - 2 n n^T).
    n0 = wp.float64(normal[0])
    n1 = wp.float64(normal[1])
    n2 = wp.float64(normal[2])
    two = wp.float64(2.0)
    one = wp.float64(1.0)

    virial_mat = wp.mat33d(
        e_slab * (one - two * n0 * n0),
        e_slab * (-two * n0 * n1),
        e_slab * (-two * n0 * n2),
        e_slab * (-two * n1 * n0),
        e_slab * (one - two * n1 * n1),
        e_slab * (-two * n1 * n2),
        e_slab * (-two * n2 * n0),
        e_slab * (-two * n2 * n1),
        e_slab * (one - two * n2 * n2),
    )

    wp.atomic_add(virial, system_id, type(virial[0])(virial_mat))


###########################################################################################
########################### Overload Registration #########################################
###########################################################################################

# Type aliases (matching ewald_kernels.py convention)
_T = [wp.float32, wp.float64]
_V = [wp.vec3f, wp.vec3d]
_M = [wp.mat33f, wp.mat33d]

# Overload dictionaries
_slab_reduce_moments_kernel_overload = {}
_slab_correction_kernel_overload = {}

for t, v, m in zip(_T, _V, _M):
    _slab_reduce_moments_kernel_overload[t] = wp.overload(
        _slab_reduce_moments_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array2d(dtype=wp.bool),  # pbc
            wp.array(dtype=m),  # cell (mat33)
            wp.array2d(dtype=wp.float64),  # mz (B, 3)
            wp.array2d(dtype=wp.float64),  # mz2 (B, 3)
            wp.array(dtype=wp.float64),  # qtotal
        ],
    )

    _slab_correction_kernel_overload[t] = wp.overload(
        _slab_correction_kernel,
        [
            wp.array(dtype=v),  # positions
            wp.array(dtype=t),  # charges
            wp.array(dtype=wp.int32),  # batch_idx
            wp.array2d(dtype=wp.bool),  # pbc
            wp.array(dtype=m),  # cell (mat33)
            wp.array2d(dtype=wp.float64),  # mz (B, 3)
            wp.array2d(dtype=wp.float64),  # mz2 (B, 3)
            wp.array(dtype=wp.float64),  # qtotal
            wp.array(dtype=wp.float64),  # energy_in
            wp.array(dtype=wp.float64),  # energy_out
            wp.array(dtype=v),  # forces
            wp.array(dtype=wp.float64),  # charge_grads
            wp.array(dtype=m),  # virial
        ],
    )


###########################################################################################
########################### Launcher Functions ############################################
###########################################################################################


def slab_reduce_moments(
    positions: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    pbc: wp.array,
    cell: wp.array,
    mz: wp.array,
    mz2: wp.array,
    qtotal: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch kernel to accumulate slab correction moments.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom.
    pbc : wp.array, shape (B, 3), dtype=wp.bool
        Per-system periodic boundary conditions.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system cell matrices.
    mz : wp.array, shape (B, 3), dtype=wp.float64
        OUTPUT: Projected dipole moment in the non-periodic axis slot.
        Must be zero-initialized.
    mz2 : wp.array, shape (B, 3), dtype=wp.float64
        OUTPUT: Projected second moment in the non-periodic axis slot.
        Must be zero-initialized.
    qtotal : wp.array, shape (B,), dtype=wp.float64
        OUTPUT: Total charge. Must be zero-initialized.
    wp_dtype : type
        Warp scalar type (wp.float32 or wp.float64).
    device : str, optional
        Warp device.
    """
    num_atoms = charges.shape[0]
    if device is None:
        device = str(charges.device)

    wp.launch(
        _slab_reduce_moments_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[positions, charges, batch_idx, pbc, cell, mz, mz2, qtotal],
        device=device,
    )


def slab_correction(
    positions: wp.array,
    charges: wp.array,
    batch_idx: wp.array,
    pbc: wp.array,
    cell: wp.array,
    mz: wp.array,
    mz2: wp.array,
    qtotal: wp.array,
    energy_in: wp.array,
    energy_out: wp.array,
    forces: wp.array,
    charge_grads: wp.array,
    virial: wp.array,
    wp_dtype: type,
    device: str | None = None,
) -> None:
    """Launch kernel to apply slab correction to energies, forces,
    charge gradients, and virial.

    Parameters
    ----------
    positions : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        Atomic coordinates.
    charges : wp.array, shape (N,), dtype=wp.float32 or wp.float64
        Atomic charges.
    batch_idx : wp.array, shape (N,), dtype=wp.int32
        System index for each atom.
    pbc : wp.array, shape (B, 3), dtype=wp.bool
        Per-system periodic boundary conditions.
    cell : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        Per-system cell matrices. Volume, slab normal, and projected height
        are computed inside the kernel.
    mz : wp.array, shape (B, 3), dtype=wp.float64
        Per-system projected dipole moment (from slab_reduce_moments).
    mz2 : wp.array, shape (B, 3), dtype=wp.float64
        Per-system projected second moment (from slab_reduce_moments).
    qtotal : wp.array, shape (B,), dtype=wp.float64
        Per-system total charge (from slab_reduce_moments).
    energy_in : wp.array, shape (N,), dtype=wp.float64
        Input per-atom energies.
    energy_out : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Corrected per-atom energies.
    forces : wp.array, shape (N,), dtype=wp.vec3f or wp.vec3d
        OUTPUT: Forces (slab contribution accumulated).
    charge_grads : wp.array, shape (N,), dtype=wp.float64
        OUTPUT: Charge gradients (slab contribution accumulated).
    virial : wp.array, shape (B,), dtype=wp.mat33f or wp.mat33d
        OUTPUT: Virial tensor (slab contribution accumulated).
    wp_dtype : type
        Warp scalar type (wp.float32 or wp.float64).
    device : str, optional
        Warp device.
    """
    num_atoms = charges.shape[0]
    if device is None:
        device = str(charges.device)

    wp.launch(
        _slab_correction_kernel_overload[wp_dtype],
        dim=num_atoms,
        inputs=[
            positions,
            charges,
            batch_idx,
            pbc,
            cell,
            mz,
            mz2,
            qtotal,
            energy_in,
            energy_out,
            forces,
            charge_grads,
            virial,
        ],
        device=device,
    )

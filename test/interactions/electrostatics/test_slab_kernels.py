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

"""
Unit tests for slab correction Warp kernel launchers.

Tests cover:
- Moment reduction correctness (M_z, M_z2, Q_total)
- Per-atom energy correctness vs analytical formula
- Per-atom force correctness vs analytical formula
- Float32 and float64 dtypes
- Non-periodic axis selection (x, y, z)
- Triclinic projected-normal geometry
- Neutral vs non-neutral systems
- Batch vs individual consistency
- Mixed-axis batches (system 0 has slab in z, system 1 has slab in y)

These tests use Warp arrays directly and do not require PyTorch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from nvalchemiops.interactions.electrostatics.slab_kernels import (
    slab_correction,
    slab_reduce_moments,
)

PI = math.pi
TWOPI = 2.0 * PI
FOURPI = 4.0 * PI


# ==============================================================================
# Helpers
# ==============================================================================


def _axis_to_pbc(axis: int) -> np.ndarray:
    """Convert a non-periodic axis index (0, 1, or 2) to a (3,) pbc bool array."""
    pbc = np.array([True, True, True], dtype=np.bool_)
    pbc[axis] = False
    return pbc


def _slab_normal(cell, axis):
    """Return the periodic-plane normal for a cell with row-vector lattice."""
    normal = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
    return normal / np.linalg.norm(normal)


def analytical_slab_correction(positions, charges, cell, axis):
    """Compute slab correction using numpy for reference.

    Returns per-atom energies, forces (N,3), charge_grads, and per-atom virial
    matrices using the normal-following triclinic geometry.
    """
    normal = _slab_normal(cell, axis)
    z = positions @ normal
    q = charges
    V = abs(np.linalg.det(cell))
    L2 = np.dot(cell[axis], normal) ** 2

    M = np.sum(q * z)
    M2 = np.sum(q * z**2)
    Q = np.sum(q)

    # Per-atom energy
    bracket = z * M - 0.5 * (M2 + Q * z**2) - Q / 12.0 * L2
    energies = (TWOPI / V) * q * bracket

    # Per-atom force along slab normal
    forces = (-(FOURPI / V) * q * (M - Q * z))[:, None] * normal[None, :]

    # Per-atom charge gradient
    charge_grads = (FOURPI / V) * bracket

    projector = np.eye(3) - 2.0 * np.outer(normal, normal)
    virial_per_atom = energies[:, None, None] * projector[None, :, :]

    return energies, forces, charge_grads, virial_per_atom


def _make_warp_arrays(system, wp_dtype, device="cpu"):
    """Convert a single-system test dict to Warp arrays for kernel calls.

    `system` must contain: positions, charges, cell, axis, num_atoms.
    Returns a dict of Warp arrays + helpful metadata.
    """
    if wp_dtype == wp.float32:
        np_dtype = np.float32
        vec_dtype = wp.vec3f
        mat_dtype = wp.mat33f
    else:
        np_dtype = np.float64
        vec_dtype = wp.vec3d
        mat_dtype = wp.mat33d

    N = system["num_atoms"]
    cell_np = system["cell"]
    axis = system["axis"]

    positions = wp.array(
        system["positions"].astype(np_dtype), dtype=vec_dtype, device=device
    )
    charges = wp.array(
        system["charges"].astype(np_dtype), dtype=wp_dtype, device=device
    )
    batch_idx = wp.zeros(N, dtype=wp.int32, device=device)

    # pbc as (1, 3) bool array (single system)
    pbc_np = _axis_to_pbc(axis)[None, :]  # (1, 3)
    pbc = wp.array(pbc_np, dtype=wp.bool, device=device)

    # Cell as (1,) array of mat33 — kernel computes volume / lengths internally
    cell_arr = wp.array(
        cell_np[None, :, :].astype(np_dtype), dtype=mat_dtype, device=device
    )

    # Moment arrays (float64, zero-initialized). mz, mz2 are (B, 3)
    # with projected M and M2 stored in the non-periodic axis slot.
    mz = wp.zeros((1, 3), dtype=wp.float64, device=device)
    mz2 = wp.zeros((1, 3), dtype=wp.float64, device=device)
    qtotal = wp.zeros(1, dtype=wp.float64, device=device)

    # Output arrays
    energy_in = wp.zeros(N, dtype=wp.float64, device=device)
    energy_out = wp.zeros(N, dtype=wp.float64, device=device)
    forces = wp.zeros(N, dtype=vec_dtype, device=device)
    charge_grads = wp.zeros(N, dtype=wp.float64, device=device)
    virial = wp.zeros(1, dtype=mat_dtype, device=device)

    return {
        "positions": positions,
        "charges": charges,
        "batch_idx": batch_idx,
        "pbc": pbc,
        "cell": cell_arr,
        "axis": axis,
        "mz": mz,
        "mz2": mz2,
        "qtotal": qtotal,
        "energy_in": energy_in,
        "energy_out": energy_out,
        "forces": forces,
        "charge_grads": charge_grads,
        "virial": virial,
        "wp_dtype": wp_dtype,
    }


def _run_kernels(w):
    """Launch both kernels with kwargs from _make_warp_arrays output."""
    slab_reduce_moments(
        positions=w["positions"],
        charges=w["charges"],
        batch_idx=w["batch_idx"],
        pbc=w["pbc"],
        cell=w["cell"],
        mz=w["mz"],
        mz2=w["mz2"],
        qtotal=w["qtotal"],
        wp_dtype=w["wp_dtype"],
    )
    slab_correction(
        positions=w["positions"],
        charges=w["charges"],
        batch_idx=w["batch_idx"],
        pbc=w["pbc"],
        cell=w["cell"],
        mz=w["mz"],
        mz2=w["mz2"],
        qtotal=w["qtotal"],
        energy_in=w["energy_in"],
        energy_out=w["energy_out"],
        forces=w["forces"],
        charge_grads=w["charge_grads"],
        virial=w["virial"],
        wp_dtype=w["wp_dtype"],
    )
    wp.synchronize()


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def slab_system_z():
    """4-atom slab system, non-periodic along z.

    Positions and charges chosen so M_z, M_z2, Q are all nonzero
    to exercise all terms in the correction formula.

    Cell: 10 x 10 x 30 (large z for vacuum gap).
    """
    positions = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [2.0, 3.0, 1.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )
    charges = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float64)
    cell = np.array(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]],
        dtype=np.float64,
    )
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "axis": 2,
        "num_atoms": 4,
    }


@pytest.fixture(scope="session")
def non_neutral_system():
    """3-atom non-neutral system (Q != 0), non-periodic along z.

    Tests the Ballenegger background charge correction terms.
    """
    positions = np.array(
        [
            [0.0, 0.0, 1.0],
            [5.0, 5.0, 4.0],
            [2.5, 2.5, 7.0],
        ],
        dtype=np.float64,
    )
    charges = np.array([1.0, 1.0, -0.5], dtype=np.float64)  # Q = 1.5
    cell = np.array(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]],
        dtype=np.float64,
    )
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "axis": 2,
        "num_atoms": 3,
    }


# ==============================================================================
# Test 1: Moment reduction correctness
# ==============================================================================


class TestMomentReduction:
    """Test that moment reduction kernel computes M_z, M_z2, Q correctly."""

    @pytest.mark.parametrize("wp_dtype", [wp.float32, wp.float64])
    def test_moments(self, slab_system_z, wp_dtype, device):
        w = _make_warp_arrays(slab_system_z, wp_dtype, device)

        slab_reduce_moments(
            positions=w["positions"],
            charges=w["charges"],
            batch_idx=w["batch_idx"],
            pbc=w["pbc"],
            cell=w["cell"],
            mz=w["mz"],
            mz2=w["mz2"],
            qtotal=w["qtotal"],
            wp_dtype=wp_dtype,
        )
        wp.synchronize()

        # Expected values from numpy
        axis = slab_system_z["axis"]
        z = slab_system_z["positions"] @ _slab_normal(slab_system_z["cell"], axis)
        q = slab_system_z["charges"]
        expected_mz = np.sum(q * z)
        expected_mz2 = np.sum(q * z**2)
        expected_q = np.sum(q)

        # mz is now (B, 3): mz[s, axis] is the slab-axis dipole.
        # mz2 is (B, 3): mz2[s, axis] is the axis-axis quadrupole.
        rtol = 1e-5 if wp_dtype == wp.float32 else 1e-12
        np.testing.assert_allclose(w["mz"].numpy()[0, axis], expected_mz, rtol=rtol)
        np.testing.assert_allclose(w["mz2"].numpy()[0, axis], expected_mz2, rtol=rtol)
        np.testing.assert_allclose(w["qtotal"].numpy()[0], expected_q, rtol=rtol)

    def test_moments_non_neutral(self, non_neutral_system, device):
        w = _make_warp_arrays(non_neutral_system, wp.float64, device)

        slab_reduce_moments(
            positions=w["positions"],
            charges=w["charges"],
            batch_idx=w["batch_idx"],
            pbc=w["pbc"],
            cell=w["cell"],
            mz=w["mz"],
            mz2=w["mz2"],
            qtotal=w["qtotal"],
            wp_dtype=wp.float64,
        )
        wp.synchronize()

        q = non_neutral_system["charges"]
        axis = non_neutral_system["axis"]
        z = non_neutral_system["positions"] @ _slab_normal(
            non_neutral_system["cell"], axis
        )
        np.testing.assert_allclose(w["qtotal"].numpy()[0], np.sum(q), rtol=1e-12)
        np.testing.assert_allclose(w["mz"].numpy()[0, axis], np.sum(q * z), rtol=1e-12)
        np.testing.assert_allclose(
            w["mz2"].numpy()[0, axis], np.sum(q * z**2), rtol=1e-12
        )

    def test_moments_3d_periodic_zero(self, slab_system_z, device):
        """For pbc=[T, T, T] (3D periodic), no contribution to moments."""
        w = _make_warp_arrays(slab_system_z, wp.float64, device)
        # Override pbc with all True
        pbc_3d = wp.array(
            np.array([[True, True, True]], dtype=np.bool_),
            dtype=wp.bool,
            device=device,
        )
        slab_reduce_moments(
            positions=w["positions"],
            charges=w["charges"],
            batch_idx=w["batch_idx"],
            pbc=pbc_3d,
            cell=w["cell"],
            mz=w["mz"],
            mz2=w["mz2"],
            qtotal=w["qtotal"],
            wp_dtype=wp.float64,
        )
        wp.synchronize()
        np.testing.assert_allclose(w["mz"].numpy()[0], 0.0, atol=1e-15)
        np.testing.assert_allclose(w["mz2"].numpy()[0], 0.0, atol=1e-15)
        np.testing.assert_allclose(w["qtotal"].numpy()[0], 0.0, atol=1e-15)
        # mz, mz2 are now (B, 3) — full row should be zero for 3D pbc.


# ==============================================================================
# Test 2: Per-atom energy correctness
# ==============================================================================


class TestSlabEnergy:
    """Test per-atom slab correction energy against analytical formula."""

    @pytest.mark.parametrize("wp_dtype", [wp.float32, wp.float64])
    def test_energy(self, slab_system_z, wp_dtype, device):
        w = _make_warp_arrays(slab_system_z, wp_dtype, device)
        _run_kernels(w)

        expected_e, _, _, _ = analytical_slab_correction(
            slab_system_z["positions"],
            slab_system_z["charges"],
            slab_system_z["cell"],
            slab_system_z["axis"],
        )

        rtol = 1e-5 if wp_dtype == wp.float32 else 1e-12
        np.testing.assert_allclose(w["energy_out"].numpy(), expected_e, rtol=rtol)

    def test_energy_non_neutral(self, non_neutral_system, device):
        w = _make_warp_arrays(non_neutral_system, wp.float64, device)
        _run_kernels(w)

        expected_e, _, _, _ = analytical_slab_correction(
            non_neutral_system["positions"],
            non_neutral_system["charges"],
            non_neutral_system["cell"],
            non_neutral_system["axis"],
        )
        np.testing.assert_allclose(w["energy_out"].numpy(), expected_e, rtol=1e-12)


# ==============================================================================
# Test 3: Per-atom force correctness
# ==============================================================================


class TestSlabForce:
    """Test per-atom slab correction force against analytical formula."""

    @pytest.mark.parametrize("wp_dtype", [wp.float32, wp.float64])
    def test_force(self, slab_system_z, wp_dtype, device):
        w = _make_warp_arrays(slab_system_z, wp_dtype, device)
        _run_kernels(w)

        _, expected_f, _, _ = analytical_slab_correction(
            slab_system_z["positions"],
            slab_system_z["charges"],
            slab_system_z["cell"],
            slab_system_z["axis"],
        )

        rtol = 1e-5 if wp_dtype == wp.float32 else 1e-12
        actual_f = w["forces"].numpy()
        np.testing.assert_allclose(actual_f, expected_f, rtol=rtol, atol=1e-15)

        # Verify periodic axes have zero force
        periodic_axes = [a for a in range(3) if a != slab_system_z["axis"]]
        for ax in periodic_axes:
            np.testing.assert_allclose(
                actual_f[:, ax],
                0.0,
                atol=1e-15,
                err_msg=f"Force along periodic axis {ax} should be zero",
            )


# ==============================================================================
# Test 3b: Triclinic projected-normal geometry
# ==============================================================================


class TestTriclinicGeometry:
    """Test slab correction for tilted periodic planes."""

    def test_triclinic_outputs(self, device):
        """Triclinic energy/force/charge-grad/virial match the reference."""
        positions = np.array(
            [[1.0, 2.0, 3.0], [4.0, 1.5, 6.0], [2.0, 3.5, 7.5]],
            dtype=np.float64,
        )
        charges = np.array([1.0, -0.5, 0.3], dtype=np.float64)
        cell = np.array(
            [[9.0, 0.0, 0.0], [2.0, 8.0, 1.5], [0.5, 0.2, 25.0]],
            dtype=np.float64,
        )
        system = {
            "positions": positions,
            "charges": charges,
            "cell": cell,
            "axis": 2,
            "num_atoms": 3,
        }

        w = _make_warp_arrays(system, wp.float64, device)
        _run_kernels(w)

        expected_e, expected_f, expected_cg, expected_v = analytical_slab_correction(
            positions, charges, cell, 2
        )
        normal = _slab_normal(cell, 2)

        np.testing.assert_allclose(
            w["mz"].numpy()[0, 2], np.sum(charges * (positions @ normal)), rtol=1e-12
        )
        np.testing.assert_allclose(
            w["mz2"].numpy()[0, 2],
            np.sum(charges * (positions @ normal) ** 2),
            rtol=1e-12,
        )
        np.testing.assert_allclose(w["energy_out"].numpy(), expected_e, rtol=1e-12)
        np.testing.assert_allclose(
            w["forces"].numpy(), expected_f, rtol=1e-12, atol=1e-15
        )
        np.testing.assert_allclose(w["charge_grads"].numpy(), expected_cg, rtol=1e-12)
        np.testing.assert_allclose(
            w["virial"].numpy()[0], expected_v.sum(axis=0), rtol=1e-12, atol=1e-15
        )


# ==============================================================================
# Test 4: Float32 vs float64 consistency
# ==============================================================================


class TestDtypeConsistency:
    """Test that float32 and float64 give consistent results."""

    def test_energy_dtype_consistency(self, slab_system_z, device):
        results = {}
        for wp_dtype in [wp.float32, wp.float64]:
            w = _make_warp_arrays(slab_system_z, wp_dtype, device)
            _run_kernels(w)
            results[wp_dtype] = w["energy_out"].numpy()

        np.testing.assert_allclose(results[wp.float32], results[wp.float64], rtol=1e-5)


# ==============================================================================
# Test 5: Non-periodic axis selection (x, y, z)
# ==============================================================================


class TestAxisSelection:
    """Test that the correction works for all three axis choices."""

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_axis(self, axis, device):
        """Rotate the same physical system so the non-periodic axis changes."""
        # Base system: slab along z
        base_positions = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.0, 3.0, 1.0]],
            dtype=np.float64,
        )
        base_charges = np.array([1.0, -1.0, 0.5], dtype=np.float64)
        base_cell = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]],
            dtype=np.float64,
        )

        # Rotate: cycle axes so that 'axis' becomes the non-periodic one
        perm = [(axis + 1) % 3, (axis + 2) % 3, axis]
        positions = base_positions[:, perm]
        cell = base_cell[np.ix_(perm, perm)]

        system = {
            "positions": positions,
            "charges": base_charges,
            "cell": cell,
            "axis": axis,
            "num_atoms": 3,
        }

        w = _make_warp_arrays(system, wp.float64, device)
        _run_kernels(w)

        expected_e, expected_f, _, _ = analytical_slab_correction(
            positions, base_charges, cell, axis
        )

        np.testing.assert_allclose(w["energy_out"].numpy(), expected_e, rtol=1e-12)
        np.testing.assert_allclose(
            w["forces"].numpy(), expected_f, rtol=1e-12, atol=1e-15
        )


# ==============================================================================
# Test 6: Neutral vs non-neutral systems
# ==============================================================================


class TestNeutralSystem:
    """Test that Q-dependent terms vanish for neutral systems."""

    def test_neutral_q_terms_vanish(self, device):
        positions = np.array(
            [[0.0, 0.0, 2.0], [0.0, 0.0, 5.0], [0.0, 0.0, 8.0], [0.0, 0.0, 3.0]],
            dtype=np.float64,
        )
        charges = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float64)
        assert np.isclose(charges.sum(), 0.0), "System must be neutral"

        cell = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]],
            dtype=np.float64,
        )

        system = {
            "positions": positions,
            "charges": charges,
            "cell": cell,
            "axis": 2,
            "num_atoms": 4,
        }

        w = _make_warp_arrays(system, wp.float64, device)
        _run_kernels(w)

        # Verify Q is zero
        np.testing.assert_allclose(w["qtotal"].numpy()[0], 0.0, atol=1e-15)

        expected_e, expected_f, expected_cg, _ = analytical_slab_correction(
            positions, charges, cell, 2
        )
        np.testing.assert_allclose(w["energy_out"].numpy(), expected_e, rtol=1e-12)
        np.testing.assert_allclose(
            w["forces"].numpy(), expected_f, rtol=1e-12, atol=1e-15
        )
        np.testing.assert_allclose(w["charge_grads"].numpy(), expected_cg, rtol=1e-12)


# ==============================================================================
# Test 7: Batch vs individual consistency
# ==============================================================================


class TestBatchConsistency:
    """Test that running systems as a batch gives same results as individually."""

    def test_batch_matches_individual(self, device):
        # System A
        pos_a = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0]], dtype=np.float64)
        q_a = np.array([1.0, -1.0], dtype=np.float64)
        cell_a = np.array(
            [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 24.0]], dtype=np.float64
        )

        # System B
        pos_b = np.array(
            [[1.0, 1.0, 3.0], [2.0, 2.0, 7.0], [3.0, 3.0, 1.0]], dtype=np.float64
        )
        q_b = np.array([0.5, -0.3, 0.2], dtype=np.float64)
        cell_b = np.array(
            [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 36.0]], dtype=np.float64
        )

        axis = 2
        wp_dtype = wp.float64

        # --- Run individually ---
        individual_energies = []
        individual_forces = []
        for pos, q, cell in [(pos_a, q_a, cell_a), (pos_b, q_b, cell_b)]:
            N = len(q)
            system = {
                "positions": pos,
                "charges": q,
                "cell": cell,
                "axis": axis,
                "num_atoms": N,
            }
            w = _make_warp_arrays(system, wp_dtype, device)
            _run_kernels(w)
            individual_energies.append(w["energy_out"].numpy().copy())
            individual_forces.append(w["forces"].numpy().copy())

        # --- Run as batch ---
        batch_pos = np.concatenate([pos_a, pos_b], axis=0)
        batch_q = np.concatenate([q_a, q_b])
        batch_idx_np = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        N_total = len(batch_q)

        wp_positions = wp.array(batch_pos, dtype=wp.vec3d, device=device)
        wp_charges = wp.array(batch_q, dtype=wp.float64, device=device)
        wp_batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        # Per-system pbc — both systems have slab along z
        pbc_np = np.array([[True, True, False], [True, True, False]], dtype=np.bool_)
        wp_pbc = wp.array(pbc_np, dtype=wp.bool, device=device)

        # Cell as (B,) array of mat33
        cell_batch = np.stack([cell_a, cell_b], axis=0).astype(np.float64)
        wp_cell = wp.array(cell_batch, dtype=wp.mat33d, device=device)

        wp_mz = wp.zeros((2, 3), dtype=wp.float64, device=device)
        wp_mz2 = wp.zeros((2, 3), dtype=wp.float64, device=device)
        wp_qtotal = wp.zeros(2, dtype=wp.float64, device=device)
        wp_energy_in = wp.zeros(N_total, dtype=wp.float64, device=device)
        wp_energy_out = wp.zeros(N_total, dtype=wp.float64, device=device)
        wp_forces = wp.zeros(N_total, dtype=wp.vec3d, device=device)
        wp_charge_grads = wp.zeros(N_total, dtype=wp.float64, device=device)
        wp_virial = wp.zeros(2, dtype=wp.mat33d, device=device)

        slab_reduce_moments(
            wp_positions,
            wp_charges,
            wp_batch_idx,
            wp_pbc,
            wp_cell,
            wp_mz,
            wp_mz2,
            wp_qtotal,
            wp_dtype,
        )
        slab_correction(
            wp_positions,
            wp_charges,
            wp_batch_idx,
            wp_pbc,
            wp_cell,
            wp_mz,
            wp_mz2,
            wp_qtotal,
            wp_energy_in,
            wp_energy_out,
            wp_forces,
            wp_charge_grads,
            wp_virial,
            wp_dtype,
        )
        wp.synchronize()

        batch_energies = wp_energy_out.numpy()
        batch_forces = wp_forces.numpy()

        # Compare system A
        np.testing.assert_allclose(
            batch_energies[:2], individual_energies[0], rtol=1e-12
        )
        np.testing.assert_allclose(
            batch_forces[:2], individual_forces[0], rtol=1e-12, atol=1e-15
        )

        # Compare system B
        np.testing.assert_allclose(
            batch_energies[2:], individual_energies[1], rtol=1e-12
        )
        np.testing.assert_allclose(
            batch_forces[2:], individual_forces[1], rtol=1e-12, atol=1e-15
        )


# ==============================================================================
# Test 8: Mixed-axis batches (system 0 has slab in z, system 1 has slab in y)
# ==============================================================================


class TestMixedAxisBatch:
    """Test a batch where different systems have different non-periodic axes."""

    def test_mixed_axis(self, device):
        wp_dtype = wp.float64

        # System A: slab in z
        pos_a = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 5.0]], dtype=np.float64)
        q_a = np.array([1.0, -1.0], dtype=np.float64)
        cell_a = np.array(
            [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 24.0]], dtype=np.float64
        )

        # System B: triclinic slab in y
        pos_b = np.array(
            [[1.0, 3.0, 0.5], [2.0, 7.0, 1.5], [3.0, 1.0, 2.5]], dtype=np.float64
        )
        q_b = np.array([0.5, -0.3, 0.2], dtype=np.float64)
        cell_b = np.array(
            [[12.0, 0.8, 1.0], [0.2, 36.0, 1.0], [1.5, 0.4, 12.0]],
            dtype=np.float64,
        )

        # --- Compute analytical reference ---
        e_ref_a, f_ref_a, _, _ = analytical_slab_correction(pos_a, q_a, cell_a, 2)
        e_ref_b, f_ref_b, _, _ = analytical_slab_correction(pos_b, q_b, cell_b, 1)

        # --- Run as a mixed-axis batch ---
        batch_pos = np.concatenate([pos_a, pos_b], axis=0)
        batch_q = np.concatenate([q_a, q_b])
        batch_idx_np = np.array([0, 0, 1, 1, 1], dtype=np.int32)
        N_total = len(batch_q)

        wp_positions = wp.array(batch_pos, dtype=wp.vec3d, device=device)
        wp_charges = wp.array(batch_q, dtype=wp.float64, device=device)
        wp_batch_idx = wp.array(batch_idx_np, dtype=wp.int32, device=device)

        # Per-system pbc: system 0 slab in z, system 1 slab in y
        pbc_np = np.array([[True, True, False], [True, False, True]], dtype=np.bool_)
        wp_pbc = wp.array(pbc_np, dtype=wp.bool, device=device)

        # Cell as (B,) array of mat33
        cell_batch = np.stack([cell_a, cell_b], axis=0).astype(np.float64)
        wp_cell = wp.array(cell_batch, dtype=wp.mat33d, device=device)

        wp_mz = wp.zeros((2, 3), dtype=wp.float64, device=device)
        wp_mz2 = wp.zeros((2, 3), dtype=wp.float64, device=device)
        wp_qtotal = wp.zeros(2, dtype=wp.float64, device=device)
        wp_energy_in = wp.zeros(N_total, dtype=wp.float64, device=device)
        wp_energy_out = wp.zeros(N_total, dtype=wp.float64, device=device)
        wp_forces = wp.zeros(N_total, dtype=wp.vec3d, device=device)
        wp_charge_grads = wp.zeros(N_total, dtype=wp.float64, device=device)
        wp_virial = wp.zeros(2, dtype=wp.mat33d, device=device)

        slab_reduce_moments(
            wp_positions,
            wp_charges,
            wp_batch_idx,
            wp_pbc,
            wp_cell,
            wp_mz,
            wp_mz2,
            wp_qtotal,
            wp_dtype,
        )
        slab_correction(
            wp_positions,
            wp_charges,
            wp_batch_idx,
            wp_pbc,
            wp_cell,
            wp_mz,
            wp_mz2,
            wp_qtotal,
            wp_energy_in,
            wp_energy_out,
            wp_forces,
            wp_charge_grads,
            wp_virial,
            wp_dtype,
        )
        wp.synchronize()

        e_out = wp_energy_out.numpy()
        f_out = wp_forces.numpy()

        # System A (slab in z): atoms 0-1
        np.testing.assert_allclose(e_out[:2], e_ref_a, rtol=1e-12)
        np.testing.assert_allclose(f_out[:2], f_ref_a, rtol=1e-12, atol=1e-15)

        # System B (slab in y): atoms 2-4
        np.testing.assert_allclose(e_out[2:], e_ref_b, rtol=1e-12)
        np.testing.assert_allclose(f_out[2:], f_ref_b, rtol=1e-12, atol=1e-15)

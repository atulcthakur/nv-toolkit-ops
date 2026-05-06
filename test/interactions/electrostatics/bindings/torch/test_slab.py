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
Integration tests for slab correction (Yeh-Berkowitz / Ballenegger Eq. 29).

Coverage:
- LAMMPS reference energy for CsCl slab
- Cross-validation against torch-pme (energies + forces)
- Analytical forces/charge_grads/virial vs autograd
- 3D periodic = zero correction
- Triclinic projected-normal geometry
- Standalone apply_slab_correction() API agrees with ewald_summation integration
"""

from __future__ import annotations

import pytest
import torch

from nvalchemiops.torch.interactions.electrostatics import apply_slab_correction
from nvalchemiops.torch.interactions.electrostatics.ewald import ewald_summation
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.neighbors import cell_list

try:
    from torchpme import EwaldCalculator
    from torchpme.potentials import CoulombPotential

    HAS_TORCHPME = True
except ModuleNotFoundError:
    HAS_TORCHPME = False
    EwaldCalculator = None
    CoulombPotential = None

KCALMOL_PER_ANGSTROM = 332.0637132991921
EWALD_ALPHA = 0.3
EWALD_K_CUTOFF = 8.0
TRICLINIC_EWALD_K_CUTOFF = 7.0
REAL_SPACE_CUTOFF = 5.0


# ==============================================================================
# Helpers
# ==============================================================================


def _make_cscl_slab_system(dtype=torch.float64, device="cpu"):
    """Create CsCl slab system matching torch-pme's LAMMPS test.

    2 atoms, cell=[10, 10, 30], pbc=[True, True, False] (slab in z).
    """
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=dtype, device=device
    )
    charges = torch.tensor([1.0, -1.0], dtype=dtype, device=device)
    cell = torch.diag(
        torch.tensor([10.0, 10.0, 30.0], dtype=dtype, device=device)
    ).unsqueeze(0)  # (1, 3, 3)
    pbc = torch.tensor([True, True, False], device=device)  # (3,)
    return positions, charges, cell, pbc


def _make_triclinic_slab_system(dtype=torch.float64, device="cpu"):
    """Create a small non-neutral triclinic T/T/F slab system."""
    positions = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 1.5, 6.0], [2.0, 3.5, 7.5]],
        dtype=dtype,
        device=device,
    )
    charges = torch.tensor([1.0, -0.5, 0.3], dtype=dtype, device=device)
    cell = torch.tensor(
        [[9.0, 0.0, 0.0], [2.0, 8.0, 1.5], [0.5, 0.2, 25.0]],
        dtype=dtype,
        device=device,
    ).unsqueeze(0)
    pbc = torch.tensor([True, True, False], device=device)
    return positions, charges, cell, pbc


def _make_cscl_ewald_inputs(dtype=torch.float64, device="cpu"):
    """Create CsCl slab inputs plus the full-3D real-space neighbor list."""
    positions, charges, cell, pbc = _make_cscl_slab_system(dtype, device)
    neighbor_list, neighbor_ptr, neighbor_shifts = _build_neighbor_list(
        positions, cell, REAL_SPACE_CUTOFF, [True, True, True], device
    )
    return positions, charges, cell, pbc, neighbor_list, neighbor_ptr, neighbor_shifts


def _make_triclinic_ewald_inputs(dtype=torch.float64, device="cpu"):
    """Create triclinic slab inputs plus the full-3D real-space neighbor list."""
    positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)
    neighbor_list, neighbor_ptr, neighbor_shifts = _build_neighbor_list(
        positions, cell, REAL_SPACE_CUTOFF, [True, True, True], device
    )
    return positions, charges, cell, pbc, neighbor_list, neighbor_ptr, neighbor_shifts


def _build_neighbor_list(positions, cell, cutoff, pbc_full3d, device="cpu"):
    """Build neighbor list using cell_list with full 3D pbc.

    The slab correction handles the 2D periodicity separately; for the
    real-space neighbor list we use full 3D periodicity (the vacuum gap
    in z guarantees no real-space neighbors leak across).
    """
    pbc_tensor = torch.tensor(pbc_full3d, dtype=torch.bool, device=device).unsqueeze(0)
    neighbor_list, neighbor_ptr, unit_shifts = cell_list(
        positions, cutoff, cell, pbc_tensor, return_neighbor_list=True
    )
    return neighbor_list, neighbor_ptr, unit_shifts


def _run_torchpme_ewald(positions, charges, cell_2d, pbc, alpha, k_cutoff):
    """Run torch-pme EwaldCalculator for cross-validation.

    Parameters
    ----------
    positions, charges : torch.Tensor
    cell_2d : torch.Tensor, shape (3, 3)
        Cell matrix without a batch dimension.
    pbc : torch.Tensor, shape (3,) bool
    alpha : float
        Ewald splitting parameter (toolkit-ops convention).
    k_cutoff : float
        K-vector cutoff.

    Returns
    -------
    potential : torch.Tensor, shape (N, 1)
        Per-atom potential from torch-pme.
    """
    import math

    dtype = positions.dtype
    device = positions.device

    # Convert alpha (toolkit-ops convention) to torch-pme smearing
    smearing = 1.0 / (math.sqrt(2.0) * alpha)
    lr_wavelength = 2.0 * math.pi / k_cutoff

    potential = CoulombPotential(smearing=smearing)
    calculator = EwaldCalculator(
        potential=potential,
        lr_wavelength=lr_wavelength,
        full_neighbor_list=True,
    ).to(device=device, dtype=dtype)

    charges_2d = charges.unsqueeze(-1)

    # Full pairwise neighbor list for this small system
    N = len(positions)
    i_indices = []
    j_indices = []
    for i in range(N):
        for j in range(N):
            if i != j:
                i_indices.append(i)
                j_indices.append(j)
    neighbor_indices = torch.tensor(
        [i_indices, j_indices], dtype=torch.int64, device=device
    ).T
    diff = positions[neighbor_indices[:, 1]] - positions[neighbor_indices[:, 0]]
    neighbor_distances = torch.norm(diff, dim=1)

    return calculator.forward(
        charges=charges_2d,
        cell=cell_2d,
        positions=positions,
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        periodic=pbc,
    )


def _reference_slab_correction(positions, charges, cell, pbc):
    """Independent Torch reference for projected-normal slab correction."""
    cell_2d = cell.squeeze(0) if cell.dim() == 3 else cell
    pbc_1d = pbc.squeeze(0) if pbc.dim() == 2 else pbc
    nonperiodic_axis = int(torch.nonzero(~pbc_1d, as_tuple=False).flatten()[0])

    periodic_a = cell_2d[(nonperiodic_axis + 1) % 3]
    periodic_b = cell_2d[(nonperiodic_axis + 2) % 3]
    normal = torch.cross(periodic_a, periodic_b, dim=0)
    normal = normal / torch.linalg.norm(normal)

    z = positions @ normal
    volume = torch.abs(torch.linalg.det(cell_2d))
    height_sq = torch.dot(cell_2d[nonperiodic_axis], normal) ** 2
    qtotal = charges.sum()
    moment = torch.sum(charges * z)
    moment2 = torch.sum(charges * z * z)

    bracket = z * moment - 0.5 * (moment2 + qtotal * z * z) - qtotal * height_sq / 12.0
    energies = (2.0 * torch.pi / volume) * charges * bracket
    forces = (-(4.0 * torch.pi / volume) * charges * (moment - qtotal * z)).unsqueeze(
        -1
    ) * normal
    charge_grads = (4.0 * torch.pi / volume) * bracket
    projector = torch.eye(
        3, dtype=positions.dtype, device=positions.device
    ) - 2.0 * torch.outer(normal, normal)
    virial = energies.sum() * projector
    return energies, forces, charge_grads, virial


# ==============================================================================
# LAMMPS reference energy
# ==============================================================================


class TestLAMMPSReference:
    """CsCl slab energy should match LAMMPS value of -383.44635 kcal/mol/A."""

    def test_lammps_cscl_slab_energy(self, device):
        dtype = torch.float64
        lammps_energy = -383.44635  # kcal/mol/A

        positions, charges, cell, pbc, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )

        energies = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            pbc=pbc,
            slab_correction=True,
        )

        total_energy_kcal = energies.sum() * KCALMOL_PER_ANGSTROM
        # Total slab-corrected Ewald energy matches the LAMMPS reference.
        torch.testing.assert_close(
            total_energy_kcal,
            torch.tensor(lammps_energy, dtype=dtype, device=device),
            rtol=1e-3,
            atol=0.0,
        )


# ==============================================================================
# Cross-validation against torch-pme
# ==============================================================================


class TestTorchPMECrossValidation:
    """Cross-validate slab correction against torch-pme EwaldCalculator."""

    @pytest.mark.skipif(not HAS_TORCHPME, reason="torch-pme not installed")
    def test_outputs_match_torchpme(self, device):
        """Energy and forces match torch-pme for the CsCl slab."""
        dtype = torch.float64

        positions, charges, cell, pbc, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )

        our_energies, our_forces = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            pbc=pbc,
            slab_correction=True,
        )

        positions_tp = positions.clone().detach().requires_grad_(True)
        torchpme_potential = _run_torchpme_ewald(
            positions_tp,
            charges,
            cell.squeeze(0),
            pbc,
            EWALD_ALPHA,
            EWALD_K_CUTOFF,
        )
        torchpme_total = (torchpme_potential.squeeze(-1) * charges).sum()
        torchpme_forces = -torch.autograd.grad(
            torchpme_total, positions_tp, create_graph=False
        )[0]

        # Total slab-corrected Ewald energy matches torch-pme.
        torch.testing.assert_close(
            our_energies.sum(), torchpme_total, rtol=1e-5, atol=0.0
        )
        # Total slab-corrected Ewald forces match torch-pme autograd forces.
        torch.testing.assert_close(our_forces, torchpme_forces, rtol=1e-4, atol=1e-8)


# ==============================================================================
# Analytical kernel outputs vs autograd
# ==============================================================================


class TestAnalyticalVsAutograd:
    """Analytical kernel outputs should match autograd derivatives."""

    def test_forces_and_charge_grads_vs_autograd(self, device):
        dtype = torch.float64

        positions, charges, cell, pbc, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )
        positions = positions.clone().detach().requires_grad_(True)
        charges = charges.clone().detach().requires_grad_(True)

        energies, analytical_forces, charge_grads = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            pbc=pbc,
            slab_correction=True,
        )

        autograd_positions, autograd_charge_grads = torch.autograd.grad(
            energies.sum(), (positions, charges), create_graph=False
        )
        autograd_forces = -autograd_positions

        # Analytical total Ewald forces match autograd forces.
        torch.testing.assert_close(
            analytical_forces, autograd_forces, rtol=1e-6, atol=1e-10
        )
        # Analytical total Ewald charge gradients match autograd charge gradients.
        torch.testing.assert_close(
            charge_grads,
            autograd_charge_grads,
            rtol=1e-6,
            atol=1e-10,
        )

    def test_virial_full_vs_autograd(self, device):
        """Full 3x3 slab virial should match -dE/d(eps) via autograd strain.

        Strain: r_i -> (I + eps) r_i, h_a -> (I + eps) h_a; W = -dE/deps at eps=0.
        The kernel emits the normal-following virial W = E * (I - 2 n n^T).
        """
        dtype = torch.float64

        positions, charges, cell, pbc = _make_cscl_slab_system(dtype, device)

        # Analytical virial via the kernel
        _, analytical_virial = apply_slab_correction(
            positions, charges, cell, pbc, compute_virial=True
        )
        # (1, 3, 3) -> (3, 3) for comparison
        analytical_W = analytical_virial[0]

        # Autograd via a strain perturbation
        eps = torch.zeros(3, 3, dtype=dtype, device=device, requires_grad=True)
        I3 = torch.eye(3, dtype=dtype, device=device)
        F = I3 + eps  # deformation gradient
        positions_strained = positions @ F.T
        cell_strained = cell @ F.T

        e_slab_total = apply_slab_correction(
            positions_strained, charges, cell_strained, pbc
        ).sum()
        autograd_W = -torch.autograd.grad(e_slab_total, eps)[0]

        # Analytical slab-only virial matches strain autograd.
        torch.testing.assert_close(
            analytical_W,
            autograd_W,
            rtol=1e-6,
            atol=1e-10,
        )

    def test_virial_full_non_neutral(self, device):
        """Full virial test with a non-neutral, non-aligned-charge system."""
        dtype = torch.float64

        # Non-neutral system with off-axis coordinates. For an axis-aligned
        # T/T/F cell, normal-following virial has zero shear components for
        # this strain convention.
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [2.0, 3.0, 7.0]],
            dtype=dtype,
            device=device,
        )
        charges = torch.tensor([1.0, -0.5, 0.3], dtype=dtype, device=device)
        cell = torch.diag(
            torch.tensor([10.0, 10.0, 30.0], dtype=dtype, device=device)
        ).unsqueeze(0)
        pbc = torch.tensor([True, True, False], device=device)

        e_slab, analytical_virial = apply_slab_correction(
            positions, charges, cell, pbc, compute_virial=True
        )
        analytical_W = analytical_virial[0]

        eps = torch.zeros(3, 3, dtype=dtype, device=device, requires_grad=True)
        F = torch.eye(3, dtype=dtype, device=device) + eps
        e_total = apply_slab_correction(positions @ F.T, charges, cell @ F.T, pbc).sum()
        autograd_W = -torch.autograd.grad(e_total, eps)[0]

        # Non-neutral slab-only virial matches strain autograd.
        torch.testing.assert_close(
            analytical_W,
            autograd_W,
            rtol=1e-6,
            atol=1e-10,
        )


# ==============================================================================
# 3D periodic = zero correction
# ==============================================================================


class TestZeroCorrection:
    """3D periodic systems should give identical results with/without slab_correction."""

    def test_pbc_3d_matches_no_apply_slab_correction(self, device):
        dtype = torch.float64

        positions, charges, cell, _, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )

        kwargs = dict(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=EWALD_ALPHA,
            k_cutoff=EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
        )

        # No slab correction (default)
        energies_off, forces_off = ewald_summation(**kwargs)

        # Slab correction enabled but pbc is 3D -> no contribution
        pbc_3d = torch.tensor([True, True, True], device=device)
        energies_3d, forces_3d = ewald_summation(
            **kwargs, pbc=pbc_3d, slab_correction=True
        )

        # 3D periodic slab correction leaves per-atom energies unchanged.
        torch.testing.assert_close(energies_3d, energies_off, rtol=0, atol=0)
        # 3D periodic slab correction leaves forces unchanged.
        torch.testing.assert_close(forces_3d, forces_off, rtol=0, atol=0)


# ==============================================================================
# Triclinic cells
# ==============================================================================


class TestTriclinicCells:
    """Triclinic slab cells use projected-normal geometry."""

    def test_triclinic_standalone_matches_reference(self, device):
        dtype = torch.float64

        positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)

        energies, forces, charge_grads, virial = apply_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        ref_e, ref_f, ref_cg, ref_v = _reference_slab_correction(
            positions, charges, cell, pbc
        )

        # Triclinic standalone slab energies match the independent reference.
        torch.testing.assert_close(energies, ref_e, rtol=1e-12, atol=1e-15)
        # Triclinic standalone slab forces match the independent reference.
        torch.testing.assert_close(forces, ref_f, rtol=1e-12, atol=1e-15)
        # Triclinic standalone slab charge gradients match the reference.
        torch.testing.assert_close(charge_grads, ref_cg, rtol=1e-12, atol=1e-15)
        # Triclinic standalone slab virial matches the independent reference.
        torch.testing.assert_close(virial[0], ref_v, rtol=1e-12, atol=1e-15)

    def test_triclinic_forces_and_charge_grads_match_autograd(self, device):
        dtype = torch.float64

        positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)
        positions = positions.clone().detach().requires_grad_(True)
        charges = charges.clone().detach().requires_grad_(True)

        energies, analytical_forces, analytical_charge_grads = apply_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
        )
        autograd_positions, autograd_charge_grads = torch.autograd.grad(
            energies.sum(), (positions, charges)
        )
        autograd_forces = -autograd_positions

        # Triclinic slab forces match autograd forces.
        torch.testing.assert_close(
            analytical_forces, autograd_forces, rtol=1e-6, atol=1e-10
        )
        # Triclinic slab charge gradients match autograd charge gradients.
        torch.testing.assert_close(
            analytical_charge_grads,
            autograd_charge_grads,
            rtol=1e-6,
            atol=1e-10,
        )

    def test_triclinic_virial_matches_autograd_strain(self, device):
        dtype = torch.float64

        positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)

        _, analytical_virial = apply_slab_correction(
            positions, charges, cell, pbc, compute_virial=True
        )
        eps = torch.zeros(3, 3, dtype=dtype, device=device, requires_grad=True)
        F = torch.eye(3, dtype=dtype, device=device) + eps
        e_total = apply_slab_correction(positions @ F.T, charges, cell @ F.T, pbc).sum()
        autograd_virial = -torch.autograd.grad(e_total, eps)[0]

        # Triclinic slab virial matches strain autograd.
        torch.testing.assert_close(
            analytical_virial[0], autograd_virial, rtol=1e-6, atol=1e-10
        )

    def test_triclinic_full_ewald_matches_decomposition(self, device):
        dtype = torch.float64

        positions, charges, cell, pbc, nl, ptr, shifts = _make_triclinic_ewald_inputs(
            dtype, device
        )

        e_full, f_full, cg_full, v_full = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=TRICLINIC_EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
            pbc=pbc,
            slab_correction=True,
        )
        e_3d, f_3d, cg_3d, v_3d = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=TRICLINIC_EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        e_slab, f_slab, cg_slab, v_slab = apply_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )

        # Full triclinic Ewald energies equal 3D Ewald plus slab correction.
        torch.testing.assert_close(e_full, e_3d + e_slab, rtol=1e-12, atol=1e-15)
        # Full triclinic Ewald forces equal 3D Ewald plus slab correction.
        torch.testing.assert_close(f_full, f_3d + f_slab, rtol=1e-10, atol=1e-12)
        # Full triclinic Ewald charge gradients equal 3D Ewald plus slab correction.
        torch.testing.assert_close(cg_full, cg_3d + cg_slab, rtol=1e-10, atol=1e-12)
        # Full triclinic Ewald virial equals 3D Ewald plus slab correction.
        torch.testing.assert_close(v_full, v_3d + v_slab, rtol=1e-10, atol=1e-12)

    def test_triclinic_full_ewald_matches_autograd(self, device):
        dtype = torch.float64

        positions, charges, cell, pbc, nl, ptr, shifts = _make_triclinic_ewald_inputs(
            dtype, device
        )
        positions = positions.clone().detach().requires_grad_(True)
        charges = charges.clone().detach().requires_grad_(True)

        energies, forces, charge_grads = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=TRICLINIC_EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_charge_gradients=True,
            pbc=pbc,
            slab_correction=True,
        )
        autograd_positions, autograd_charge_grads = torch.autograd.grad(
            energies.sum(), (positions, charges), create_graph=False
        )
        autograd_forces = -autograd_positions

        # Full triclinic Ewald forces match autograd forces.
        torch.testing.assert_close(forces, autograd_forces, rtol=2e-6, atol=2e-8)
        # Full triclinic Ewald charge gradients match autograd charge gradients.
        torch.testing.assert_close(
            charge_grads, autograd_charge_grads, rtol=1e-6, atol=1e-10
        )

        _, _, virial = ewald_summation(
            positions.detach(),
            charges.detach(),
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=TRICLINIC_EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_virial=True,
            pbc=pbc,
            slab_correction=True,
        )
        eps = torch.zeros(3, 3, dtype=dtype, device=device, requires_grad=True)
        deformation = torch.eye(3, dtype=dtype, device=device) + eps
        e_strained = ewald_summation(
            positions.detach() @ deformation.T,
            charges.detach(),
            cell @ deformation.T,
            alpha=EWALD_ALPHA,
            k_cutoff=TRICLINIC_EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            pbc=pbc,
            slab_correction=True,
        ).sum()
        autograd_virial = -torch.autograd.grad(e_strained, eps)[0]

        # Full triclinic Ewald virial matches strain autograd.
        torch.testing.assert_close(virial[0], autograd_virial, rtol=3e-6, atol=1e-7)

    def test_triclinic_translation_invariance_non_neutral(self, device):
        dtype = torch.float64

        positions, charges, cell, pbc = _make_triclinic_slab_system(dtype, device)
        shift = torch.tensor([1.3, -0.7, 2.1], dtype=dtype, device=device)

        e0, f0, cg0 = apply_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
        )
        e1, f1, cg1 = apply_slab_correction(
            positions + shift,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        # Non-neutral triclinic slab total energy is translation invariant.
        torch.testing.assert_close(e1.sum(), e0.sum(), rtol=1e-12, atol=1e-15)
        # Non-neutral triclinic slab forces are translation invariant.
        torch.testing.assert_close(f1, f0, rtol=1e-12, atol=1e-15)
        # Non-neutral triclinic slab charge gradients are translation invariant.
        torch.testing.assert_close(cg1, cg0, rtol=1e-12, atol=1e-15)


# ==============================================================================
# pbc=None handling
# ==============================================================================


class TestPbcNoneHandling:
    """slab_correction=True without pbc must raise a clear error."""

    def test_missing_pbc_raises(self, device):
        dtype = torch.float64

        positions, charges, cell, _ = _make_cscl_slab_system(dtype, device)
        nl, ptr, shifts = _build_neighbor_list(
            positions, cell, 5.0, [True, True, True], device
        )

        with pytest.raises(ValueError, match="pbc"):
            ewald_summation(
                positions,
                charges,
                cell,
                alpha=EWALD_ALPHA,
                k_cutoff=EWALD_K_CUTOFF,
                neighbor_list=nl,
                neighbor_ptr=ptr,
                neighbor_shifts=shifts,
                slab_correction=True,  # no pbc provided
            )


# ==============================================================================
# Standalone apply_slab_correction() API tests
# ==============================================================================


class TestStandaloneSlabAPI:
    """Standalone apply_slab_correction() should produce the same delta that
    ewald_summation(slab_correction=True) applies internally."""

    def test_standalone_matches_integrated(self, device):
        dtype = torch.float64

        positions, charges, cell, pbc, nl, ptr, shifts = _make_cscl_ewald_inputs(
            dtype, device
        )

        # Integrated: with slab correction
        e_with, f_with = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            pbc=pbc,
            slab_correction=True,
        )

        # Without slab correction
        e_without, f_without = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_cutoff=EWALD_K_CUTOFF,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
        )

        # Standalone slab correction
        e_slab, f_slab = apply_slab_correction(
            positions, charges, cell, pbc, compute_forces=True
        )

        # Integrated slab energy delta equals the standalone slab correction.
        torch.testing.assert_close(e_with - e_without, e_slab, rtol=1e-12, atol=1e-15)
        # Integrated slab force delta equals the standalone slab correction.
        torch.testing.assert_close(f_with - f_without, f_slab, rtol=1e-10, atol=1e-12)

    def test_standalone_outputs_subset(self, device):
        """Standalone API should return the right tuple based on flags."""
        dtype = torch.float64

        positions, charges, cell, pbc = _make_cscl_slab_system(dtype, device)

        # Energy only -> single tensor
        out = apply_slab_correction(positions, charges, cell, pbc)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (positions.shape[0],)

        # Energy + forces
        out = apply_slab_correction(positions, charges, cell, pbc, compute_forces=True)
        assert isinstance(out, tuple)
        assert len(out) == 2
        assert out[1].shape == positions.shape

        # Energy + forces + charge grads + virial
        out = apply_slab_correction(
            positions,
            charges,
            cell,
            pbc,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        assert isinstance(out, tuple)
        assert len(out) == 4
        e, f, cg, v = out
        assert e.shape == (positions.shape[0],)
        assert f.shape == positions.shape
        assert cg.shape == (positions.shape[0],)
        assert v.shape == (1, 3, 3)

    def test_standalone_pbc_broadcast(self, device):
        """A (3,) pbc tensor should broadcast across the batch."""
        dtype = torch.float64

        positions, charges, cell, _ = _make_cscl_slab_system(dtype, device)

        pbc_1d = torch.tensor([True, True, False], device=device)
        pbc_2d = torch.tensor([[True, True, False]], device=device)

        e_1d = apply_slab_correction(positions, charges, cell, pbc_1d)
        e_2d = apply_slab_correction(positions, charges, cell, pbc_2d)

        # Broadcasted 1D and explicit 2D pbc produce identical energies.
        torch.testing.assert_close(e_1d, e_2d, rtol=0, atol=0)


# ==============================================================================
# Hybrid forces integration
# ==============================================================================


class TestSlabHybridForces:
    """Slab correction must respect ewald_summation hybrid_forces semantics."""

    def _ewald_inputs(self, positions, cell):
        """Build detached geometry inputs for focused hybrid tests."""
        device = positions.device
        nl, ptr, shifts = _build_neighbor_list(
            positions.detach(),
            cell.detach(),
            REAL_SPACE_CUTOFF,
            [True, True, True],
            device,
        )
        k_vectors = generate_k_vectors_ewald_summation(cell.detach(), EWALD_K_CUTOFF)
        return nl, ptr, shifts, k_vectors

    def test_hybrid_slab_positions_and_cell_no_grad(self, device):
        """Hybrid slab energy must not attach position/cell autograd paths."""
        dtype = torch.float64

        positions, charges, cell, pbc = _make_cscl_slab_system(dtype, device)
        positions = positions.clone().requires_grad_(True)
        charges = charges.clone().requires_grad_(True)
        cell = cell.clone().requires_grad_(True)
        nl, ptr, shifts, k_vectors = self._ewald_inputs(positions, cell)

        energies = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            pbc=pbc,
            slab_correction=True,
            hybrid_forces=True,
        )
        energies.sum().backward()

        assert positions.grad is None or torch.all(positions.grad == 0)
        assert cell.grad is None or torch.all(cell.grad == 0)
        assert charges.grad is not None
        assert torch.isfinite(charges.grad).all()

    def test_hybrid_slab_charge_grad_matches_standard(self, device):
        """Injected hybrid charge gradients include the slab contribution."""
        dtype = torch.float64

        positions, charges_ref, cell, pbc = _make_cscl_slab_system(dtype, device)
        nl, ptr, shifts, k_vectors = self._ewald_inputs(positions, cell)

        charges_std = charges_ref.clone().requires_grad_(True)
        e_std = ewald_summation(
            positions,
            charges_std,
            cell,
            alpha=EWALD_ALPHA,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            pbc=pbc,
            slab_correction=True,
        )
        grad_std = torch.autograd.grad(e_std.sum(), charges_std)[0]

        charges_hyb = charges_ref.clone().requires_grad_(True)
        e_hyb = ewald_summation(
            positions,
            charges_hyb,
            cell,
            alpha=EWALD_ALPHA,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            pbc=pbc,
            slab_correction=True,
            hybrid_forces=True,
        )
        grad_hyb = torch.autograd.grad(e_hyb.sum(), charges_hyb)[0]

        # Hybrid injected charge gradients match standard autograd gradients.
        torch.testing.assert_close(grad_hyb, grad_std, rtol=1e-6, atol=1e-10)

    def test_hybrid_slab_forces_match_standard(self, device):
        """Hybrid forward forces include the same slab force as standard mode."""
        dtype = torch.float64

        positions, charges, cell, pbc = _make_cscl_slab_system(dtype, device)
        nl, ptr, shifts, k_vectors = self._ewald_inputs(positions, cell)

        e_std, f_std = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            pbc=pbc,
            slab_correction=True,
        )
        e_hyb, f_hyb = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            pbc=pbc,
            slab_correction=True,
            hybrid_forces=True,
        )

        # Hybrid forward energies match standard mode energies.
        torch.testing.assert_close(e_hyb, e_std, rtol=1e-12, atol=1e-15)
        # Hybrid forward forces match standard mode forces.
        torch.testing.assert_close(f_hyb, f_std, rtol=1e-10, atol=1e-12)

    def test_hybrid_slab_virial_forward_only(self, device):
        """Hybrid virial values match standard mode and remain forward-only."""
        dtype = torch.float64

        positions, charges, cell, pbc = _make_cscl_slab_system(dtype, device)
        nl, ptr, shifts, k_vectors = self._ewald_inputs(positions, cell)

        _, _, v_std = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_virial=True,
            pbc=pbc,
            slab_correction=True,
        )
        _, _, v_hyb = ewald_summation(
            positions,
            charges,
            cell,
            alpha=EWALD_ALPHA,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=ptr,
            neighbor_shifts=shifts,
            compute_forces=True,
            compute_virial=True,
            pbc=pbc,
            slab_correction=True,
            hybrid_forces=True,
        )

        # Hybrid forward virial matches standard mode virial.
        torch.testing.assert_close(v_hyb, v_std, rtol=1e-10, atol=1e-12)
        assert v_hyb.grad_fn is None

# 3D Multiphase-Field Verification Report

- Summary: 20/20 cases passed
- Main physics: 9/9 passed
- Quick sanity: 11/11 passed
- Artifact root: `tests\output\latest`
- CSV: `tests\output\latest\verification_results.csv`
- Compile-time KMAX: `50`
- Threads per block: `(4, 4, 4)`

## Notes

- Loaded base parameters from config_3d.yaml.
- config_3d.yaml gpu.KMAX=18 differs from compile-time KMAX=50; verification uses compile-time KMAX.
- gpu.MAX_GRAINS=20 is smaller than compile-time KMAX=50; verification allocates APT arrays with depth max(MAX_GRAINS, KMAX, number_of_grain).

## Main Physics

| Case | Status | Level | Interface mean | Interface max | Volume growth | Winner/Loser | Root shift | Front velocity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [single_grain_preferred_growth_benchmark](tests/output/latest/single_grain_preferred_growth_benchmark) | PASS | L2 | 1.756e-03 | 1.900e-03 | 1.110e-09 | - | - | 2.891e+00 |
| [directional_preference_map](tests/output/latest/directional_preference_map) | PASS | L2 | 1.756e-03 | 1.900e-03 | 1.110e-09 | - | - | 2.891e+00 |
| [anisotropy_threshold_test](tests/output/latest/anisotropy_threshold_test) | PASS | L2 | - | - | - | - | - | 9.375e-02 |
| [bicrystal_competition_low_high_driving](tests/output/latest/bicrystal_competition_low_high_driving) | PASS | L1 | 1.783e-03 | 1.900e-03 | 1.215e-09 | 1.017e+00 | 9.395e-07 | 2.833e+00 |
| [bicrystal_interfacial_energy_dominated_regime_test](tests/output/latest/bicrystal_interfacial_energy_dominated_regime_test) | PASS | L2 | 6.611e-04 | 7.000e-04 | 2.003e-11 | 1.027e+00 | 1.896e-06 | 1.587e-02 |
| [bicrystal_kinetic_dominated_regime_test](tests/output/latest/bicrystal_kinetic_dominated_regime_test) | PASS | L2 | 2.150e-03 | 2.300e-03 | 1.625e-09 | 1.068e+00 | -2.134e-04 | 3.000e+00 |
| [multigrain_competition_extension](tests/output/latest/multigrain_competition_extension) | PASS | L1 | 2.167e-03 | 2.300e-03 | 3.645e-09 | 1.103e+00 | - | 3.050e+00 |
| [groove_depth_only](tests/output/latest/groove_depth_only) | PASS | L2 | 6.450e-04 | 7.000e-04 | -1.663e-11 | 1.015e+00 | 0.000e+00 | 0.000e+00 |
| [robust_groove_angle_estimation](tests/output/latest/robust_groove_angle_estimation) | PASS | L2 | 6.450e-04 | 7.000e-04 | -1.663e-11 | 1.015e+00 | 0.000e+00 | 0.000e+00 |

## Quick Sanity

| Case | Status | Level | Interface mean | Interface max | Volume growth | Winner/Loser | Root shift | Front velocity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [basic_constraints](tests/output/latest/basic_constraints) | PASS | L0 | 1.300e-03 | 1.300e-03 | - | - | - | - |
| [kmax_overflow_detection](tests/output/latest/kmax_overflow_detection) | PASS | L0 | 1.000e-04 | 1.000e-04 | - | - | - | - |
| [static_flat_interface](tests/output/latest/static_flat_interface) | PASS | L0 | 5.000e-04 | 5.000e-04 | - | - | - | - |
| [two_d_limit_consistency](tests/output/latest/two_d_limit_consistency) | PASS | L0 | 1.300e-03 | 1.300e-03 | - | - | - | - |
| [boundary_conditions](tests/output/latest/boundary_conditions) | PASS | L0 | 1.333e-04 | 2.000e-04 | - | - | - | - |
| [isotropic_orientation_independence](tests/output/latest/isotropic_orientation_independence) | PASS | L0 | 1.300e-03 | 1.300e-03 | - | - | - | - |
| [anisotropic_preferred_growth](tests/output/latest/anisotropic_preferred_growth) | PASS | L0 | 5.375e-04 | 7.000e-04 | - | - | - | - |
| [torque_term_contribution](tests/output/latest/torque_term_contribution) | PASS | L0 | 5.375e-04 | 7.000e-04 | - | - | - | - |
| [grain_competition](tests/output/latest/grain_competition) | PASS | L0 | 1.500e-03 | 1.500e-03 | - | - | - | - |
| [grain_boundary_groove_trijunction](tests/output/latest/grain_boundary_groove_trijunction) | PASS | L0 | 5.333e-04 | 6.000e-04 | - | - | - | - |
| [convergence_grid_dt_delta](tests/output/latest/convergence_grid_dt_delta) | PASS | L0 | 5.000e-04 | 6.000e-04 | - | - | - | - |

## Case Notes

### single_grain_preferred_growth_benchmark

- Category: main
- Status: PASS
- Physical level: L2
- Summary: Single-grain preferred growth spread_on=9.38e-02 m/s vs isotropic spread_off=0.00e+00 m/s; best=identity, worst=aligned_111_to_z, aligned penalty=9.38e-02 m/s.
- Artifacts: `tests\output\latest\single_grain_preferred_growth_benchmark`
- Key metrics: phi_sum_error_max=-, solid_fraction=-, surviving_grains=-1, interface_mean=1.756e-03, interface_max=1.900e-03, winner_ratio=-, root_shift=-, interface_angle=-, front_velocity=2.891e+00
- Extra: `{"spread_on": 0.09375, "spread_off": 0.0, "best_orientation": "identity", "worst_orientation": "aligned_111_to_z", "best_front_velocity": 2.8906249999999996, "worst_front_velocity": 2.7968749999999996, "aligned_front_velocity": 2.7968749999999996, "aligned_penalty": 0.09375}`

### directional_preference_map

- Category: main
- Status: PASS
- Physical level: L2
- Summary: Directional map spreads: low=3.13e-02, high=9.38e-02; aligned penalty grew 3.13e-02 -> 9.38e-02 m/s.
- Artifacts: `tests\output\latest\directional_preference_map`
- Key metrics: phi_sum_error_max=-, solid_fraction=-, surviving_grains=-1, interface_mean=1.756e-03, interface_max=1.900e-03, winner_ratio=-, root_shift=-, interface_angle=-, front_velocity=2.891e+00
- Extra: `{"low": {"best_orientation": "x_30", "worst_orientation": "aligned_111_to_z", "spread": 0.03125000000000011, "iso_spread": 0.0, "aligned_penalty": 0.03125000000000011}, "high": {"best_orientation": "identity", "worst_orientation": "aligned_111_to_z", "spread": 0.09375, "iso_spread": 0.0, "aligned_penalty": 0.09375}}`

### anisotropy_threshold_test

- Category: main
- Status: PASS
- Physical level: L2
- Summary: Directional contrast grew from 0.00e+00 to 9.38e-02 as delta_a increased.
- Artifacts: `tests\output\latest\anisotropy_threshold_test`
- Key metrics: phi_sum_error_max=-, solid_fraction=-, surviving_grains=-1, interface_mean=-, interface_max=-, winner_ratio=-, root_shift=-, interface_angle=-, front_velocity=9.375e-02
- Extra: `{"rows": [{"delta_a": 0.0, "front_velocity_diff": 0.0}, {"delta_a": 0.1, "front_velocity_diff": 0.0625}, {"delta_a": 0.2, "front_velocity_diff": 0.078125}, {"delta_a": 0.35, "front_velocity_diff": 0.09375}, {"delta_a": 0.5, "front_velocity_diff": 0.09375}], "monotonic_non_decreasing": true}`

### bicrystal_competition_low_high_driving

- Category: main
- Status: PASS
- Physical level: L1
- Summary: Bicrystal winner switched aligned_111_to_z -> identity between low/high driving.
- Artifacts: `tests\output\latest\bicrystal_competition_low_high_driving`
- Key metrics: phi_sum_error_max=5.960e-08, solid_fraction=6.699e-01, surviving_grains=2, interface_mean=1.783e-03, interface_max=1.900e-03, winner_ratio=1.017e+00, root_shift=9.395e-07, interface_angle=4.407e+01, front_velocity=2.833e+00
- Extra: `{"pair": {"low_orientation": "aligned_111_to_z", "high_orientation": "identity"}, "low_winner": "aligned_111_to_z", "high_winner": "identity", "low_ratio": 1.0131704029452953, "high_ratio": 1.0173857938590862, "low_root_shift": 0.0, "high_root_shift": 9.39527617044611e-07}`

### bicrystal_interfacial_energy_dominated_regime_test

- Category: main
- Status: PASS
- Physical level: L2
- Summary: Low-driving bicrystal favored aligned_111_to_z with front-velocity gain 1.59e-02 m/s and ratio gain 4.83e-03 over anisotropy-off.
- Artifacts: `tests\output\latest\bicrystal_interfacial_energy_dominated_regime_test`
- Key metrics: phi_sum_error_max=5.960e-08, solid_fraction=2.747e-01, surviving_grains=2, interface_mean=6.611e-04, interface_max=7.000e-04, winner_ratio=1.027e+00, root_shift=1.896e-06, interface_angle=2.874e+01, front_velocity=1.587e-02
- Extra: `{"expected_low_winner": "aligned_111_to_z", "winner_on": "aligned_111_to_z", "winner_off": "aligned_111_to_z", "ratio_on": 1.027248586896309, "ratio_off": 1.022420320701637, "front_velocity_on": 0.01587301587301578, "front_velocity_off": 0.0, "front_velocity_gain": 0.01587301587301578, "root_shift_on": 1.8956693022987637e-06, "root_shift_off": -1.4901161193847657e-12, "root_shift_gain": 1.8956678121826443e-06}`

### bicrystal_kinetic_dominated_regime_test

- Category: main
- Status: PASS
- Physical level: L2
- Summary: High-driving bicrystal favored identity with ratio gain 5.75e-02 over the anisotropy-off baseline.
- Artifacts: `tests\output\latest\bicrystal_kinetic_dominated_regime_test`
- Key metrics: phi_sum_error_max=5.960e-08, solid_fraction=8.055e-01, surviving_grains=2, interface_mean=2.150e-03, interface_max=2.300e-03, winner_ratio=1.068e+00, root_shift=-2.134e-04, interface_angle=2.939e+01, front_velocity=3.000e+00
- Extra: `{"expected_high_winner": "identity", "winner_on": "identity", "winner_off": "aligned_111_to_z", "ratio_on": 1.0677204538628355, "ratio_off": 1.0102149658592796, "root_shift_on": -0.00021341969124695145, "root_shift_off": 3.421803468966083e-05, "root_switch": true}`

### multigrain_competition_extension

- Category: main
- Status: PASS
- Physical level: L1
- Summary: Multigrain extension switched dominant family aligned_111_to_z -> identity; winner ratios were 1.0066 and 1.1027.
- Artifacts: `tests\output\latest\multigrain_competition_extension`
- Key metrics: phi_sum_error_max=1.192e-07, solid_fraction=8.104e-01, surviving_grains=6, interface_mean=2.167e-03, interface_max=2.300e-03, winner_ratio=1.103e+00, root_shift=-, interface_angle=-, front_velocity=3.050e+00
- Extra: `{"low_family_name": "aligned_111_to_z", "high_family_name": "identity", "low_group": {"volume_a": 9.092395725218694e-10, "volume_b": 9.033093439914097e-10, "winner_loser_volume_ratio": 1.0065650029747906, "winner_family": "A"}, "high_group": {"volume_a": 2.590022085691146e-09, "volume_b": 2.8559957176987893e-09, "winner_loser_volume_ratio": 1.1026916463288259, "winner_family": "B"}, "low_surviving_grains": 6, "high_surviving_grains": 6}`

### groove_depth_only

- Category: main
- Status: PASS
- Physical level: L2
- Summary: Groove depth benchmark gave depth=1.04e-04 m.
- Artifacts: `tests\output\latest\groove_depth_only`
- Key metrics: phi_sum_error_max=0.000e+00, solid_fraction=3.070e-01, surviving_grains=2, interface_mean=6.450e-04, interface_max=7.000e-04, winner_ratio=1.015e+00, root_shift=0.000e+00, interface_angle=1.920e+01, front_velocity=0.000e+00
- Extra: `{"groove_depth": 0.00010447949201567762, "groove_metrics": {"root_x_cells": 9.5, "root_z_cells": 6.140266745898822, "trijunction_x_cells": 9.5, "trijunction_z_cells": 6.140266745898822, "left_angle_deg": -10.57752338207936, "right_angle_deg": 8.623655974516568, "left_fit_rmse": 0.006562300317471668, "right_fit_rmse": 0.00436683999097148, "groove_depth_cells": 1.044794920156776, "dihedral_angle_like_deg": 19.20117935659593}}`

### robust_groove_angle_estimation

- Category: main
- Status: PASS
- Physical level: L2
- Summary: Robust groove fit produced dihedral-like angle=19.20 deg with fit_rmse=6.56e-03 cells.
- Artifacts: `tests\output\latest\robust_groove_angle_estimation`
- Key metrics: phi_sum_error_max=0.000e+00, solid_fraction=3.070e-01, surviving_grains=2, interface_mean=6.450e-04, interface_max=7.000e-04, winner_ratio=1.015e+00, root_shift=0.000e+00, interface_angle=1.920e+01, front_velocity=0.000e+00
- Extra: `{"groove_metrics": {"root_x_cells": 9.5, "root_z_cells": 6.140266745898822, "trijunction_x_cells": 9.5, "trijunction_z_cells": 6.140266745898822, "left_angle_deg": -10.57752338207936, "right_angle_deg": 8.623655974516568, "left_fit_rmse": 0.006562300317471668, "right_fit_rmse": 0.00436683999097148, "groove_depth_cells": 1.044794920156776, "dihedral_angle_like_deg": 19.20117935659593}, "angle_history_deg": [14.459181221271223, 14.661731290308964, 15.035149861886143, 15.530506939995853, 16.110621306914986, 16.736929566158693, 17.379949177165766, 18.017309905623254, 18.629417911288115, 19.20117935659593], "fit_rmse": 0.006562300317471668}`

### basic_constraints

- Category: quick
- Status: PASS
- Physical level: L0
- Summary: Short multi-grain run stayed bounded with phi_sum_error_max=5.96e-08 and nf_max=5.
- Artifacts: `tests\output\latest\basic_constraints`
- Key metrics: phi_sum_error_max=5.960e-08, solid_fraction=9.999e-01, surviving_grains=4, interface_mean=1.300e-03, interface_max=1.300e-03, winner_ratio=-, root_shift=-, interface_angle=-, front_velocity=-
- Extra: `{"enable_anisotropy": true, "enable_torque": true}`

### kmax_overflow_detection

- Category: quick
- Status: PASS
- Physical level: L0
- Summary: KMAX guard detected an overflow request of 52 active phases.
- Artifacts: `tests\output\latest\kmax_overflow_detection`
- Key metrics: phi_sum_error_max=0.000e+00, solid_fraction=1.532e-02, surviving_grains=0, interface_mean=1.000e-04, interface_max=1.000e-04, winner_ratio=-, root_shift=-, interface_angle=-, front_velocity=-
- Extra: `{"status": [1, 52], "overflow_info": {"location": [0, 1, 1], "count": 52, "active_ids_preview": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}, "requested_number_of_grain": 52}`

### static_flat_interface

- Category: quick
- Status: PASS
- Physical level: L0
- Summary: Flat interface mean shift=1.00e-04 m, roughness=0.00e+00 m with anisotropy=False, torque=False.
- Artifacts: `tests\output\latest\static_flat_interface`
- Key metrics: phi_sum_error_max=0.000e+00, solid_fraction=3.931e-01, surviving_grains=1, interface_mean=5.000e-04, interface_max=5.000e-04, winner_ratio=-, root_shift=-, interface_angle=-, front_velocity=-
- Extra: `{"initial_interface_mean": 0.0004, "final_interface_mean": 0.0005, "mean_shift": 9.999999999999999e-05}`

### two_d_limit_consistency

- Category: quick
- Status: PASS
- Physical level: L0
- Summary: 3D mid-y slice matched 2D reference with L_inf=0.00e+00, L2=0.00e+00.
- Artifacts: `tests\output\latest\two_d_limit_consistency`
- Key metrics: phi_sum_error_max=0.000e+00, solid_fraction=1.000e+00, surviving_grains=1, interface_mean=1.300e-03, interface_max=1.300e-03, winner_ratio=-, root_shift=-, interface_angle=-, front_velocity=-
- Extra: `{"phi_linf_vs_2d": 0.0, "phi_l2_vs_2d": 0.0}`

### boundary_conditions

- Category: quick
- Status: PASS
- Physical level: L0
- Summary: Boundary-condition probes: x_periodic_wrap=True, y_periodic_wrap=True, z_mirror_adjacent=True, z_no_wrap=True
- Artifacts: `tests\output\latest\boundary_conditions`
- Key metrics: phi_sum_error_max=0.000e+00, solid_fraction=2.000e-02, surviving_grains=0, interface_mean=1.333e-04, interface_max=2.000e-04, winner_ratio=-, root_shift=-, interface_angle=-, front_velocity=-
- Extra: `{"checks": {"x_periodic_wrap": true, "y_periodic_wrap": true, "z_mirror_adjacent": true, "z_no_wrap": true}}`

### isotropic_orientation_independence

- Category: quick
- Status: PASS
- Physical level: L0
- Summary: Isotropic orientation sensitivity stayed small: mean_diff=0.00e+00 m, solid_diff=0.00e+00, L2=0.00e+00.
- Artifacts: `tests\output\latest\isotropic_orientation_independence`
- Key metrics: phi_sum_error_max=0.000e+00, solid_fraction=1.000e+00, surviving_grains=1, interface_mean=1.300e-03, interface_max=1.300e-03, winner_ratio=-, root_shift=-, interface_angle=-, front_velocity=-
- Extra: `{"interface_mean_diff": 0.0, "solid_fraction_diff": 0.0, "phi_l2_diff": 0.0}`

### anisotropic_preferred_growth

- Category: quick
- Status: PASS
- Physical level: L0
- Summary: Anisotropy created orientation-dependent evolution: |delta mean|=0.00e+00 m, roughness_diff=0.00e+00 m, L2=1.01e-02; aligned111 advanced further.
- Artifacts: `tests\output\latest\anisotropic_preferred_growth`
- Key metrics: phi_sum_error_max=0.000e+00, solid_fraction=3.495e-01, surviving_grains=1, interface_mean=5.375e-04, interface_max=7.000e-04, winner_ratio=-, root_shift=-, interface_angle=-, front_velocity=-
- Extra: `{"interface_mean_identity": 0.0005375, "interface_mean_aligned111": 0.0005375, "interface_mean_abs_diff": 0.0, "roughness_diff": 0.0, "phi_l2_diff": 0.010136204771697521, "preferred_orientation": "aligned111", "enable_torque": false}`

### torque_term_contribution

- Category: quick
- Status: PASS
- Physical level: L0
- Summary: Torque changed the solution by L2=4.95e-03, roughness_diff=0.00e+00 m.
- Artifacts: `tests\output\latest\torque_term_contribution`
- Key metrics: phi_sum_error_max=0.000e+00, solid_fraction=3.517e-01, surviving_grains=1, interface_mean=5.375e-04, interface_max=7.000e-04, winner_ratio=-, root_shift=-, interface_angle=-, front_velocity=-
- Extra: `{"phi_l2_diff": 0.004951349459588528, "roughness_diff": 0.0, "anisotropy_enabled": true}`

### grain_competition

- Category: quick
- Status: PASS
- Physical level: L0
- Summary: Competition changed grain volumes by max 3.57e-10 m^3 and surviving grains 6->6.
- Artifacts: `tests\output\latest\grain_competition`
- Key metrics: phi_sum_error_max=1.192e-07, solid_fraction=9.998e-01, surviving_grains=6, interface_mean=1.500e-03, interface_max=1.500e-03, winner_ratio=-, root_shift=-, interface_angle=-, front_velocity=-
- Extra: `{"max_volume_change": 3.5733681291727493e-10, "top_surface_phase_count_initial": 0, "top_surface_phase_count_final": 6}`

### grain_boundary_groove_trijunction

- Category: quick
- Status: PASS
- Physical level: L0
- Summary: GB groove depth=5.00e-05 m, groove angle=0.00 deg.
- Artifacts: `tests\output\latest\grain_boundary_groove_trijunction`
- Key metrics: phi_sum_error_max=0.000e+00, solid_fraction=3.545e-01, surviving_grains=3, interface_mean=5.333e-04, interface_max=6.000e-04, winner_ratio=-, root_shift=-, interface_angle=0.000e+00, front_velocity=-
- Extra: `{"groove_depth": 5e-05, "angle_split1_deg": 0.0, "angle_split2_deg": 0.0}`

### convergence_grid_dt_delta

- Category: quick
- Status: PASS
- Physical level: L0
- Summary: Convergence sanity check: grid_error=1.06e-01, dt_error=1.06e-01, delta_sensitivity=5.61e-02, interface_diffs=(3.13e-05, 0.00e+00, 0.00e+00).
- Artifacts: `tests\output\latest\convergence_grid_dt_delta`
- Key metrics: phi_sum_error_max=0.000e+00, solid_fraction=3.815e-01, surviving_grains=1, interface_mean=5.000e-04, interface_max=6.000e-04, winner_ratio=-, root_shift=-, interface_angle=-, front_velocity=-
- Extra: `{"grid_error": 0.10639186203479767, "dt_error": 0.10627716034650803, "delta_sensitivity": 0.0561206117272377, "grid_interface_diff": 3.125000000000003e-05, "dt_interface_diff": 0.0, "delta_interface_diff": 0.0, "fine_runtime_s": 0.010281499940901995}`

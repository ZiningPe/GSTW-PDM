"""
gstw_pdm
========
Spatial Durbin panels with same-period and cross-period weights that may be
endogenous (Wang, "Generalized Spatio-Temporally Weighted Panel Data Model").

Model (time-major rows):

    Y = rho_0 W_0 Y + rho_1 W_1 Y + X beta + W_0 X theta_0 + W_1 X theta_1 + V,
    V = delta_c A eps + xi,   h = Z_W pi + eps,

with W_0 the same-period blocks and W_1 the cross-period blocks of the weight
matrix, and h the variable that forms the weights.

  * networks      : KroneckerNetwork (M kron W_S), RecursiveNetwork (only the past
                    transmits, Lemma 2), DenseNetwork (any W), TiedNetwork
  * L = N         : cf_qmle (Theorems 1-2, two-step variance), cf_2sls
  * unit effects  : dummy_qmle, dummy_drift (Proposition 2), aqs_fit (Theorem 3), cf_2sls_fe
  * L = T         : within_period_qmle (Theorem 4), anderson_rubin (Appendix I)
  * effects       : effects_inference (Section 6) with BH / BY control
"""

from .weight_construction import (
    rook_weights, queen_weights, inverse_distance_weights, knn_weights,
    compute_morans_i, compute_gearys_c, compute_getis_ord_g, compute_spatial_gini,
    build_twm_from_stats, build_twm_parametric,
    kronecker_weights, block_row_normalise, parameter_space, stwm_summary,
    eigvals_kronecker, logdet_kronecker,
)

from .first_stage import first_stage, first_stage_stats, first_stage_fit, projection_matrix

from .control_function import aggregate_eps_hat, aggregation_matrix, detect_mode

from .multi_w import (
    split_same_cross, city_demean,
    RecursiveNetwork, DenseNetwork, KroneckerNetwork, TiedNetwork, make_network,
)

from .cf_qmle import cf_qmle, qmle, build_design, profile_loglik, wald_delta_c

from .cf_2sls import cf_2sls, cf_2sls_design, cf_2sls_summary

from .period_level import within_period_qmle, anderson_rubin, ar_confidence_set, complement_basis

from .fixed_effects import (
    first_stage_fe, joint_pi_covariance, WithinDesign, dummy_qmle, dummy_drift,
    aqs_solve, aqs_variance, aqs_fit, cf_2sls_fe, wald_test, restriction_matrix,
)

from .inference import (
    spillover_effects, effects_inference,
    fdr_correction, bh_correction, by_correction, print_ie_table,
)

from .utils import (
    ned_test, eigs_sorted, spectral_radius, check_admissibility,
    to_time_major, to_unit_major, panel_demean, matrix_condition, check_instrument_strength,
)

from .report import stars, regression_table, comparison_table, tests_block, horizon_table

__version__ = "0.3.0"
__author__ = "Wang Zining"
__all__ = [
    # weights
    "rook_weights", "queen_weights", "inverse_distance_weights", "knn_weights",
    "compute_morans_i", "compute_gearys_c", "compute_getis_ord_g", "compute_spatial_gini",
    "build_twm_from_stats", "build_twm_parametric",
    "kronecker_weights", "block_row_normalise", "parameter_space", "stwm_summary",
    "eigvals_kronecker", "logdet_kronecker",
    # first stage and control function
    "first_stage", "first_stage_stats", "first_stage_fit", "projection_matrix",
    "aggregate_eps_hat", "aggregation_matrix", "detect_mode",
    # networks
    "split_same_cross", "city_demean",
    "RecursiveNetwork", "DenseNetwork", "KroneckerNetwork", "TiedNetwork", "make_network",
    # L = N estimators
    "cf_qmle", "qmle", "build_design", "profile_loglik", "wald_delta_c",
    "cf_2sls", "cf_2sls_design", "cf_2sls_summary",
    # L = T
    "within_period_qmle", "anderson_rubin", "ar_confidence_set", "complement_basis",
    # unit fixed effects at fixed T
    "first_stage_fe", "joint_pi_covariance", "WithinDesign", "dummy_qmle", "dummy_drift",
    "aqs_solve", "aqs_variance", "aqs_fit", "cf_2sls_fe", "wald_test", "restriction_matrix",
    # effects and multiple testing
    "spillover_effects", "effects_inference",
    "fdr_correction", "bh_correction", "by_correction", "print_ie_table",
    # utils
    "ned_test", "eigs_sorted", "spectral_radius", "check_admissibility",
    "to_time_major", "to_unit_major", "panel_demean", "matrix_condition", "check_instrument_strength",
    # reporting
    "stars", "regression_table", "comparison_table", "tests_block", "horizon_table",
]

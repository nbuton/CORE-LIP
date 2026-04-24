"""
core_lip/features.py
--------------------
Canonical feature-name lists shared by training, inference, and evaluation.

Import these lists wherever you need to build or consume a ProteinDataset so
that the feature set never drifts between scripts.
"""

SCALAR_FEATURES: list[str] = [
    "asphericity_mean",
    "asphericity_std",
    "avg_maximum_diameter",
    "avg_squared_Ree",
    "gyration_eigenvalues_l1_mean",
    "gyration_eigenvalues_l1_std",
    "gyration_eigenvalues_l2_mean",
    "gyration_eigenvalues_l2_std",
    "gyration_eigenvalues_l3_mean",
    "gyration_eigenvalues_l3_std",
    "gyration_l1_per_l2_mean",
    "gyration_l1_per_l2_std",
    "gyration_l1_per_l3_mean",
    "gyration_l1_per_l3_std",
    "gyration_l2_per_l3_mean",
    "gyration_l2_per_l3_std",
    "normalized_acylindricity_mean",
    "normalized_acylindricity_std",
    "prolateness_mean",
    "prolateness_std",
    "radius_of_gyration_mean",
    "radius_of_gyration_std",
    "rel_shape_anisotropy_mean",
    "rel_shape_anisotropy_std",
    "scaling_exponent",
    "std_maximum_diameter",
    "std_squared_Ree",
]

LOCAL_FEATURES: list[str] = [
    "phi_entropy",
    "psi_entropy",
    "sasa_abs_mean",
    "sasa_abs_std",
    "sasa_rel_mean",
    "sasa_rel_std",
    "ss_propensity_B",
    "ss_propensity_C",
    "ss_propensity_E",
    "ss_propensity_G",
    "ss_propensity_H",
    "ss_propensity_I",
    "ss_propensity_S",
    "ss_propensity_T",
]

PAIRWISE_FEATURES: list[str] = [
    "dccm",
    "contact_map",
    "distance_fluctuations",
]

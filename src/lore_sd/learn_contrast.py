#!/usr/bin/env python3

import lore_sd  # noqa: F401  caps BLAS threads; must be imported before numpy

import argparse
import pickle
import numpy as np

from sklearn.linear_model import Ridge

from lore_sd.mrtrix_io.io.image import load_mrtrix
from lore_sd.utils.SphericalHarmonics import zhgaussian


def build_rf_dictionary(
    A,
    B,
    bvals=(0, 500, 1000, 2000, 5000),
    da_max=3.3e-3,
    dr_max=3.3e-3,
    lmax=8,
):

    Da = np.linspace(0.0, da_max, A)
    Dr = np.linspace(0.0, dr_max, B)

    dictionary = []

    for i in range(A):

        row = []

        for j in range(B):

            rf = zhgaussian(
                np.asarray(bvals, dtype=float),
                Da[i],
                Dr[j],
                lmax=lmax,
            )

            row.append(rf.reshape(-1))

        dictionary.append(row)

    D = np.asarray(
        dictionary,
        dtype=np.float32,
    )

    return D.reshape(
        A * B,
        -1,
    )


def fractions_to_rf(
    fractions,
    bvals=(0, 500, 1000, 2000, 5000),
    lmax=8,
):

    A = fractions.shape[-2]
    B = fractions.shape[-1]

    D = build_rf_dictionary(
        A=A,
        B=B,
        bvals=bvals,
        lmax=lmax,
    )

    F = fractions.reshape(
        -1,
        A * B,
    )

    RF = F @ D

    rf_dim = D.shape[1]

    RF = RF.reshape(
        *fractions.shape[:3],
        rf_dim,
    )

    return RF


def compute_fod_features(odf):
    """
    Returns

        p2
        p4
        p6
        p8

    using RMS SH amplitudes.
    """

    ncoeff = odf.shape[-1]

    expected = 1 + 5 + 9 + 13 + 17

    if ncoeff < expected:
        raise ValueError(
            f"Expected at least {expected} SH coefficients, "
            f"got {ncoeff}"
        )

    i0 = 1
    i2 = i0 + 5
    i4 = i2 + 9
    i6 = i4 + 13
    i8 = i6 + 17

    p2 = np.linalg.norm(
        odf[..., i0:i2],
        axis=-1,
    )

    p4 = np.linalg.norm(
        odf[..., i2:i4],
        axis=-1,
    )

    p6 = np.linalg.norm(
        odf[..., i4:i6],
        axis=-1,
    )

    p8 = np.linalg.norm(
        odf[..., i6:i8],
        axis=-1,
    )

    return p2, p4, p6, p8


def build_feature_matrix(
    fractions,
    odf,
    valid,
):

    RF = fractions_to_rf(
        fractions,
    )

    rf_features = RF[valid].reshape(
        valid.sum(),
        -1,
    )

    p2, p4, p6, p8 = compute_fod_features(
        odf
    )

    eps = 1e-8

    extras = np.column_stack(
        [
            p2[valid],
            p4[valid] / (p2[valid] + eps),
            p6[valid] / (p2[valid] + eps),
            p8[valid] / (p2[valid] + eps),
        ]
    )

    F = np.concatenate(
        [
            rf_features,
            extras,
        ],
        axis=1,
    )

    return F


def build_design_matrix(
    F,
    model="bilinear",
):

    N, P = F.shape

    if model == "linear":

        X = np.empty(
            (N, 1 + P),
            dtype=np.float64,
        )

        X[:, 0] = 1.0
        X[:, 1:] = F

        return X

    if model == "bilinear":

        iu = np.triu_indices(
            P,
            k=1,
        )

        interactions = (
            F[:, iu[0]]
            * F[:, iu[1]]
        )

        X = np.empty(
            (
                N,
                1 + P + interactions.shape[1]
            ),
            dtype=np.float64,
        )

        X[:, 0] = 1.0
        X[:, 1:1 + P] = F
        X[:, 1 + P:] = interactions

        return X

    raise ValueError(model)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fractions",
        required=True,
    )

    parser.add_argument(
        "--odf",
        required=True,
    )

    parser.add_argument(
        "--target",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    parser.add_argument(
        "--mask",
        default=None,
    )

    parser.add_argument(
        "--weights",
        default=None,
    )

    parser.add_argument(
        "--model",
        choices=[
            "linear",
            "bilinear",
        ],
        default="bilinear",
    )

    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--max-voxels",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=13,
    )

    args = parser.parse_args()

    frac_img = load_mrtrix(
        args.fractions
    )

    odf_img = load_mrtrix(
        args.odf
    )

    target_img = load_mrtrix(
        args.target
    )

    fractions = np.nan_to_num(
        frac_img.data.astype(np.float32)
    )

    odf = np.nan_to_num(
        odf_img.data.astype(np.float32)
    )

    target = target_img.data

    spatial = target.shape

    mask = np.ones(
        spatial,
        dtype=bool,
    )

    if args.mask:

        mask &= (
            load_mrtrix(args.mask).data
            > 0.5
        )

    valid = (
        mask
        & np.isfinite(target)
    )

    F = build_feature_matrix(
        fractions,
        odf,
        valid,
    )

    y = target[valid].reshape(-1)

    keep = (
        np.all(
            np.isfinite(F),
            axis=1,
        )
        & np.isfinite(y)
    )

    F = F[keep]
    y = y[keep]

    voxel_weights = np.ones(
        len(y),
        dtype=np.float64,
    )

    if args.weights:

        w = load_mrtrix(
            args.weights
        ).data

        voxel_weights = (
            w[valid]
            .reshape(-1)
        )

        voxel_weights = voxel_weights[
            keep
        ]

    if (
        args.max_voxels > 0
        and
        F.shape[0] > args.max_voxels
    ):

        rng = np.random.default_rng(
            args.seed
        )

        idx = rng.choice(
            F.shape[0],
            size=args.max_voxels,
            replace=False,
        )

        F = F[idx]
        y = y[idx]
        voxel_weights = voxel_weights[idx]

    print(
        "Training voxels:",
        F.shape[0]
    )

    print(
        "RF features:",
        F.shape[1] - 4
    )

    print(
        "Total features:",
        F.shape[1]
    )

    X = build_design_matrix(
        F,
        model=args.model,
    )

    print(
        "Design matrix:",
        X.shape
    )

    reg = Ridge(
        alpha=args.ridge,
        fit_intercept=False,
    )

    reg.fit(
        X,
        y,
        sample_weight=voxel_weights,
    )

    pred = reg.predict(X)

    mse = np.average(
        (pred - y) ** 2,
        weights=voxel_weights,
    )

    print(
        "Weighted training MSE:",
        mse,
    )

    with open(
        args.out,
        "wb",
    ) as fp:

        pickle.dump(
            {
                "regressor": reg,
                "model": args.model,
                "rf_bvals":
                    [0, 500, 1000, 2000, 5000],
                "lmax": 8,
            },
            fp,
        )

    print(
        "Saved:",
        args.out,
    )


if __name__ == "__main__":
    main()
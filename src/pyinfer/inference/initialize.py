from types import SimpleNamespace

import numpy as np

from ..nuisance_model.background import Step
from ..nuisance_model.sig_plus_bg import (
    GaussEMGLeftPolyStep,
    GaussEMGLeftPolynomial,
    GaussPolyStep,
    GaussPolynomial,
)


MODELS = {
    "GaussEMGLeftPolyStep": GaussEMGLeftPolyStep,
    "GaussEMGLeftPolynomial": GaussEMGLeftPolynomial,
    "GaussPolyStep": GaussPolyStep,
    "GaussPolynomial": GaussPolynomial,
}

TAIL_MODELS = {
    "GaussEMGLeftPolyStep",
    "GaussEMGLeftPolynomial",
}

STEP_MODELS = {
    "GaussEMGLeftPolyStep",
    "GaussPolyStep",
}


def _poly_basis(edges, x_ref, x_scale, degree):
    z_lo = (edges[:-1] - x_ref) / x_scale
    z_hi = (edges[1:] - x_ref) / x_scale

    return np.column_stack([
        x_scale
        * (z_hi**(i + 1) - z_lo**(i + 1))
        / (i + 1)
        for i in range(degree + 1)
    ])


def _safe_poly(raw, floor):
    degree = len(raw) - 1

    if not np.all(np.isfinite(raw)) or raw[0] <= floor:
        return float(floor), np.zeros(degree)

    b0 = float(raw[0])
    hs = np.clip(
        np.asarray(raw[1:], dtype=float) / b0,
        -0.8,
        0.8,
    )

    if len(hs):
        z = np.linspace(-1.0, 1.0, 201)
        shape = np.polynomial.polynomial.polyval(
            z,
            (1.0, *hs),
        )

        minimum = float(np.min(shape))

        if minimum < 0.2:
            hs *= np.clip(
                0.8 / (1.0 - minimum),
                0.0,
                1.0,
            )

    return b0, hs


def _smooth(values):
    if len(values) < 3:
        return values.copy()

    return np.convolve(
        values,
        (0.25, 0.5, 0.25),
        mode="same",
    )


def _estimate_sig(
    x,
    widths,
    excess,
    mu,
    span,
    min_width,
):
    dx = x - mu

    # Use the right side to reduce sensitivity to the left EMG tail.
    mask = (
        (dx >= 0.0)
        & (dx < 0.2 * span)
    )

    weights = excess[mask] * widths[mask]

    if np.sum(weights) > 0:
        sig = np.sqrt(
            np.sum(
                weights * dx[mask]**2
            )
            / np.sum(weights)
        )

    else:
        sig = 2.0 * min_width

    return float(
        np.clip(
            sig,
            min_width / 2.0,
            span / 5.0,
        )
    )


def _fit_background(
    edges,
    y,
    x,
    poly_basis,
    mu,
    sig,
    degree,
    with_step,
    floor,
):
    span = edges[-1] - edges[0]

    exclusion = max(
        4.0 * sig,
        0.05 * span,
    )

    mask = np.abs(x - mu) > exclusion

    if with_step:
        step_basis = Step().integral(
            edges[:-1],
            edges[1:],
            1.0,
            mu,
            sig,
        )

        design = np.column_stack(
            (poly_basis, step_basis)
        )

    else:
        step_basis = None
        design = poly_basis

    if np.count_nonzero(mask) < design.shape[1]:
        mask = np.ones(len(x), dtype=bool)

    raw = np.linalg.lstsq(
        design[mask],
        y[mask],
        rcond=None,
    )[0]

    b0, hs = _safe_poly(
        raw[:degree + 1],
        floor,
    )

    coeffs = np.concatenate(
        ([b0], b0 * hs)
    )

    poly_counts = poly_basis @ coeffs
    A_step = 0.0

    if with_step:
        denominator = np.dot(
            step_basis[mask],
            step_basis[mask],
        )

        if denominator > 0:
            A_step = np.dot(
                step_basis[mask],
                y[mask] - poly_counts[mask],
            ) / denominator

            A_step = max(
                0.0,
                float(A_step),
            )

    background = poly_counts.copy()

    if with_step:
        background += A_step * step_basis

    return b0, hs, A_step, background


def _model_args(
    model_name,
    A,
    mu,
    sig,
    f_tail,
    tau,
    A_step,
    b0,
    hs,
):
    background = (b0, *hs)

    if model_name == "GaussEMGLeftPolyStep":
        return (
            A,
            mu,
            sig,
            f_tail,
            tau,
            A_step,
            *background,
        )

    if model_name == "GaussEMGLeftPolynomial":
        return (
            A,
            mu,
            sig,
            f_tail,
            tau,
            *background,
        )

    if model_name == "GaussPolyStep":
        return (
            A,
            mu,
            sig,
            A_step,
            *background,
        )

    return (
        A,
        mu,
        sig,
        *background,
    )


def _score(model, edges, y, args):
    expected = model.integral(
        edges[:-1],
        edges[1:],
        *args,
    )

    if (
        np.any(~np.isfinite(expected))
        or np.any(expected <= 0)
    ):
        return np.inf

    return float(
        np.sum(
            expected
            - y * np.log(expected)
        )
    )


def get_initial_params(
    model_name,
    bin_edges,
    bin_contents,
    degree=1,
):
    edges = np.asarray(
        bin_edges,
        dtype=float,
    )

    y = np.asarray(
        bin_contents,
        dtype=float,
    )

    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    if edges.ndim != 1 or y.ndim != 1:
        raise ValueError(
            "bin_edges and bin_contents must be one-dimensional"
        )

    if not isinstance(degree, int) or degree < 0:
        raise ValueError(
            "degree must be a non-negative integer"
        )

    if len(edges) != len(y) + 1:
        raise ValueError(
            "bin_edges must have length len(bin_contents) + 1"
        )

    if len(y) < 3:
        raise ValueError(
            "At least three bins are required"
        )

    if (
        np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(
            "bin_edges must be finite and strictly increasing"
        )

    if (
        np.any(~np.isfinite(y))
        or np.any(y < 0)
    ):
        raise ValueError(
            "bin_contents must be finite and non-negative"
        )

    if np.any(y != np.floor(y)):
        raise ValueError(
            "bin_contents must contain integer counts"
        )

    if degree >= len(y):
        raise ValueError(
            "degree must be smaller than the number of bins"
        )

    x_lo = edges[:-1]
    x_hi = edges[1:]
    x = 0.5 * (x_lo + x_hi)
    widths = x_hi - x_lo

    xmin = edges[0]
    xmax = edges[-1]
    span = xmax - xmin
    min_width = np.min(widths)

    x_ref = 0.5 * (xmin + xmax)
    x_scale = 0.5 * span

    with_tail = model_name in TAIL_MODELS
    with_step = model_name in STEP_MODELS

    poly_basis = _poly_basis(
        edges,
        x_ref,
        x_scale,
        degree,
    )

    n_edge = min(
        len(x) // 2,
        max(
            degree + 2,
            int(0.2 * len(x)),
        ),
    )

    edge_mask = np.zeros(
        len(x),
        dtype=bool,
    )

    edge_mask[:n_edge] = True
    edge_mask[-n_edge:] = True

    y_density = y / widths

    positive_edge = y_density[
        edge_mask
    ][
        y_density[edge_mask] > 0
    ]

    scale = (
        np.median(positive_edge)
        if len(positive_edge)
        else max(np.mean(y_density), 1.0)
    )

    floor = max(
        1e-9,
        1e-3 * scale,
    )

    raw_poly = np.linalg.lstsq(
        poly_basis[edge_mask],
        y[edge_mask],
        rcond=None,
    )[0]

    b0, hs = _safe_poly(
        raw_poly,
        floor,
    )

    background = poly_basis @ np.concatenate(
        ([b0], b0 * hs)
    )

    excess = np.clip(
        (y - background) / widths,
        0.0,
        None,
    )

    mu = float(
        x[np.argmax(_smooth(excess))]
    )

    sig = _estimate_sig(
        x,
        widths,
        excess,
        mu,
        span,
        min_width,
    )

    A_step = 0.0

    # Two deterministic passes are enough to stabilize
    # background, peak position and width.
    for _ in range(2):
        b0, hs, A_step, background = _fit_background(
            edges,
            y,
            x,
            poly_basis,
            mu,
            sig,
            degree,
            with_step,
            floor,
        )

        excess = np.clip(
            (y - background) / widths,
            0.0,
            None,
        )

        mu = float(
            x[np.argmax(_smooth(excess))]
        )

        sig = _estimate_sig(
            x,
            widths,
            excess,
            mu,
            span,
            min_width,
        )

    A = max(
        float(
            np.sum(
                np.clip(
                    y - background,
                    0.0,
                    None,
                )
            )
        ),
        1.0,
    )

    if with_tail:
        tail_candidates = [
            (0.0, sig),
        ]

        for f_tail in (0.10, 0.25, 0.50):
            for scale_tau in (1.0, 2.0, 4.0):
                tau = float(
                    np.clip(
                        scale_tau * sig,
                        min_width / 2.0,
                        span / 2.0,
                    )
                )

                tail_candidates.append(
                    (f_tail, tau)
                )

    else:
        tail_candidates = [
            (None, None)
        ]

    model = MODELS[model_name](
        x_ref=x_ref,
        x_scale=x_scale,
    )

    best = None
    n_eval = 0

    for f_tail, tau in tail_candidates:
        args = _model_args(
            model_name,
            A,
            mu,
            sig,
            f_tail,
            tau,
            A_step,
            b0,
            hs,
        )

        score = _score(
            model,
            edges,
            y,
            args,
        )

        n_eval += 1

        if (
            best is None
            or score < best[0]
        ):
            best = (
                score,
                f_tail,
                tau,
            )

    score, f_tail, tau = best

    bg_names = (
        "b0",
    ) + tuple(
        f"h{i}"
        for i in range(1, degree + 1)
    )

    bg_values = (
        b0,
        *hs,
    )

    bg_bounds = (
        (0, None),
    ) + (
        (-0.95, 0.95),
    ) * degree

    if model_name == "GaussEMGLeftPolyStep":
        names = (
            "A",
            "mu",
            "sig",
            "f_tail",
            "tau",
            "A_step",
            *bg_names,
        )

        values = (
            A,
            mu,
            sig,
            f_tail,
            tau,
            A_step,
            *bg_values,
        )

        bounds = (
            (0, None),
            (xmin, xmax),
            (min_width / 10.0, span),
            (0, 1),
            (min_width / 10.0, span),
            (0, None),
            *bg_bounds,
        )

    elif model_name == "GaussEMGLeftPolynomial":
        names = (
            "A",
            "mu",
            "sig",
            "f_tail",
            "tau",
            *bg_names,
        )

        values = (
            A,
            mu,
            sig,
            f_tail,
            tau,
            *bg_values,
        )

        bounds = (
            (0, None),
            (xmin, xmax),
            (min_width / 10.0, span),
            (0, 1),
            (min_width / 10.0, span),
            *bg_bounds,
        )

    elif model_name == "GaussPolyStep":
        names = (
            "A",
            "mu",
            "sig",
            "A_step",
            *bg_names,
        )

        values = (
            A,
            mu,
            sig,
            A_step,
            *bg_values,
        )

        bounds = (
            (0, None),
            (xmin, xmax),
            (min_width / 10.0, span),
            (0, None),
            *bg_bounds,
        )

    else:
        names = (
            "A",
            "mu",
            "sig",
            *bg_names,
        )

        values = (
            A,
            mu,
            sig,
            *bg_values,
        )

        bounds = (
            (0, None),
            (xmin, xmax),
            (min_width / 10.0, span),
            *bg_bounds,
        )

    params = dict(
        zip(
            names,
            values,
        )
    )

    limits = {
        "eps_S": (0, 1),
        "eps_B": (0, 1),
        **dict(
            zip(
                names,
                bounds,
            )
        ),
    }

    result = SimpleNamespace(
        success=np.isfinite(score),
        fun=float(score),
        x=np.asarray(
            values,
            dtype=float,
        ),
        nit=0,
        nfev=n_eval,
        message="deterministic coarse initialization",
    )

    return params, limits, result

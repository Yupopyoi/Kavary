# This software is based on original work provided under the MIT License:
# - Copyright (c) 2023 liujing04
# - Copyright (c) 2023 源文雨
# - Copyright (c) 2023 Ftps
#
# The current implementation includes modifications and extensions beyond the original code.
#
# Modified and maintained by:
#
# - Copyright (c) 2025 Yupopyoi
#
# See the LICENSE file for more details.

import torch
from torch.nn import functional as F


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    is_inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    """
    Unified interface to apply rational quadratic spline or its unconstrained variant.
    Automatically selects the appropriate spline function depending on tail behavior.
    """
    spline_fn = rational_quadratic_spline if tails is None else unconstrained_rational_quadratic_spline

    return spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        is_inverse=is_inverse,
        tail_bound=tail_bound,
        tails=tails,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    

def searchsorted(bin_locations, inputs, epsilon=1e-6):
    bin_locations[..., -1] += epsilon
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    is_inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    inside_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_mask = ~inside_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, (1, 1))
        fixed_endpoint_slope_logit = torch.log(torch.exp(torch.tensor(1.0 - min_derivative)) - 1.0)
        unnormalized_derivatives[...,  0] = fixed_endpoint_slope_logit
        unnormalized_derivatives[..., -1] = fixed_endpoint_slope_logit

        outputs[outside_mask] = inputs[outside_mask]
        logabsdet[outside_mask] = 0
    else:
        raise NotImplementedError(f"Tails mode '{tails}' is not implemented")

    outputs[inside_mask], logabsdet[inside_mask] = rational_quadratic_spline(
        inputs=inputs[inside_mask],
        unnormalized_widths=unnormalized_widths[inside_mask, :],
        unnormalized_heights=unnormalized_heights[inside_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_mask, :],
        is_inverse=is_inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet

def _compute_bin_positions(
    unnormalized: torch.Tensor,
    min_val: float,
    total_width: float,
    num_bins: int,
    offset: float = 0.0,
):
    softmaxed = F.softmax(unnormalized, dim=-1)
    bin_sizes = min_val + (1.0 - min_val * num_bins) * softmaxed
    cum_bins = torch.cumsum(bin_sizes, dim=-1)
    cum_bins = F.pad(cum_bins, (1, 0), mode="constant", value=0.0)
    cum_bins = cum_bins * total_width + offset
    cum_bins[..., 0] = offset
    cum_bins[..., -1] = offset + total_width
    return cum_bins, bin_sizes


def _compute_theta(inputs, bin_edges):
    bin_idx = torch.sum(inputs[..., None] >= bin_edges, dim=-1) - 1
    bin_idx = bin_idx.clamp(min=0)
    bin_widths = bin_edges[..., 1:] - bin_edges[..., :-1]
    left = bin_edges.gather(-1, bin_idx.unsqueeze(-1))[..., 0]
    width = bin_widths.gather(-1, bin_idx.unsqueeze(-1))[..., 0]
    theta = (inputs - left) / width.clamp(min=1e-6)
    return theta, bin_idx, left, width


def _forward_rqs(theta, delta, height, d_left, d_right, cum_heights):
    theta_product = theta * (1 - theta)
    numerator = height * (delta * theta**2 + d_left * theta_product)
    denominator = delta + (d_left + d_right - 2 * delta) * theta_product
    output = cum_heights + numerator / denominator

    deriv_num = delta**2 * (
        d_right * theta**2 + 2 * delta * theta_product + d_left * (1 - theta)**2
    )
    logabsdet = torch.log(deriv_num) - 2 * torch.log(denominator)
    return output, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    is_inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]
    if min_bin_width * num_bins > 1.0 or min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin size too large for number of bins")

    cumwidths, widths = _compute_bin_positions(
        unnormalized_widths, min_bin_width, right - left, num_bins, left
    )
    cumheights, heights = _compute_bin_positions(
        unnormalized_heights, min_bin_height, top - bottom, num_bins, bottom
    )
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)
    delta = heights / widths

    if is_inverse:
        # === Inverse Transform ===
        # Step 1: Determine bin index and local variables from cumulated heights
        theta, bin_idx, bottom_bin, bin_height = _compute_theta(inputs, cumheights)
        
        bin_idx_unsq = bin_idx.unsqueeze(-1)
        
        d_left = derivatives.gather(-1, bin_idx_unsq).squeeze(-1)
        d_right = derivatives[..., 1:].gather(-1, bin_idx_unsq).squeeze(-1)
        delta_bin = delta.gather(-1, bin_idx_unsq).squeeze(-1)

        # Step 2: Solve quadratic equation for theta using inverse formula
        a = (inputs - bottom_bin) * (d_left + d_right - 2 * delta_bin)
        b = bin_height * (delta_bin - d_left)
        c = -delta_bin * (inputs - bottom_bin)
        discriminant = b**2 - 4 * a * c
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        root = root.clamp(0.0, 1.0)

        # Step 3: Reconstruct input from normalized root
        theta = root
        bin_left = cumwidths.gather(-1, bin_idx_unsq).squeeze(-1)
        bin_width = widths.gather(-1, bin_idx_unsq).squeeze(-1)
        outputs = bin_left + theta * bin_width

        # Step 4: Compute log|det(Jacobian)| analytically
        theta_product = theta * (1 - theta)
        denominator = delta_bin + (d_left + d_right - 2 * delta_bin) * theta_product
        numerator = delta_bin**2 * (
            d_right * theta**2 + 2 * delta_bin * theta_product + d_left * (1 - theta)**2
        )
        logabsdet = torch.log(numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        # === Forward Transform ===
        # Step 1: Determine bin index and local variables from cumulated widths
        theta, bin_idx, bin_left, bin_width = _compute_theta(inputs, cumwidths)
        
        bin_idx_unsq = bin_idx.unsqueeze(-1)
        
        cum_height = cumheights.gather(-1, bin_idx_unsq).squeeze(-1)
        height = heights.gather(-1, bin_idx_unsq).squeeze(-1)
        delta_bin = delta.gather(-1, bin_idx_unsq).squeeze(-1)
        d_left = derivatives.gather(-1, bin_idx_unsq).squeeze(-1)
        d_right = derivatives[..., 1:].gather(-1, bin_idx_unsq).squeeze(-1)
        
        # Step 2: Apply forward spline transform
        return _forward_rqs(theta, delta_bin, height, d_left, d_right, cum_height)
    
    
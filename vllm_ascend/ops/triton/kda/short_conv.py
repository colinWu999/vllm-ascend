# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Width-four GLM short convolution over admitted verification rows."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _verify_short_conv(
    x,
    weight,
    state,
    starts,
    slots,
    accepted,
    out,
    channels: tl.constexpr,
    x_row: tl.constexpr,
    x_channel: tl.constexpr,
    w_row: tl.constexpr,
    w_channel: tl.constexpr,
    state_slot: tl.constexpr,
    state_row: tl.constexpr,
    state_channel: tl.constexpr,
    start_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    accepted_stride: tl.constexpr,
    out_row: tl.constexpr,
    out_channel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(starts + req * start_stride).to(tl.int64)
    end = tl.load(starts + (req + 1) * start_stride).to(tl.int64)
    if start == end:
        return
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = c < channels
    dst = out + c * out_channel
    slot = tl.load(slots + req * slot_stride).to(tl.int64)
    if slot <= 0:
        for token in range(start, end):
            tl.store(dst + token * out_row, 0, mask)
        return

    page = state + slot * state_slot + c * state_channel
    history = page + (tl.load(accepted + req * accepted_stride).to(tl.int64) - 1) * state_row
    h0 = tl.load(history, mask, other=0).to(tl.float32)
    h1 = tl.load(history + state_row, mask, other=0).to(tl.float32)
    h2 = tl.load(history + 2 * state_row, mask, other=0).to(tl.float32)
    w = weight + c * w_channel
    w0 = tl.load(w, mask, other=0).to(tl.float32)
    w1 = tl.load(w + w_row, mask, other=0).to(tl.float32)
    w2 = tl.load(w + 2 * w_row, mask, other=0).to(tl.float32)
    w3 = tl.load(w + 3 * w_row, mask, other=0).to(tl.float32)

    # The next call reads three entries starting at its accepted count minus
    # one. Publish the two old entries and every new input in that order.
    tl.store(page, h1, mask)
    tl.store(page + state_row, h2, mask)
    for token in range(start, end):
        current = tl.load(x + token * x_row + c * x_channel, mask, other=0).to(tl.float32)
        total = h0 * w0 + h1 * w1 + h2 * w2 + current * w3
        tl.store(dst + token * out_row, total / (1.0 + tl.exp(-total)), mask)
        tl.store(page + (2 + token - start) * state_row, current, mask)
        h0, h1, h2 = h1, h2, current


def verify_short_conv(x, weight, state, starts, slots, accepted, max_query_len, out):
    """Read the previous accepted window and write one candidate history per prefix."""
    rows, channels = x.shape
    requests = slots.numel()
    if (
        weight.shape != (4, channels)
        or state.ndim != 3
        or state.shape[2] != channels
        or max_query_len < 1
        or state.shape[1] < max_query_len + 2
        or slots.ndim != 1
        or starts.shape != (requests + 1,)
        or accepted.shape != (requests,)
    ):
        raise ValueError("GLM verification needs four convolution taps and a full rollback window per request")
    if out is None:
        out = torch.zeros_like(x)
    elif out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError("GLM short convolution output must match its input shape, dtype and device")
    if requests and rows:
        block = 256
        _verify_short_conv[(requests, triton.cdiv(channels, block))](
            x,
            weight,
            state,
            starts,
            slots,
            accepted,
            out,
            channels,
            *x.stride(),
            *weight.stride(),
            *state.stride(),
            starts.stride(0),
            slots.stride(0),
            accepted.stride(0),
            *out.stride(),
            BLOCK=block,
        )
    return out

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""XPU-calibrated vLLM backend. Overrides the base TTFT/TPOT formulas:

  * TTFT: own-prefill (isl/ctx, not ceil(isl/ctx)) times an XPU queuing factor.
  * TPOT: per-request mix-step count with a roofline mixed-step attention cost.
"""

import logging
import math
from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk.backends.vllm_backend import VLLMBackend
from aiconfigurator_core.sdk.config import RuntimeConfig
from aiconfigurator_core.sdk.models import BaseModel
from aiconfigurator_core.sdk.perf_database import PerfDatabase
from aiconfigurator_core.sdk.step_estimate import MixedStepInput, StepEstimate

logger = logging.getLogger(__name__)

# TTFT queuing factor constants (see _xpu_ttft_queuing_factor).
TTFT_QUEUE_COEFF = 0.60
TTFT_QUEUE_EFF_EXP = 0.70
# Linear-in-b term: the log term alone flattens too early and under-projects high bs.
TTFT_QUEUE_LIN_SLOPE = 0.025

# Fraction of peak HBM bandwidth the fused mixed-step decode attention achieves
# (see _fused_decode_attn_ms).
MIXED_STEP_DECODE_FUSED_BW_FRACTION = 0.15

# gpt-oss decode MoE correction (see _gpt_oss_moe_eff_nt): the MoE table over-projects
# decode because power_law routing activates more experts than gpt-oss's router.
# E_supp, rho fingerprint the real router. Decode only.
GPT_OSS_ARCH = "GptOssForCausalLM"
GPT_OSS_MOE_ESUPP = 21.4
GPT_OSS_MOE_RHO = 0.572


class VLLMXPUBackend(VLLMBackend):
    """XPU-calibrated vLLM backend: TTFT + mixed-step TPOT."""

    # ====================== TTFT (XPU-calibrated) ======================

    def _compute_ttft(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        b: int,
        isl: int,
        osl: int,
        ctx_tokens: int,
        prefix: int,
        *,
        prefill_step_ms: float,
        genonly_step_latency_ms: float,
        encoder_latency_ms: float,
        steps_to_finish_ctx: float,
    ) -> float:
        """TTFT = encoder + (own_prefill + dispatch) * queuing_factor.

        own_prefill uses the fill fraction isl/ctx (prefill_step_ms is a full
        ctx-token forward pass), unlike the base backend's ceil(isl/ctx)."""
        mix_ms = prefill_step_ms
        dispatch_ms = self._prefill_dispatch_overhead_ms(model)
        own_prefill_ms = mix_ms * isl / ctx_tokens + dispatch_ms
        factor = self._xpu_ttft_queuing_factor(b, ctx_tokens, isl)
        return encoder_latency_ms + own_prefill_ms * factor

    @staticmethod
    def _xpu_ttft_queuing_factor(b: int, ctx_tokens: int, isl: int) -> float:
        """Serialized-prefill queue multiplier:
        1 + COEFF*log2(b)*eff^EFF_EXP + SLOPE*b, eff = r*b/(r+b), r = ctx/isl.
        Returns 1.0 at b <= 1."""
        if b <= 1:
            return 1.0
        r = ctx_tokens / isl
        eff = r * b / (r + b)
        return (1.0 + TTFT_QUEUE_COEFF * math.log2(b) * eff ** TTFT_QUEUE_EFF_EXP
                + TTFT_QUEUE_LIN_SLOPE * b)

    @staticmethod
    def _prefill_dispatch_overhead_ms(model: BaseModel) -> float:
        # Per-request dispatch overhead, not captured per-op:
        # 0.3/layer (vs the base backend's 0.8/layer).
        return model._num_layers * 0.3

    # ====================== TPOT (XPU-calibrated) ======================

    def _compute_tpot(self, *, b, isl, osl, ctx_tokens, num_mix_steps, num_genonly_steps,
                      num_mix_steps_for_tpot_calc, mix_step_latency_ms, genonly_step_latency_ms):
        """Per-request mix-step TPOT: a mix step is charged only to the requests
        decoding alongside its prefill, not fleet-averaged. max(1, ctx/isl) requests
        prefill per step, so a request sees (b - prefillers)/b of the mix steps.
        Falls back to the base step-weighted average when b<=1 / osl<=1."""
        if osl <= 1 or b <= 1:
            return super()._compute_tpot(
                b=b, isl=isl, osl=osl, ctx_tokens=ctx_tokens,
                num_mix_steps=num_mix_steps, num_genonly_steps=num_genonly_steps,
                num_mix_steps_for_tpot_calc=num_mix_steps_for_tpot_calc,
                mix_step_latency_ms=mix_step_latency_ms,
                genonly_step_latency_ms=genonly_step_latency_ms,
            )
        prefillers_per_step = max(1.0, ctx_tokens / isl)
        nmix_eff = num_mix_steps * max(0.0, b - prefillers_per_step) / b
        ngen_eff = osl - nmix_eff
        if nmix_eff + ngen_eff <= 0:
            return 0.0
        return (mix_step_latency_ms * nmix_eff + genonly_step_latency_ms * ngen_eff) / (nmix_eff + ngen_eff)

    # ==================== GEN-ONLY MoE (gpt-oss) =======================

    @staticmethod
    def _collector_active_experts(num_tokens, topk, num_experts, alpha):
        """Expected #active experts for the collector's power_law router, replaying
        _generate_power_law_distribution deterministically (quantiles, then
        clamp-at-num_tokens + round-robin redistribution)."""
        i = (np.arange(num_experts) + 0.5) / num_experts
        xmin, xmax = (0.01, 2.0) if num_tokens * topk <= num_experts else (1.0, num_tokens * 0.8)
        vals = ((xmax ** (1 - alpha) - xmin ** (1 - alpha)) * i + xmin ** (1 - alpha)) ** (1 / (1 - alpha))
        cap = math.ceil(num_tokens)
        cnt = np.minimum(np.round(vals / vals.sum() * (num_tokens * topk)), cap)
        target = round(num_tokens * topk)
        for _ in range(int(num_experts * cap) + 1):  # bounded: at most cap per expert
            deficit = int(target - cnt.sum())
            if deficit == 0:
                break
            if deficit > 0:
                j = int(np.argmin(cnt + (cnt >= cap) * 1e9))
                if cnt[j] >= cap:
                    break
                cnt[j] += 1
            else:
                j = int(np.argmax(cnt - (cnt <= 0) * 1e9))
                if cnt[j] <= 0:
                    break
                cnt[j] -= 1
        return float((cnt > 0).sum())

    def _gpt_oss_moe_eff_nt(self, gen_tokens, topk, num_experts, alpha):
        """num_tokens to re-query MoE at: where the collector's routing hits gpt-oss's real
        active-expert count. Fractional (query_moe interpolates); no-op if not below collector's."""
        real_active = GPT_OSS_MOE_ESUPP * (1.0 - (1.0 - topk / GPT_OSS_MOE_ESUPP) ** (gen_tokens ** GPT_OSS_MOE_RHO))
        if self._collector_active_experts(gen_tokens, topk, num_experts, alpha) <= real_active:
            return float(gen_tokens)
        try:
            return brentq(
                lambda n: self._collector_active_experts(n, topk, num_experts, alpha) - real_active,
                1.0, float(gen_tokens),
            )
        except ValueError:
            return float(gen_tokens)

    def _get_genonly_step_latency(self, model, database, runtime_config, gen_tokens, isl, osl):
        """gpt-oss decode: re-price generation_moe at the active-expert-matched num_tokens.

        Incremental (moe.query(eff) - moe.query(gen_tokens)) so it applies on top of
        super()'s baseline regardless of whether that came from the Python or Rust
        engine step, without folding the Python/Rust pricing gap into the correction."""
        lat, energy, per_ops, per_src = super()._get_genonly_step_latency(
            model, database, runtime_config, gen_tokens, isl, osl
        )
        if model.architecture != GPT_OSS_ARCH or gen_tokens <= 1 or "generation_moe" not in per_ops:
            return lat, energy, per_ops, per_src
        moe_op = next((o for o in model.generation_ops if o._name == "generation_moe"), None)
        if moe_op is None:
            return lat, energy, per_ops, per_src
        alpha = float(moe_op._workload_distribution.rsplit("_", 1)[1])  # "power_law_1.2" -> 1.2
        eff = self._gpt_oss_moe_eff_nt(gen_tokens, moe_op._topk, moe_op._num_experts, alpha)
        delta = float(moe_op.query(database, x=eff)) - float(moe_op.query(database, x=float(gen_tokens)))
        per_ops = {**per_ops, "generation_moe": per_ops["generation_moe"] + delta}
        return lat + delta, energy, per_ops, per_src

    # ==================== MIX-STEP LATENCY (shared) ====================

    def run_mixed(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        step: MixedStepInput,
    ) -> StepEstimate:
        # Swap the mixed step's isolated generation-attention cost for the roofline
        # estimate (_fused_decode_attn_ms); gen-only steps are left untouched.
        # run_agg calls run_mixed directly (not _get_mix_step_latency), so the
        # correction is applied here to reach every projection path.
        estimate = super().run_mixed(model, database, runtime_config, step)
        gen_tokens = step.num_decode_requests
        if gen_tokens <= 0:
            return estimate
        # run_mixed derives the image-augmented isl from the config's image
        # fields; mirror that so the roofline KV length matches the base step.
        isl = int(runtime_config.isl or 0) + self._visual_context_tokens(model, runtime_config)
        osl = int(runtime_config.osl or 0)
        attn = estimate.per_op_latency_ms.get("generation_attention", 0.0)
        fused = self._fused_decode_attn_ms(model, database, gen_tokens, isl, osl)
        delta = fused - attn
        per_ops = {**estimate.per_op_latency_ms, "generation_attention": fused}
        return replace(estimate, latency_ms=estimate.latency_ms + delta, per_op_latency_ms=per_ops)

    def _fused_decode_attn_ms(self, model, database, gen_tokens, isl, osl):
        """Memory-roofline cost (ms) of the mixed step's decode attention: KV bytes
        read across all layers / effective fused HBM bandwidth. Sliding-window
        layers cap KV at the window; iterate the model's own attention ops so each
        layer group uses its real KV length (full model: window=0 -> full context)."""
        kv_seq_len = isl + osl // 2  # avg KV length a decode token attends over
        kv_bytes_per_elem = getattr(getattr(model.config, "kvcache_quant_mode", None), "value", None)
        kv_bytes_per_elem = kv_bytes_per_elem.memory if kv_bytes_per_elem is not None else 2
        peak_bw = database.system_spec["gpu"]["mem_bw"]
        eff_bw = peak_bw * MIXED_STEP_DECODE_FUSED_BW_FRACTION

        kv_bytes = 0.0
        for op in (o for o in model.generation_ops if isinstance(o, ops.GenerationAttention)):
            eff_kv = min(kv_seq_len, op._window_size) if op._window_size > 0 else kv_seq_len
            kv_bytes += (gen_tokens * eff_kv * 2 * op._n_kv
                         * op._head_size * kv_bytes_per_elem * op._scale_factor)
        return kv_bytes / eff_bw * 1000.0

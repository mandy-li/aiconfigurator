# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# vLLM disagg queueing model env vars
# ---------------------------------------------------------------------------
# Set this env var to "true" or "1" to enable the queueing-aware
# TTFT correction model.  When unset or "false"/"0", the old fixed 1.8x
# factor is used.
_TTFT_QUEUEING_MODEL_ENV = "AICONFIG_TTFT_QUEUE_MODEL"

# Number of request iterations (waves) in the benchmark.  genai-perf default
# is N_total = 10 * C, so each concurrency level sees ~10 waves.  The
# first-wave burst is diluted over this many waves when computing the
# mean TTFT correction.  Default: 10.
_TTFT_NUM_REQUEST_ITERS_ENV = "AICONFIG_TTFT_NUM_REQUEST_ITERS"
_TTFT_NUM_REQUEST_ITERS_DEFAULT = 10


class VLLMBackend(BaseBackend):
    """vLLM backend.

    Currently mirrors TRT-LLM's activation-memory model (the pre-refactor
    implementation literally delegated ``_get_memory_usage`` to TRTLLMBackend),
    with no KV-cache-aware OOM accounting yet. We reuse both TRT-LLM's
    per-family coefficient table and its ``_moe_workspace_width`` hook so
    estimates stay byte-identical with the old delegation; the agg-pipeline
    hooks (``_resolve_agg_kwargs``, ``_oom_check_kwargs``, ...) remain at
    BaseBackend defaults — vLLM does not yet do KV-cache OOM probing.
    """

    # Reuse TRT-LLM's per-family activation coefficients until a vLLM-specific
    # tuning lands.
    ACTIVATION_COEFFICIENTS = TRTLLMBackend.ACTIVATION_COEFFICIENTS

    # Mirror TRT-LLM's MoE workspace accounting (raw h for DEEPSEEK family,
    # ``_hidden_size`` for GEMMA4MOE). Plain class-attribute alias to the
    # function object — Python binds it to the VLLMBackend instance at call
    # time; the function does not touch any TRTLLMBackend-specific state.
    _moe_workspace_width = TRTLLMBackend._moe_workspace_width

    def __init__(
        self,
    ):
        super().__init__()
        self._agg_cache = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
        self.name = common.BackendName.vllm

    # ============== Agg pipeline =======================================

    def run_agg(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Run the agg inference. TODO: add vLLM's own implementation
        """
        isl = runtime_config.isl
        osl = runtime_config.osl
        prefix = runtime_config.prefix
        b = runtime_config.batch_size
        ctx_seq_imbalance_correction_scale = runtime_config.seq_imbalance_correction_scale
        gen_seq_imbalance_correction_scale = runtime_config.gen_seq_imbalance_correction_scale
        ctx_tokens = kwargs.get("ctx_tokens")
        assert ctx_tokens is not None, "ctx_tokens is required"
        balance_score = isl * b / ctx_tokens / osl

        try:
            summary = self._agg_cache[isl][osl][b][ctx_tokens]
        except KeyError:
            # we would like to calculate num_mix_steps and num_genonly_steps based on
            # isl, osl, b, ctx_tokens within osl steps, need to finish all the ctx tokens
            #
            # Exact step count: as requests finish prefill they start consuming
            # 1 decode token per step, shrinking the effective prefill budget.
            # The simple formula ceil(b*isl/ctx) ignores this and under-counts
            # by 1 when the residual tokens spill into an extra step.
            def _count_prefill_steps(b_reqs, isl, ctx_tokens):
                remaining = [isl] * b_reqs
                decode_count = 0
                next_k = 0
                n_steps = 0
                for _ in range(isl * b_reqs + 1):
                    if next_k >= b_reqs and all(r <= 0 for r in remaining):
                        break
                    budget = max(1, ctx_tokens - decode_count)
                    pf = 0
                    k = next_k
                    while k < b_reqs and budget > 0:
                        if remaining[k] <= 0:
                            k += 1
                            continue
                        c = min(remaining[k], budget)
                        remaining[k] -= c
                        budget -= c
                        pf += c
                        if remaining[k] <= 0:
                            decode_count += 1
                            next_k = max(next_k, k + 1)
                        k += 1
                    while next_k < b_reqs and remaining[next_k] <= 0:
                        next_k += 1
                    if pf > 0:
                        n_steps += 1
                return n_steps

            steps_to_finish_ctx = _count_prefill_steps(b, isl, ctx_tokens)
            num_mix_steps = num_genonly_steps = 0
            num_mix_steps_for_tpot_calc = 0  # this is a correction for tpot calc only.
            if b > 1:
                if steps_to_finish_ctx >= osl:
                    num_mix_steps = steps_to_finish_ctx
                    num_mix_ctx_tokens = ctx_tokens
                    num_mix_gen_tokens = max(1, b // (steps_to_finish_ctx / osl))
                    num_genonly_steps = 0
                    num_genonly_tokens = 0
                    num_mix_steps_for_tpot_calc = num_mix_steps
                else:
                    # 3-step is an empirical correction for pipelining requests where new requests
                    # cannot be enqueued immediately after last request's exit
                    num_mix_steps = steps_to_finish_ctx
                    num_mix_ctx_tokens = ctx_tokens
                    # When ctx_tokens >= b*isl, all requests prefill in one step
                    # and there are 0 concurrent decode tokens.
                    num_mix_gen_tokens = max(0, b - int(np.ceil(ctx_tokens / isl)))
                    num_genonly_steps = osl - num_mix_steps
                    num_genonly_tokens = b
                    num_mix_steps_for_tpot_calc = max(1, num_mix_steps - 3)
            elif b == 1:
                # special case for b=1
                # When chunked prefill is enabled, ctx_tokens < isl means the
                # prefill is split across multiple steps.  Use the same helper
                # as the b>1 path so throughput is not inflated.
                num_mix_steps = _count_prefill_steps(1, isl, ctx_tokens)
                num_mix_ctx_tokens = ctx_tokens
                num_mix_gen_tokens = 0
                num_genonly_steps = max(0, osl - num_mix_steps)
                num_genonly_tokens = 1
                # For TPOT: when all steps are mix steps (prefill takes >= osl
                # steps), attribute them so TPOT reflects the mix-step latency;
                # otherwise keep 0 so TPOT = genonly latency (original behavior).
                if num_genonly_steps == 0 and num_mix_steps > 0:
                    num_mix_steps_for_tpot_calc = num_mix_steps
                else:
                    num_mix_steps_for_tpot_calc = 0

            # Per-ops latency collection (and parallel data-source breakdown).
            per_ops_data = {}
            per_ops_source = {}

            def _compute_chunked_ctx_attention(
                model: BaseModel,
                database: PerfDatabase,
                ctx_tokens: int,
                isl: int,
                prefix: int,
                ctx_seq_imbalance_correction_scale: float,
            ) -> tuple[float, float, str]:
                """
                Compute context attention latency for chunked prefill by
                averaging per-chunk attention across all chunks.

                When chunked prefill is active (ctx_tokens < isl), each step
                processes Q=ctx_tokens query tokens against a growing KV cache.
                Chunk c has KV=(c+1)*ctx_tokens tokens.  Computing each chunk
                individually and averaging captures the real per-step cost,
                whereas the old method (full ISL / num_chunks) under-estimates
                because smaller-Q kernels are less compute-efficient.

                When ctx_tokens >= isl (no chunking), falls back to the
                original single-shot computation.

                Returns:
                    tuple: (latency_ms, energy_wms, source_str)
                """
                if ctx_tokens < isl:
                    num_chunks = int(np.ceil(isl / ctx_tokens))
                    chunk_attn_ms = 0.0
                    chunk_attn_energy = 0.0
                    ctx_attn_source = "silicon"
                    for c in range(num_chunks):
                        chunk_prefix = prefix + c * ctx_tokens
                        chunk_isl = ctx_tokens + chunk_prefix
                        summary = self.run_static(
                            model,
                            database,
                            RuntimeConfig(
                                batch_size=1,
                                beam_width=1,
                                isl=chunk_isl,
                                osl=1,
                                prefix=chunk_prefix,
                                seq_imbalance_correction_scale=ctx_seq_imbalance_correction_scale,
                            ),
                            mode="static_ctx",
                        )
                        ld = summary.get_context_latency_dict()
                        ed = summary.get_context_energy_wms_dict()
                        chunk_attn_ms += ld["context_attention"]
                        chunk_attn_energy += ed.get("context_attention", 0.0)
                        if c == 0:
                            ctx_attn_source = summary.get_context_source_dict().get(
                                "context_attention", "silicon"
                            )
                    return chunk_attn_ms / num_chunks, chunk_attn_energy / num_chunks, ctx_attn_source
                else:
                    batch_size = np.ceil(ctx_tokens / isl)
                    summary = self.run_static(
                        model,
                        database,
                        RuntimeConfig(
                            batch_size=batch_size,
                            beam_width=1,
                            isl=isl,
                            osl=1,
                            prefix=prefix,
                            seq_imbalance_correction_scale=ctx_seq_imbalance_correction_scale,
                        ),
                        mode="static_ctx",
                    )
                    ld = summary.get_context_latency_dict()
                    ed = summary.get_context_energy_wms_dict()
                    src = summary.get_context_source_dict().get("context_attention", "silicon")
                    scale_factor = isl / ctx_tokens if ctx_tokens <= isl else 1.0
                    return (
                        ld["context_attention"] / scale_factor,
                        ed.get("context_attention", 0.0) / scale_factor,
                        src,
                    )

            # FIXME, fix for DS. DS has different ops for attn in ctx and gen.
            def _get_mix_step_latency(
                model: BaseModel,
                database: PerfDatabase,
                ctx_tokens: int,
                gen_tokens: int,
                isl: int,
                osl: int,
                prefix: int,
            ) -> tuple[float, float, float]:
                """
                Get mixed step latency and energy.

                Returns:
                    tuple: (latency in ms, energy in watt-milliseconds,
                           generation_attention latency in ms)
                """
                # Decode tokens share the ctx_tokens budget (not additive),
                # so total tokens processed per step = ctx_tokens.
                num_tokens = ctx_tokens
                # treat this as a combined single batch inference, extract non-attention latency
                summary = self.run_static(
                    model,
                    database,
                    # num tokens for gemm needs to be adjusted for prefix, depends on the avg prefix len per request
                    RuntimeConfig(
                        batch_size=1,
                        beam_width=1,
                        isl=num_tokens,
                        osl=1,
                        prefix=prefix * np.floor(ctx_tokens / isl),
                        seq_imbalance_correction_scale=ctx_seq_imbalance_correction_scale,
                    ),
                    mode="static_ctx",
                )
                latency_dict = summary.get_context_latency_dict()
                energy_wms_dict = summary.get_context_energy_wms_dict()
                source_dict = summary.get_context_source_dict()
                non_attention_latency_ms = 0.0
                non_attention_energy_wms = 0.0
                mix_non_attn_ops = {}
                mix_non_attn_sources = {}
                for layer_name, latency in latency_dict.items():
                    if layer_name != "context_attention":
                        non_attention_latency_ms += latency
                        non_attention_energy_wms += energy_wms_dict.get(layer_name, 0.0)
                        mix_non_attn_ops[layer_name] = latency
                        mix_non_attn_sources[layer_name] = source_dict.get(layer_name, "silicon")

                # second pass to get ctx attn + third pass to get gen attn
                _sm_version = database.system_spec["gpu"].get("sm_version", -1)
                _is_unified_attn = _sm_version == -1 and gen_tokens > 0

                # Pass 2: context attention - use per-chunk computation
                ctx_attention_latency_ms, ctx_attention_energy_wms, ctx_attn_source = (
                    _compute_chunked_ctx_attention(
                        model, database, ctx_tokens, isl, prefix,
                        ctx_seq_imbalance_correction_scale,
                    )
                )

                # --- OLD Pass 2 (single-shot, divides full ISL by num_chunks) ---
                # num_tokens = isl
                # batch_size = np.ceil(ctx_tokens / isl)
                # summary = self.run_static(
                #     model,
                #     database,
                #     RuntimeConfig(
                #         batch_size=batch_size,
                #         beam_width=1,
                #         isl=num_tokens,
                #         osl=1,
                #         prefix=prefix,
                #         seq_imbalance_correction_scale=ctx_seq_imbalance_correction_scale,
                #     ),
                #     mode="static_ctx",
                # )
                # latency_dict = summary.get_context_latency_dict()
                # energy_wms_dict = summary.get_context_energy_wms_dict()
                # ctx_attn_source = summary.get_context_source_dict().get("context_attention", "silicon")
                # scale_factor = isl / ctx_tokens if ctx_tokens <= isl else 1.0
                # ctx_attention_latency_ms = latency_dict["context_attention"] / scale_factor
                # ctx_attention_energy_wms = energy_wms_dict.get("context_attention", 0.0) / scale_factor
                # --- END OLD Pass 2 ---

                # Pass 3: generation attention
                gen_attention_latency_ms = 0.0
                gen_attention_energy_wms = 0.0
                gen_attn_source = "silicon"
                if gen_tokens > 0:
                    if _is_unified_attn:
                        # On XPU (no sm_version), decode and prefill tokens share a single
                        # flash_attn_varlen_fwd kernel call. The kernel uses a global
                        # max_seqlen_q from the entire batch, so decode tokens (q_len=1)
                        # run through a code path sized for the prefill's q_len (e.g. 512).
                        # Benchmarking shows this inflates per-decode-token attention cost
                        # by ~4.5× vs decode-only batches (where max_seqlen_q=1).
                        # The silicon generation_attention data is collected with decode-only
                        # batches, so we apply an inflation factor here.
                        # TODO: replace with mixed-batch silicon data collection.
                        _UNIFIED_ATTN_DECODE_INFLATION = 4.5

                    num_tokens = gen_tokens
                    summary = self.run_static(
                        model,
                        database,
                        RuntimeConfig(
                            batch_size=num_tokens,
                            beam_width=1,
                            isl=isl + osl // 2,
                            osl=2,
                            gen_seq_imbalance_correction_scale=gen_seq_imbalance_correction_scale,
                        ),
                        mode="static_gen",
                    )
                    latency_dict = summary.get_generation_latency_dict()
                    energy_wms_dict = summary.get_generation_energy_wms_dict()
                    gen_attention_latency_ms = latency_dict["generation_attention"]
                    gen_attention_energy_wms = energy_wms_dict.get("generation_attention", 0.0)
                    gen_attn_source = summary.get_generation_source_dict().get("generation_attention", "silicon")

                    if _is_unified_attn:
                        gen_attention_latency_ms *= _UNIFIED_ATTN_DECODE_INFLATION
                        gen_attention_energy_wms *= _UNIFIED_ATTN_DECODE_INFLATION

                # Collect per-op breakdown for mix step
                per_ops_data["mix_step"] = {
                    **mix_non_attn_ops,
                    "context_attention (scaled)": ctx_attention_latency_ms,
                    "generation_attention": gen_attention_latency_ms,
                }
                per_ops_source["mix_step"] = {
                    **mix_non_attn_sources,
                    "context_attention (scaled)": ctx_attn_source,
                    "generation_attention": gen_attn_source,
                }

                # Combine all components (simple addition)
                total_latency_ms = non_attention_latency_ms + ctx_attention_latency_ms + gen_attention_latency_ms
                total_energy_wms = non_attention_energy_wms + ctx_attention_energy_wms + gen_attention_energy_wms

                return total_latency_ms, total_energy_wms, gen_attention_latency_ms

            def _get_genonly_step_latency(
                model: BaseModel, database: PerfDatabase, gen_tokens: int, isl: int, osl: int
            ) -> tuple[float, float]:
                """
                Get generation-only step latency and energy.

                Returns:
                    tuple: (latency in ms, energy in watt-milliseconds)
                """
                if gen_tokens <= 0:
                    return 0.0, 0.0
                num_tokens = gen_tokens
                summary = self.run_static(
                    model,
                    database,
                    RuntimeConfig(
                        batch_size=num_tokens,
                        beam_width=1,
                        isl=isl + osl // 2,
                        osl=2,
                        gen_seq_imbalance_correction_scale=gen_seq_imbalance_correction_scale,
                    ),
                    mode="static_gen",
                )
                latency_dict = summary.get_generation_latency_dict()
                energy_wms_dict = summary.get_generation_energy_wms_dict()
                source_dict = summary.get_generation_source_dict()
                genonly_step_latency_ms = 0.0
                genonly_step_energy_wms = 0.0
                genonly_ops = {}
                genonly_sources = {}
                for layer_name, latency in latency_dict.items():
                    genonly_step_latency_ms += latency
                    genonly_step_energy_wms += energy_wms_dict.get(layer_name, 0.0)
                    genonly_ops[layer_name] = latency
                    genonly_sources[layer_name] = source_dict.get(layer_name, "silicon")

                per_ops_data["genonly_step"] = genonly_ops
                per_ops_source["genonly_step"] = genonly_sources

                return genonly_step_latency_ms, genonly_step_energy_wms

            # Call helpers (now return energy in wms instead of power)
            mix_step_latency_ms, mix_step_energy_wms, mix_gen_attn_ms = _get_mix_step_latency(
                model, database, num_mix_ctx_tokens, num_mix_gen_tokens, isl, osl, prefix
            )
            mix_step_base_ms = mix_step_latency_ms - mix_gen_attn_ms
            genonly_step_latency_ms, genonly_step_energy_wms = _get_genonly_step_latency(
                model, database, num_genonly_tokens, isl, osl
            )

            # Calculate timing - multi-wave FIFO simulation
            #
            # vLLM scheduler uses a single token_budget = max_num_batched_tokens.
            # RUNNING (decode) requests are scheduled first, each consuming 1 token.
            # WAITING (prefill) requests get the leftover budget in FIFO order.
            #
            # In a genai-perf benchmark with N request iterations, N*b total
            # requests are sent with concurrency=b.  Wave 1 has all b requests
            # arriving simultaneously; as each completes decode, a replacement
            # enters the prefill queue.  Subsequent waves inherit the exit
            # stagger from the previous wave, creating bursty arrivals that
            # queue behind the single-threaded prefill server.
            #
            # We simulate all N waves to capture this queuing faithfully.

            # Client-side tokenization rate governs wave1 arrival spread.
            # The benchmark client (vllm bench serve) tokenizes prompts on
            # the CPU before sending them to the server.  At ISL=8192 this
            # produces ~17 ms between successive request arrivals.  The rate
            # is CPU/workload-dependent and should be tuned per test system.
            # Set to 0 to disable (all requests arrive at t=0).
            _CLIENT_TOKENIZE_TOKS_PER_MS = 500.0  # ~500K tokens/sec

            _NUM_REQUEST_ITERS = 10  # genai-perf default: N_total = 10 * concurrency

            def _simulate_wave(b_reqs, isl, ctx_tokens, num_running_decode=0,
                               mix_step_ms=1.0, gen_step_ms=0.0,
                               mix_base_ms=None, gen_attn_ms=0.0,
                               max_gen_tokens=0, arrival_times=None):
                """Simulate FIFO prefill for b_reqs with staggered arrivals.

                Returns list of TTFT in ms for each request.  Step latency is
                interpolated between gen_step_ms (decode-only) and mix_step_ms
                (full-budget prefill) based on the fraction of ctx_tokens used
                for prefill in that step.

                When mix_base_ms is provided, mix step latency is decomposed as
                mix_base_ms + gen_attn_ms * (decode_count / max_gen_tokens) so
                that generation-attention cost ramps with the actual number of
                concurrent decode tokens instead of assuming steady-state.

                arrival_times: optional list of ms timestamps at which each
                    request becomes visible to the scheduler (default: all 0).
                """
                _scale_gen_attn = (mix_base_ms is not None and max_gen_tokens > 0
                                   and gen_attn_ms > 0)
                _arrivals = arrival_times if arrival_times is not None else [0.0] * b_reqs
                remaining = [isl] * b_reqs
                decode_count = num_running_decode
                next_to_prefill = 0
                ttft_ms = [0.0] * b_reqs
                cumulative_ms = 0.0

                for step in range(isl * b_reqs + 1):  # safety upper bound
                    if next_to_prefill >= b_reqs and all(r <= 0 for r in remaining):
                        break
                    budget = ctx_tokens - decode_count
                    if budget <= 0:
                        budget = 1  # at least 1 token

                    prefill_tokens = 0
                    finished_this_step = []

                    # Process prefilling requests in FIFO order
                    k = next_to_prefill
                    while k < b_reqs and budget > 0:
                        if remaining[k] <= 0:
                            k += 1
                            continue
                        if _arrivals[k] > cumulative_ms:
                            break  # FIFO: this and later requests not yet arrived
                        consume = min(remaining[k], budget)
                        remaining[k] -= consume
                        budget -= consume
                        prefill_tokens += consume
                        if remaining[k] <= 0:
                            finished_this_step.append(k)
                            decode_count += 1
                            next_to_prefill = max(next_to_prefill, k + 1)
                        k += 1

                    # Update next_to_prefill for any unseen requests
                    while next_to_prefill < b_reqs and remaining[next_to_prefill] <= 0:
                        next_to_prefill += 1

                    # Step latency: mix if prefilling, genonly if not.
                    # decode_count_before: decode tokens running at the start of
                    # this step (before any new finishers are counted).
                    decode_count_before = decode_count - len(finished_this_step)
                    if _scale_gen_attn:
                        _gen_scale = min(1.0, (decode_count_before - num_running_decode)
                                         / max_gen_tokens)
                        _effective_mix = mix_base_ms + gen_attn_ms * _gen_scale
                    else:
                        _effective_mix = mix_step_ms

                    if prefill_tokens * 2 >= ctx_tokens:
                        step_ms = _effective_mix
                    elif prefill_tokens == 0:
                        step_ms = gen_step_ms
                    else:
                        frac = prefill_tokens / ctx_tokens
                        step_ms = gen_step_ms + frac * (_effective_mix - gen_step_ms)

                    cumulative_ms += step_ms

                    for j in finished_this_step:
                        ttft_ms[j] = cumulative_ms - _arrivals[j]

                return ttft_ms

            if b <= 1:
                wave1_ms = _simulate_wave(
                    1, isl, ctx_tokens, num_running_decode=0,
                    mix_step_ms=mix_step_latency_ms, gen_step_ms=genonly_step_latency_ms,
                    mix_base_ms=mix_step_base_ms, gen_attn_ms=mix_gen_attn_ms,
                    max_gen_tokens=num_mix_gen_tokens,
                )
                ttft = wave1_ms[0]
            else:
                # Wave 1: requests trickle in as the client tokenizes them.
                if _CLIENT_TOKENIZE_TOKS_PER_MS > 0:
                    _wave1_arrivals = [i * isl / _CLIENT_TOKENIZE_TOKS_PER_MS
                                       for i in range(b)]
                else:
                    _wave1_arrivals = None
                wave1_ttfts = _simulate_wave(
                    b, isl, ctx_tokens, num_running_decode=0,
                    mix_step_ms=mix_step_latency_ms, gen_step_ms=genonly_step_latency_ms,
                    mix_base_ms=mix_step_base_ms, gen_attn_ms=mix_gen_attn_ms,
                    max_gen_tokens=num_mix_gen_tokens,
                    arrival_times=_wave1_arrivals,
                )

                # Multi-wave step-level simulation for waves 2..N.
                #
                # After wave1, decode exits trigger new prefill requests.
                # Step latency depends on whether a prefill is in progress
                # (mix_step) or not (genonly_step).  This creates a feedback
                # loop: mix steps slow down decode progression, spreading
                # out arrivals and reducing queue depth.  A static formula
                # cannot capture this; we simulate step-by-step.
                #
                # Client serialization: in practice, the benchmark client
                # processes completions one at a time (~17 ms each).  When
                # multiple decodes exit in the same step, their replacement
                # requests arrive staggered rather than simultaneously.
                # We model this by adding 1 extra delay step per additional
                # exit in the same step.

                # Get wave1 exit steps (reuse _count_prefill_steps logic)
                _w1_remaining = [isl] * b
                _w1_dc = 0
                _w1_nk = 0
                _w1_exit_steps = []
                _w1_step = 0
                for _ in range(isl * b + 1):
                    if _w1_nk >= b and all(r <= 0 for r in _w1_remaining):
                        break
                    _w1_budget = max(1, ctx_tokens - _w1_dc)
                    _w1_pf = 0
                    _w1_k = _w1_nk
                    _w1_finished = []
                    while _w1_k < b and _w1_budget > 0:
                        if _w1_remaining[_w1_k] <= 0:
                            _w1_k += 1
                            continue
                        _w1_c = min(_w1_remaining[_w1_k], _w1_budget)
                        _w1_remaining[_w1_k] -= _w1_c
                        _w1_budget -= _w1_c
                        _w1_pf += _w1_c
                        if _w1_remaining[_w1_k] <= 0:
                            _w1_finished.append(_w1_k)
                            _w1_dc += 1
                            _w1_nk = max(_w1_nk, _w1_k + 1)
                        _w1_k += 1
                    while _w1_nk < b and _w1_remaining[_w1_nk] <= 0:
                        _w1_nk += 1
                    if _w1_pf > 0:
                        _w1_step += 1
                        for _j in _w1_finished:
                            _w1_exit_steps.append(_w1_step)

                _ss_budget = max(1, ctx_tokens - (b - 1))
                _ss_service_steps = int(np.ceil(isl / _ss_budget))
                _ss_service_ms = _ss_service_steps * mix_step_latency_ms

                # Budget sharing: the last prefill step of request N uses
                # only (isl % budget) tokens, leaving the rest for request
                # N+1 in the SAME step.  This makes the inter-completion
                # gap (exit_stagger) shorter than the per-request service.
                _ss_remainder = isl % _ss_budget
                if _ss_remainder == 0:
                    _ss_exit_stagger = _ss_service_steps
                else:
                    _ss_leftover = _ss_budget - _ss_remainder
                    _ss_exit_stagger = int(np.ceil(
                        (isl - _ss_leftover) / _ss_budget
                    ))
                    # When ceil rounds up to service_steps, the actual
                    # inter-completion gap alternates between (svc-1) and
                    # svc steps due to varying leftover budget.  Using a
                    # fixed ceiling of svc causes the multi-wave simulation
                    # to model zero pipelining, leading to unbounded queue
                    # growth and wildly inflated TTFT.  Detect this case
                    # and alternate the stagger in the simulation loop.
                    _ss_alternating = _ss_exit_stagger >= _ss_service_steps

                # Step-level simulation from wave1 end through N-1 waves.
                # Delay = client tokenization + scheduler pipeline + HTTP.
                # Measured as step-counts (rounded up) for the discrete sim.
                _CLIENT_DELAY_MS = (
                    isl / max(1, _CLIENT_TOKENIZE_TOKS_PER_MS)
                    + 50.0
                )
                # When all steps are mix (steps_to_finish_ctx >= osl),
                # genonly_step_latency_ms is 0 because no gen-only tokens
                # are modeled.  Fall back to mix_step_latency_ms so the
                # delay calculation and simulation step timing remain valid.
                _sim_genonly_ms = (genonly_step_latency_ms
                                  if genonly_step_latency_ms > 0
                                  else mix_step_latency_ms)
                _SCHED_DELAY_STEPS = max(2, int(np.ceil(
                    _CLIENT_DELAY_MS / _sim_genonly_ms
                ))) + 2   # +2 for scheduler pipeline ticks

                _decode_remaining = [
                    es + (osl - 1) - steps_to_finish_ctx
                    for es in _w1_exit_steps
                ]
                _cur_time = steps_to_finish_ctx * mix_step_latency_ms
                _pf_rem = 0          # steps until prefill slot frees (exit_stagger)
                _pf_output_due = []  # [(output_step_count, arrival_time)] deferred outputs
                _wait_q = []         # arrival times of waiting requests
                _stagger_toggle = True  # for alternating stagger
                _pending = []        # [[countdown, arrival_time], ...]
                _active = list(_decode_remaining)
                _all_ttfts = list(wave1_ttfts)
                _total_needed = (_NUM_REQUEST_ITERS - 1) * b

                for _sim_step in range((_NUM_REQUEST_ITERS + 1) * (osl + isl) + 1):
                    if len(_all_ttfts) - len(wave1_ttfts) >= _total_needed:
                        break

                    # Step latency: mix if prefill running or output pending, else genonly
                    _is_mix = _pf_rem > 0 or len(_pf_output_due) > 0
                    if _is_mix:
                        _step_ms = mix_step_latency_ms
                        if _pf_rem > 0:
                            _pf_rem -= 1
                    else:
                        _step_ms = _sim_genonly_ms

                    _cur_time += _step_ms

                    # Check for deferred outputs completing this step
                    _new_outputs = []
                    for _od in _pf_output_due:
                        _od[0] -= 1
                        if _od[0] <= 0:
                            _ttft_i = _cur_time - _od[1]
                            _all_ttfts.append(_ttft_i)
                            _active.append(osl - 1)
                        else:
                            _new_outputs.append(_od)
                    _pf_output_due = _new_outputs

                    # Advance all active decodes; collect exits this step
                    _exits_this_step = 0
                    _new_active = []
                    for _d in _active:
                        _d -= 1
                        if _d <= 0:
                            _exits_this_step += 1
                        else:
                            _new_active.append(_d)
                    _active = _new_active

                    # Client serialization: stagger replacement arrivals
                    for _ei in range(_exits_this_step):
                        _pending.append([_SCHED_DELAY_STEPS + _ei, _cur_time])

                    # Tick pending requests; move ready ones to wait queue
                    _new_pending = []
                    for _p in _pending:
                        _p[0] -= 1
                        if _p[0] <= 0:
                            _wait_q.append(_p[1])
                        else:
                            _new_pending.append(_p)
                    _pending = _new_pending

                    # Start next prefill when slot frees and queue non-empty.
                    # Budget sharing lets the tail of request N overlap with
                    # the head of request N+1, giving an inter-completion gap
                    # of exit_stagger steps instead of service_steps.  But
                    # this pipeline benefit only applies when there IS a next
                    # request waiting - otherwise the slot simply sits idle
                    # after service_steps and no pipelining occurs.
                    if _pf_rem == 0 and _wait_q:
                        _arr = _wait_q.pop(0)
                        if _wait_q:
                            if _ss_alternating:
                                # Alternate between (svc-1) and svc to model
                                # the real budget-sharing inter-completion gap.
                                _pf_rem = (_ss_exit_stagger - 1) if _stagger_toggle else _ss_exit_stagger
                                _stagger_toggle = not _stagger_toggle
                            else:
                                _pf_rem = _ss_exit_stagger
                        else:
                            _pf_rem = _ss_service_steps
                        _pf_output_due.append([_ss_service_steps, _arr])

                ttft = float(np.mean(_all_ttfts))

                logger.debug(
                    f"Multi-wave TTFT: wave1_mean={np.mean(wave1_ttfts):.1f} ms, "
                    f"service={_ss_service_ms:.1f} ms ({_ss_service_steps} steps), "
                    f"sched_delay={_SCHED_DELAY_STEPS} steps, "
                    f"waves={_NUM_REQUEST_ITERS}"
                )

            mean_ttft_steps = ttft / mix_step_latency_ms  # effective steps for logging
            logger.debug(
                f"ttft: {ttft:.1f} ms, mean_steps: {mean_ttft_steps:.2f}, "
                f"b: {b}, ctx_tokens: {ctx_tokens}, isl: {isl}, "
                f"steps_to_finish_ctx: {steps_to_finish_ctx}"
            )

            # TPOT: weighted average of mix-step and genonly-step latencies,
            # weighted by the number of steps a request experiences in each phase.
            _num_genonly_for_tpot = num_genonly_steps

            tpot = (mix_step_latency_ms * num_mix_steps_for_tpot_calc + genonly_step_latency_ms * _num_genonly_for_tpot) / (
                num_mix_steps_for_tpot_calc + _num_genonly_for_tpot
            )
            output_throughput = (
                1000
                / (num_mix_steps * mix_step_latency_ms + num_genonly_steps * genonly_step_latency_ms)
                * b
                * (osl - 1)
            )
            logger.debug(
                f"ctx_tokens: {ctx_tokens}, b: {b}, osl: {osl}, isl: {isl}, "
                f"num_mix_steps: {num_mix_steps}, num_genonly_steps: {num_genonly_steps}, "
                f"num_mix_ctx_tokens: {num_mix_ctx_tokens}, "
                f"num_mix_gen_tokens: {num_mix_gen_tokens}, "
                f"num_genonly_tokens: {num_genonly_tokens}"
            )
            logger.debug(
                f"mix_step_latency: {mix_step_latency_ms} ms, genonly_step_latency: {genonly_step_latency_ms} ms"
            )
            logger.debug(
                f"mix_step_energy: {mix_step_energy_wms} W·ms, genonly_step_energy: {genonly_step_energy_wms} W·ms"
            )
            logger.debug(f"ttft: {ttft}, tpot: {tpot}, output_throughput: {output_throughput}")

            # Calculate weighted average power (SIMPLIFIED!)
            # Step 1: Calculate total energy (simple multiplication and addition)
            total_mix_energy_wms = num_mix_steps * mix_step_energy_wms
            total_genonly_energy_wms = num_genonly_steps * genonly_step_energy_wms
            total_energy_wms = total_mix_energy_wms + total_genonly_energy_wms

            # Step 2: Calculate total latency (simple multiplication and addition)
            total_latency_ms = num_mix_steps * mix_step_latency_ms + num_genonly_steps * genonly_step_latency_ms

            # Step 3: Derive average power (single division)
            if total_latency_ms > 0:
                agg_power_avg_w = total_energy_wms / total_latency_ms
            else:
                agg_power_avg_w = 0.0

            logger.debug(f"Aggregated power: {agg_power_avg_w}W (from {total_energy_wms} W·ms / {total_latency_ms} ms)")

            num_ctx_requests = np.ceil(ctx_tokens / isl)
            num_gen_requests = b - num_ctx_requests
            if b == 1:
                num_ctx_requests = 1
                num_gen_requests = 1

            # correct output_throughput and concurrency for attention dp (global batch)
            scale_factor = model.config.pp_size * model.config.attention_dp_size
            output_throughput = output_throughput * scale_factor
            concurrency = b * scale_factor

            request_rate = output_throughput / (osl - 1)
            if b > 1:
                # will not be corrected by balance score when it's larger than 1.0
                # in order to indicate what's happening
                num_tokens = num_gen_requests + ctx_tokens
            else:
                num_tokens = ctx_tokens
            memory = self._get_memory_usage(model, database, b, 1, isl, osl, num_tokens, prefix=prefix)
            tp = model.config.tp_size
            pp = model.config.pp_size
            dp = model.config.attention_dp_size
            moe_tp = model.config.moe_tp_size
            moe_ep = model.config.moe_ep_size
            tokens_s_gpu = output_throughput / pp / tp / dp
            tokens_s_user = 1000 / tpot
            seq_s = request_rate
            seq_s_gpu = seq_s / pp / tp / dp
            tokens_s = output_throughput
            request_latency = ttft + tpot * max(osl - 1, 0)
            num_total_gpus = tp * pp * dp
            parallel = f"tp{tp}pp{pp}dp{dp}etp{moe_tp}ep{moe_ep}"
            gemm = model.config.gemm_quant_mode.name
            kvcache = model.config.kvcache_quant_mode.name
            fmha = model.config.fmha_quant_mode.name
            moe = model.config.moe_quant_mode.name
            comm = model.config.comm_quant_mode.name
            mem = memory["total"]

            result_dict = {
                "model": model.model_path,
                "isl": isl,
                "osl": osl,
                "prefix": prefix,
                "concurrency": concurrency,
                "request_rate": request_rate,
                "bs": b,
                "global_bs": b * model.config.attention_dp_size,
                "ttft": ttft,
                "tpot": tpot,
                "seq/s": seq_s,
                "seq/s/gpu": seq_s_gpu,
                "tokens/s": tokens_s,
                "tokens/s/gpu": tokens_s_gpu,
                "tokens/s/user": tokens_s_user,
                "request_latency": request_latency,
                "num_total_gpus": num_total_gpus,
                "tp": tp,
                "pp": pp,
                "dp": dp,
                "moe_tp": moe_tp,
                "moe_ep": moe_ep,
                "parallel": parallel,
                "gemm": gemm,
                "kvcache": kvcache,
                "fmha": fmha,
                "moe": moe,
                "comm": comm,
                "memory": mem,
                "balance_score": balance_score,
                "num_ctx_reqs": num_ctx_requests,
                "num_gen_reqs": num_gen_requests,
                "num_tokens": num_tokens,
                "ctx_tokens": ctx_tokens,
                "gen_tokens": num_gen_requests,
                "backend": database.backend,
                "version": database.version,
                "system": database.system,
                "power_w": agg_power_avg_w,  # Weighted average power for AGG mode
            }
            result = pd.DataFrame([result_dict], columns=common.ColumnsAgg).round(3)
            summary = InferenceSummary(RuntimeConfig(isl=isl, osl=osl))
            summary.set_memory_and_check_oom(memory, database.system_spec["gpu"]["mem_capacity"])
            summary.set_summary_df(result)
            summary.set_result_dict(result_dict)

            # Store per-ops latency breakdown
            per_ops_data["scheduling"] = {
                "num_mix_steps": float(num_mix_steps),
                "num_genonly_steps": float(num_genonly_steps),
                "mix_step_latency_ms": float(mix_step_latency_ms),
                "genonly_step_latency_ms": float(genonly_step_latency_ms),
            }
            # scheduling entries (num_*_steps, *_step_latency_ms) are scheduling math
            # / aggregate sums, not DB queries -- no per-op source applies. Skip them
            # in per_ops_source; per_ops_data still carries the values themselves.
            summary.set_per_ops_data(per_ops_data)
            summary.set_per_ops_source(per_ops_source)

            # caching
            self._agg_cache[isl][osl][b][ctx_tokens] = summary

        return summary

    def _run_chunked_context_phase(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        batch_size: int,
        isl: int,
        ctx_tokens: int,
        prefix: int,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
        """Compute context-phase latency with vLLM chunked prefill simulation.

        When ``--enable-chunked-prefill`` is on and ``ctx_tokens < isl``,
        vLLM splits the prefill into multiple forward passes (chunks).
        Each chunk processes ``ctx_tokens`` new tokens, but attention must
        attend to all previously computed KV entries (growing sequence
        length across chunks).

        This correctly models the per-chunk GPU utilisation and attention
        cost that differs from a hypothetical single-pass computation.

        This is vLLM-specific because chunked prefill semantics (how
        chunks are scheduled, how attention KV is managed across chunks)
        vary by backend.

        Args:
            model: The model to evaluate.
            database: The performance database.
            runtime_config: Runtime config (seq_imbalance_correction_scale used).
            batch_size: Number of requests in the batch.
            isl: Total input sequence length per request.
            ctx_tokens: Token budget per scheduling step (max_num_batched_tokens).
            prefix: Prefix length already computed (subtracted from isl).
        """
        context_latency_dict: dict[str, float] = defaultdict(float)
        context_energy_wms_dict: dict[str, float] = defaultdict(float)
        context_source_dict: dict[str, str] = {}

        effective_isl = isl - prefix
        if effective_isl <= 0:
            raise ValueError(f"isl must be greater than 0 after removing prefix, but got {effective_isl}")

        # Number of tokens the prefill engine can process per step for this batch.
        # With batch_size requests, the per-request chunk size is ctx_tokens // batch_size.
        tokens_per_request_per_step = max(1, ctx_tokens // batch_size)
        num_chunks = -(-effective_isl // tokens_per_request_per_step)  # ceil division

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * tokens_per_request_per_step
            chunk_tokens = min(tokens_per_request_per_step, effective_isl - chunk_start)
            cumulative_seq_len = chunk_start + chunk_tokens

            # Attention prefix for this chunk: tokens from prior chunks
            # are already in KV cache and should not be re-computed.
            # By passing prefix=chunk_start, the attention cost model
            # correctly charges only for the new query tokens attending
            # to the full KV (incremental chunk work), instead of a full
            # causal pass at cumulative_seq_len.
            attn_prefix = prefix + chunk_start

            for op in model.context_ops:
                if "logits_gemm" in op._name:
                    # logits_gemm only runs on the final chunk
                    if chunk_idx < num_chunks - 1:
                        continue
                    x = batch_size
                else:
                    x = batch_size * chunk_tokens

                # For attention: s=chunk_tokens (new queries), prefix=attn_prefix (cached KV)
                # full_s = chunk_tokens + attn_prefix = cumulative_seq_len + prefix(outer)
                is_attention = "attention" in op._name
                op_s = chunk_tokens if is_attention else cumulative_seq_len
                op_prefix = attn_prefix if is_attention else prefix

                result = op.query(
                    database,
                    x=x,
                    batch_size=batch_size,
                    beam_width=1,
                    s=op_s,
                    prefix=op_prefix,
                    model_name=getattr(model, "model_name", ""),
                    seq_imbalance_correction_scale=runtime_config.seq_imbalance_correction_scale,
                )
                context_latency_dict[op._name] += float(result)
                context_energy_wms_dict[op._name] += getattr(result, "energy", 0.0)
                new_src = getattr(result, "source", "silicon")
                existing = context_source_dict.get(op._name)
                if existing is None or existing == new_src:
                    context_source_dict[op._name] = new_src
                else:
                    context_source_dict[op._name] = "mixed"

        return context_latency_dict, context_energy_wms_dict, context_source_dict

    def find_best_agg_result_under_constraints(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Find the best agg result under constraints.

        Args:
            model: the model to be tested
            database: the database to be tested
            runtime_config: the runtime configuration
            top_k: the number of best results to return
            max_batch_size: the maximum batch size to test
            ctx_stride: the stride of ctx tokens to test, it will impact the time to run the test.
            enable_chunked_prefill: whether to enable chunked prefill, it will impact the time to
                run the test while have little impact on the result. Default off.

        Returns:
            A summary of the best agg result under constraints.
        """
        isl = runtime_config.isl
        osl = runtime_config.osl
        ttft = runtime_config.ttft
        tpot = runtime_config.tpot
        prefix = runtime_config.prefix
        top_k = kwargs.get("top_k", 1)
        max_batch_size = kwargs.get("max_batch_size", 512)
        ctx_stride = kwargs.get("ctx_stride", 512)
        enable_chunked_prefill = kwargs.get("enable_chunked_prefill", False)

        # when b is larger than 1024, the result is not good as the data collection is not enough
        # to cover this.
        b_list_default = (
            list(range(1, 16, 1))
            + list(range(16, 32, 4))
            + list(range(32, 64, 8))
            + list(range(64, 256, 16))
            + list(range(256, 512, 32))
            + list(range(512, 1024, 256))
            + [1024]
        )

        # sweep for batch_size and ctx_tokens
        # ctx_tokens will have a step of ctx_stride. When it's larger than 8192, we will increase
        # the step to ctx_stride_large.
        # outer_loop is over batch_size dimention, from 1 to max_batch_size
        # inner_loop is over ctx_tokens dimention, from 0 to max_ctx_tokens where it's
        # max(8192, 4*isl).
        # during the loop, as b, ctx_tokens and system memory are monotonic, we can break the
        # inner loop when the system is oom.
        b_list = [b for b in b_list_default if b <= max_batch_size]
        ctx_tokens_list = self._get_ctx_tokens_list_for_agg_sweep(isl, ctx_stride, enable_chunked_prefill)

        results_df = pd.DataFrame(columns=common.ColumnsAgg)
        results_dict_list = []
        results_per_ops_source: list[dict | None] = []
        capped_b = []
        all_oom = True
        for b in b_list:
            for ctx_tokens in ctx_tokens_list:
                if b - np.ceil(ctx_tokens / isl) < 0:  # allow b==1
                    break

                if b > 1 and (
                    b - np.ceil(ctx_tokens / isl) < 1
                ):  # general case, to ensure there's at least one gen req
                    break

                # filter out repeated records for balance score correction
                balance_score = isl * b / ctx_tokens / osl
                if balance_score > 1:
                    gen_tokens = b // balance_score
                    if gen_tokens > 1 and gen_tokens in capped_b:
                        continue
                    else:
                        capped_b.append(gen_tokens)

                summary = self.run_agg(
                    model=model,
                    database=database,
                    runtime_config=RuntimeConfig(
                        batch_size=b,
                        isl=isl,
                        osl=osl,
                        prefix=prefix,
                        seq_imbalance_correction_scale=runtime_config.seq_imbalance_correction_scale,
                    ),
                    ctx_tokens=ctx_tokens,
                )

                if summary.check_oom() or summary.check_kv_cache_oom():
                    break  # larger ctx tokens will cause oom
                all_oom = False
                result_dict = summary.get_result_dict()
                if result_dict and result_dict["tpot"] <= tpot and result_dict["ttft"] <= ttft:
                    results_dict_list.append(result_dict)
                    results_per_ops_source.append(summary.get_per_ops_source())

        if results_dict_list:
            results_df = pd.DataFrame(results_dict_list, columns=common.ColumnsAgg).round(3)
            # Carry per-row per_ops_source as an object column (stripped before CSV write).
            results_df["_per_ops_source"] = results_per_ops_source

        sorted_results_df = results_df.sort_values(by="seq/s", ascending=False).round(3)
        if top_k > 0:
            sorted_results_df = sorted_results_df.head(top_k)

        summary = InferenceSummary(runtime_config)
        summary.set_summary_df(sorted_results_df)
        summary.set_oom(all_oom)
        return summary

    def _get_memory_usage(
        self,
        model: BaseModel,
        database: PerfDatabase,
        batch_size: int,
        beam_width: int,
        isl: int,
        osl: int,
        num_tokens: int = 0,
        prefix: int = 0,
    ) -> dict[str, float]:
        # TODO
        from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend

        return TRTLLMBackend()._get_memory_usage(
            model, database, batch_size, beam_width, isl, osl, num_tokens, prefix=prefix
        )

    # ============== Disagg queueing model ==============================

    @property
    def use_queue_model(self) -> bool:
        """Whether the queueing-aware TTFT correction model is enabled."""
        return os.environ.get(_TTFT_QUEUEING_MODEL_ENV, "1").lower() in ("true", "1")

    def compute_ttft_correction_factor(
        self,
        decode_concurrency: int,
        prefill_num_worker: int = 1,
        t_decode_ms: float = 0.0,
        t_prefill_ms: float = 0.0,
        prefill_bs: int = 1,
        decoder_max_concurrent: int = 0,
    ) -> float:
        """Compute TTFT correction factor using vLLM queueing model.

        When ``AICONFIG_TTFT_QUEUE_MODEL`` is **not** enabled, returns the
        legacy fixed factor (1.8).

        When enabled, uses a queueing model that accounts for prefill
        batch size and number of request iterations (waves).
        """
        if not self.use_queue_model:
            return self._LEGACY_TTFT_CORRECTION_FACTOR

        P = max(prefill_num_worker, 1)
        B = max(prefill_bs, 1)
        C = decode_concurrency
        lc = C / P  # load per prefill worker

        # Number of request iterations (waves) -- controls first-wave dilution.
        try:
            num_request_iters = int(os.environ.get(_TTFT_NUM_REQUEST_ITERS_ENV, _TTFT_NUM_REQUEST_ITERS_DEFAULT))
        except (ValueError, TypeError):
            num_request_iters = _TTFT_NUM_REQUEST_ITERS_DEFAULT
        num_request_iters = max(num_request_iters, 1)

        if t_prefill_ms > 0 and t_decode_ms > 0:
            if B == 1:
                factor, _ = _simulate_prefill_queue(
                    int(lc), t_prefill_ms, t_decode_ms, num_request_iters,
                    decoder_max_concurrent=decoder_max_concurrent,
                )
            else:
                R_eff = B * t_decode_ms / t_prefill_ms
                if lc > R_eff:
                    first_wave = (lc / B + 1) / 2
                    steady_state = 1.0 + (lc - R_eff) / B
                    factor = (first_wave + (num_request_iters - 1) * steady_state) / num_request_iters
                else:
                    factor = 1.15 + (lc / B - 1) / (2 * num_request_iters)
        else:
            factor = 1.1 + (lc / B - 1) / (2 * num_request_iters)

        return max(factor, 1.0)

    def compute_effective_decode_bs(
        self,
        decode_concurrency: int,
        prefill_num_worker: int = 1,
        t_decode_ms: float = 0.0,
        t_prefill_ms: float = 0.0,
        prefill_bs: int = 1,
        decoder_max_concurrent: int = 0,
    ) -> float:
        """Compute effective decode batch size using vLLM queueing model.

        In disagg serving, a finished-decode slot stays empty while its
        replacement goes through the prefill queue.  The average decode
        occupancy is therefore lower than the configured concurrency.
        """
        P = max(prefill_num_worker, 1)
        B = max(prefill_bs, 1)
        C = decode_concurrency
        lc = C / P

        if t_prefill_ms <= 0 or t_decode_ms <= 0:
            return float(C)

        if B == 1:
            try:
                num_request_iters = int(
                    os.environ.get(_TTFT_NUM_REQUEST_ITERS_ENV, _TTFT_NUM_REQUEST_ITERS_DEFAULT)
                )
            except (ValueError, TypeError):
                num_request_iters = _TTFT_NUM_REQUEST_ITERS_DEFAULT
            num_request_iters = max(num_request_iters, 1)

            _, avg_occupancy = _simulate_prefill_queue(
                int(lc), t_prefill_ms, t_decode_ms, num_request_iters,
                decoder_max_concurrent=decoder_max_concurrent,
            )
            return avg_occupancy * P
        else:
            mean_ttft = t_prefill_ms * self.compute_ttft_correction_factor(
                C, P, t_decode_ms, t_prefill_ms, B,
                decoder_max_concurrent=decoder_max_concurrent,
            )
            return C * t_decode_ms / (mean_ttft + t_decode_ms)


# ---------------------------------------------------------------------------
# Disagg queueing simulation helper (used by VLLMBackend methods above)
# ---------------------------------------------------------------------------


def _simulate_prefill_queue(
    concurrency: int,
    t_prefill_ms: float,
    t_decode_ms: float,
    num_request_iters: int = 10,
    decoder_max_concurrent: int = 0,
) -> tuple[float, float]:
    """Simulate the prefill queue for B=1 to compute mean TTFT and decode occupancy.

    Models the exact deterministic queueing of a max-concurrency benchmark:
    - C requests arrive at t=0 (first-wave burst).
    - The prefiller processes one request at a time (FIFO), taking
      t_prefill_ms each.
    - If decoder_max_concurrent > 0, at most that many requests can
      decode simultaneously (KV cache capacity constraint).  Excess
      requests wait for a decoder slot, adding to their TTFT.
    - When a request finishes decode (after t_decode_ms), a new request
      is immediately submitted to the prefill queue.
    - Total requests = C * num_request_iters.

    Returns:
        (ttft_factor, avg_decode_occupancy):
            ttft_factor: mean_TTFT / t_prefill_ms (correction factor).
            avg_decode_occupancy: average number of requests in the decode
                phase at any point -- i.e. the effective decode batch size
                for a single prefill worker's concurrency pool.
    """
    import heapq

    C = concurrency
    N = num_request_iters
    total_requests = C * N

    # Ring buffer: decode_finish_times[i % C] = when request i finishes decode
    decode_finish_ring = [0.0] * C

    prefill_free_at = 0.0
    total_ttft = 0.0

    # Track decode occupancy via time-weighted counting.
    # Each request contributes decode_time to total_decode_time.
    # Simulation wall time = when last request finishes decode.
    total_decode_time = 0.0  # sum of decode durations across all requests

    # Decoder capacity: limits how many requests can decode simultaneously.
    use_decoder_cap = decoder_max_concurrent > 0 and decoder_max_concurrent < C
    if use_decoder_cap:
        # Priority queue of times when decoder slots become free.
        decoder_slots: list[float] = [0.0] * decoder_max_concurrent
        heapq.heapify(decoder_slots)

    for i in range(total_requests):
        if i < C:
            arrive = 0.0
        else:
            arrive = decode_finish_ring[i % C]

        prefill_start = max(arrive, prefill_free_at)
        prefill_end = prefill_start + t_prefill_ms
        prefill_free_at = prefill_end

        if use_decoder_cap:
            # Wait for a decoder slot to become available.
            slot_available = heapq.heappop(decoder_slots)
            decode_start = max(prefill_end, slot_available)
        else:
            decode_start = prefill_end

        # TTFT = time from client submission to first token.
        ttft = decode_start - arrive
        total_ttft += ttft

        decode_finish = decode_start + t_decode_ms
        decode_finish_ring[i % C] = decode_finish
        total_decode_time += t_decode_ms

        if use_decoder_cap:
            heapq.heappush(decoder_slots, decode_finish)

    mean_ttft = total_ttft / total_requests
    ttft_factor = mean_ttft / t_prefill_ms

    # Average decode occupancy = total decode-seconds / wall-clock time
    wall_time = max(decode_finish_ring)
    avg_decode_occupancy = total_decode_time / wall_time if wall_time > 0 else float(C)

    return ttft_factor, avg_decode_occupancy

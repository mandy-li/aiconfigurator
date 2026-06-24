# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import functools
import logging
import math
import warnings

import pandas as pd

from aiconfigurator.sdk import common, config, models, perf_database
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.errors import NoFeasibleConfigError
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.picking import (
    _RATE_MATCHING_DECODE_DEGRADATION_FACTOR,
    _RATE_MATCHING_PREFILL_DEGRADATION_FACTOR,
    _build_disagg_summary_dict,
)
from aiconfigurator.sdk.utils import enumerate_ttft_tpot_constraints

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# Admit-gate slack: real engines tolerate slight preemption above the
# conservative GPU-memory estimate. Applied ONLY in
# ``_check_kv_capacity_exceeded``; the queueing simulator stalls at the raw
# cap to match real engine behaviour.
_KV_CAPACITY_BUFFER = 1.05


def _check_kv_capacity_exceeded(
    total_decode_concurrency: int,
    decode_num_worker: int,
    max_kv_slots: int,
) -> bool:
    """Check if total decode concurrency exceeds KV cache capacity.

    Applies the ``_KV_CAPACITY_BUFFER`` slack (real engines tolerate slight
    preemption above the conservative GPU-memory estimate). Used only as the
    boolean admit gate; the queueing simulator stalls at the raw cap.

    Args:
        total_decode_concurrency: Total decode requests across all workers.
        decode_num_worker: Number of decode workers.
        max_kv_slots: Raw maximum KV cache slots per decode worker.

    Returns:
        True if capacity is exceeded, False otherwise.
    """
    ceiling = math.ceil(max_kv_slots * _KV_CAPACITY_BUFFER)
    return total_decode_concurrency > ceiling * decode_num_worker


def _model_config_with_parallel(base_mc: config.ModelConfig, row: dict) -> config.ModelConfig:
    """Copy ``base_mc`` and stamp parallel dims from a candidate row.

    Used before ``run_disagg`` so MoE models have ``moe_tp_size`` /
    ``moe_ep_size`` set when ``models.get_model`` runs
    ``resolve_moe_parallelism``.

    Args:
        base_mc: Base model config (parallel dims may be unset).
        row: Candidate row with ``tp``, ``pp``, ``dp``, ``moe_tp``, ``moe_ep`` keys.

    Returns:
        A deep-copied ``ModelConfig`` with parallel dims stamped from ``row``.
    """
    mc = copy.deepcopy(base_mc)
    mc.tp_size = int(row["tp"])
    mc.pp_size = int(row["pp"])
    mc.attention_dp_size = int(row["dp"])
    mc.moe_tp_size = int(row["moe_tp"])
    mc.moe_ep_size = int(row["moe_ep"])
    return mc


class InferenceSession:
    """
    InferenceSession holds the model and database to run inference loop

    Attributes:
        model (models.BaseModel): the model to run inference
        database (perf_database.PerfDatabase): the database to run inference
        backend (backend.Backend): the backend to run inference

    Methods:
        run_static (static, static_ctx, static_gen): to support static batching and disagg,
            returns details of a static run
        run_agg (static, static_ctx, static_gen): run agg inference, returns summary of the
            perf result with given agg config and runtime config (concurrency)
        find_best_agg_result_under_constraints (static, static_ctx, static_gen):
            find the best agg result under constraints, returns summary
            which contains all the possible agg config and perf that matchs SLA.
    """

    def __init__(self, model: models.BaseModel, database: perf_database.PerfDatabase, backend: BaseBackend) -> None:
        """
        Initialize the InferenceSession
        """
        self._model = model
        self._database = database
        self._backend = backend

    def run_static(
        self,
        runtime_config: config.RuntimeConfig,
        mode: str,
        stride: int = 32,
        latency_correction_scale: float = 1.0,
    ) -> InferenceSummary:
        """
        Run static inference

        Args:
            runtime_config (RuntimeConfig): the runtime config
            mode (str): the mode to run inference, static, static_ctx, static_gen
            stride (int): the stride is used to accelerate the estimation, for a give osl,
                will only computes the i, i+stride, i+2*stride, ... step, default is 32.

        Returns:
            InferenceSummary: the summary of the inference result
        """
        return self._backend.run_static(
            self._model, self._database, runtime_config, mode, stride, latency_correction_scale
        )

    def run_static_latency_only(
        self,
        runtime_config: config.RuntimeConfig,
        mode: str,
        stride: int = 32,
        latency_correction_scale: float = 1.0,
    ) -> float:
        """
        Run static inference and return only scalar latency in milliseconds.

        Args:
            runtime_config (RuntimeConfig): the runtime config
            mode (str): the mode to run inference, static, static_ctx, static_gen
            stride (int): the stride is used to accelerate the estimation, for a give osl,
                will only computes the i, i+stride, i+2*stride, ... step, default is 32.

        Returns:
            float: the total latency in milliseconds
        """
        return self._backend.run_static_latency_only(
            self._model, self._database, runtime_config, mode, stride, latency_correction_scale
        )

    def run_agg(self, runtime_config: config.RuntimeConfig, **kwargs) -> InferenceSummary:
        """
        Run agg inference

        Args:
            runtime_config (RuntimeConfig): the runtime config
            **kwargs: other arguments to run agg, depends on the backend specific design

        Returns:
            InferenceSummary: the summary of the inference result
        """
        return self._backend.run_agg(self._model, self._database, runtime_config, **kwargs)

    # Optimization
    def find_best_agg_result_under_constraints(
        self, runtime_config: config.RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Find the best agg result under constraints

        Args:
            runtime_config (RuntimeConfig): the runtime config
            **kwargs: other arguments to find the best agg result under constraints,
                depends on the backend specific design

        Returns:
            InferenceSummary: the summary of the inference result, contains all the possible
                agg config and perf that matchs SLA.
        """
        return self._backend.find_best_agg_result_under_constraints(
            self._model, self._database, runtime_config, **kwargs
        )


DECODE_FILTER_RATIO_MIN = 0.0
DECODE_FILTER_RATIO_MAX = 1.0
MAX_DECODE_WORKERS_PER_CATEGORY = 16
MAX_PREFILL_WORKERS = 32
MAX_NUM_DECODE_WORKER_CANDIDATES = 64
MAX_NUM_PREFILL_WORKER_CANDIDATES = 32


class DisaggInferenceSession:
    """
    Disaggregated inference session
    Run prefill and generation separately, with different models (parallel and precision config can
    be different) and databases
    0. init func only takes database and backend, model is passed in run_disagg
    1. run_disagg, given model, database and backend, given everything fixed ((max)batchsize and
       num_workers) , return the perf result of the system
    2. find_best_disagg_result_under_constraints, given database and backend, sweep batchsize and
       model parallel to match SLA, sweep workers to get best system perf/gpu if allowed.
       Return config (parallel, batchsize and num_workers) and perf.
    3. TODO, should consider kvcache model in future
    Disagg is more like a post processing step to do rate matching, that's why it's a
    DiaggInferenceSession instread of using InferenceSession.

    Attributes:
        prefill_database (perf_database.PerfDatabase): the database to run prefill
        prefill_backend (backend.Backend): the backend to run prefill
        decode_database (perf_database.PerfDatabase): the database to run decode
        decode_backend (backend.Backend): the backend to run decode

    Methods:
        run_disagg (model_path, runtime_config, prefill_model_config, prefill_batch_size,
                    prefill_num_worker, decode_model_config, decode_batch_size,
                    decode_num_worker)
            run disagg with given prefill/decode worker info
        find_best_disagg_result_under_constraints (model_path,runtime_config, prefill_model_config,
                    prefill_parallel_config_list, prefill_max_num_tokens, prefill_num_worker_list,
                    decode_model_config, decode_parallel_config_list, decode_max_num_tokens,
                    decode_num_worker_list, num_gpu_list)
            find the best disagg result under constraints
        set_latency_correction_scales (prefill_latency_correction_scale,
                                       decode_latency_correction_scale):
            set the correction scales for better alignment with real system
    """

    def __init__(
        self,
        prefill_database: perf_database.PerfDatabase,
        prefill_backend: BaseBackend,
        decode_database: perf_database.PerfDatabase,
        decode_backend: BaseBackend,
        encoder_database: perf_database.PerfDatabase | None = None,
        encoder_backend: BaseBackend | None = None,
    ) -> None:
        """
        Initialize the DisaggInferenceSession
        """
        self._prefill_database = prefill_database
        self._prefill_backend = prefill_backend
        self._decode_database = decode_database
        self._decode_backend = decode_backend
        self._encoder_database = encoder_database
        self._encoder_backend = encoder_backend

        # allow user to set correction scales for better alignment with real system
        # now the corection scales are used to correct the latency, not throughput,
        # corrected latency = latency * correction_scale
        self._prefill_latency_correction_scale = 1.0
        self._decode_latency_correction_scale = 1.0
        self._encoder_latency_correction_scale = 1.0

        self._rate_matching_prefill_degradation_factor = _RATE_MATCHING_PREFILL_DEGRADATION_FACTOR
        self._rate_matching_decode_degradation_factor = _RATE_MATCHING_DECODE_DEGRADATION_FACTOR

        # Optional caches populated by the picker (find_best_disagg_*) to share
        # per-(parallel, bs) static-run results across (P, D) variations of the
        # same combo. None disables caching (default for single-point use).
        self._static_run_cache: dict | None = None
        self._max_kv_slots_cache: dict | None = None
        self._chunked_prefill_cache: dict | None = None

    def set_latency_correction_scales(
        self,
        prefill_latency_correction_scale: float,
        decode_latency_correction_scale: float,
        encoder_latency_correction_scale: float = 1.0,
    ):
        """
        Set the correction scales for better alignment with real system
        """
        self._prefill_latency_correction_scale = prefill_latency_correction_scale
        self._decode_latency_correction_scale = decode_latency_correction_scale
        self._encoder_latency_correction_scale = encoder_latency_correction_scale

    def set_rate_matching_degradation_factors(
        self,
        prefill_degradation_factor: float = _RATE_MATCHING_PREFILL_DEGRADATION_FACTOR,
        decode_degradation_factor: float = _RATE_MATCHING_DECODE_DEGRADATION_FACTOR,
    ):
        """
        Set the degradation factors used during rate matching between prefill and decode workers.

        Args:
            prefill_degradation_factor: Multiplicative factor applied to prefill throughput
                to account for pipeline bubbles (default 0.9).
            decode_degradation_factor: Multiplicative factor applied to decode throughput
                to account for batch-size under-saturation (default 0.92).
        """
        self._rate_matching_prefill_degradation_factor = prefill_degradation_factor
        self._rate_matching_decode_degradation_factor = decode_degradation_factor

    def _get_disagg_summary_df(
        self,
        prefill_summary_df: pd.DataFrame,
        prefill_num_worker: int,
        decode_summary_df: pd.DataFrame,
        decode_num_worker: int,
        decode_concurrency_override: int | None = None,
        decoder_max_concurrent: int = 0,
        max_kv_slots: int = 0,
        kv_capacity_exceeded: bool = False,
    ) -> pd.DataFrame:
        """
        Get the disagg summary df based on prefill and decode summary df
        """
        prefill_dict = prefill_summary_df.iloc[0].to_dict()
        decode_dict = decode_summary_df.iloc[0].to_dict()
        if decode_concurrency_override is not None:
            decode_concurrency = decode_concurrency_override
        else:
            decode_concurrency = int(decode_dict.get("bs", 1)) * decode_num_worker
        t_prefill = float(prefill_dict.get("ttft", 0))
        tpot = float(decode_dict.get("tpot", 0))
        osl = int(decode_dict.get("osl", prefill_dict.get("osl", 1)))
        t_decode = tpot * max(osl - 1, 0)
        prefill_bs = int(prefill_dict.get("bs", 1))
        ttft_correction = self._prefill_backend.compute_ttft_correction_factor(
            decode_concurrency, prefill_num_worker,
            t_decode_ms=t_decode, t_prefill_ms=t_prefill,
            prefill_bs=prefill_bs,
            decoder_max_concurrent=decoder_max_concurrent,
        )
        # Apply queueing correction, then add the KV-transfer overhead
        # (prefill_latency_correction_scale).  The correction scale is NOT
        # baked into the raw t_prefill used by the queueing model above.
        prefill_dict["ttft"] = (
            prefill_dict["ttft"] * ttft_correction
            * self._prefill_latency_correction_scale
        )

        summary_dict = _build_disagg_summary_dict(
            prefill_dict,
            prefill_num_worker,
            decode_dict,
            decode_num_worker,
            prefill_degradation_factor=self._rate_matching_prefill_degradation_factor,
            decode_degradation_factor=self._rate_matching_decode_degradation_factor,
            max_kv_slots=max_kv_slots,
            kv_capacity_exceeded=kv_capacity_exceeded,
        )
        return pd.DataFrame([summary_dict], columns=common.ColumnsDisagg).round(3)

    def _compute_max_kv_slots(
        self,
        decode_model: "models.BaseModel",
        database: "perf_database.PerfDatabase",
        isl: int,
        osl: int,
        gpu_memory_utilization: float = 0.95,
        block_size: int = 64,
    ) -> int:
        """Compute the maximum number of concurrent decode sequences that fit
        in GPU memory, based on KV cache capacity.

        Args:
            decode_model: The model instance (used for weights + KV sizing).
            database: Performance database (for system spec / GPU memory).
            isl: Input sequence length.
            osl: Output sequence length.
            gpu_memory_utilization: Fraction of GPU memory available (default 0.95).
            block_size: vLLM KV block size in tokens (default 64).

        Returns:
            Maximum number of sequences that can reside concurrently.
        """
        gpu_mem_bytes = database.system_spec["gpu"]["mem_capacity"]
        gpu_usable = gpu_mem_bytes * gpu_memory_utilization

        # Normalise kvcache_quant_mode from string -> enum so that
        # _get_memory_usage (which calls get_kvcache_bytes_per_sequence)
        # does not crash on the string form.
        orig_mode = decode_model.config.kvcache_quant_mode
        if isinstance(orig_mode, str):
            decode_model.config.kvcache_quant_mode = common.KVCacheQuantMode[orig_mode]

        try:
            # Non-KV overhead: weights + minimal activations + nccl + others
            mem = self._decode_backend._get_memory_usage(
                decode_model, database, batch_size=1, beam_width=1,
                isl=1, osl=1, num_tokens=1,
            )
        finally:
            decode_model.config.kvcache_quant_mode = orig_mode

        overhead_bytes = (mem["weights"] + mem["activations"] + mem["nccl"] + mem["others"]) * (1 << 30)

        kv_budget_bytes = gpu_usable - overhead_bytes
        if kv_budget_bytes <= 0:
            return 0

        # KV per sequence: elements_per_token * bytes_per_element * max_seq_len
        max_seq_len = isl + osl
        kv_elem_per_tok = decode_model.get_kvcache_elements_per_token()
        kvcache_mode = decode_model.config.kvcache_quant_mode
        if hasattr(kvcache_mode, "value"):
            bytes_per_elem = kvcache_mode.value.memory
        else:
            # String fallback: fp8 -> 1 byte, bf16/fp16 -> 2 bytes
            bytes_per_elem = 1 if "fp8" in str(kvcache_mode) else 2

        # Block-aligned: round up to blocks
        blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        kv_bytes_per_block = block_size * kv_elem_per_tok * bytes_per_elem
        total_blocks = int(kv_budget_bytes / kv_bytes_per_block)
        max_seqs = total_blocks // blocks_per_seq

        return max(max_seqs, 1)

    def run_disagg(
        self,
        model_path: str,
        runtime_config: config.RuntimeConfig,
        prefill_model_config: config.ModelConfig,
        prefill_batch_size: int,
        prefill_num_worker: int,
        decode_model_config: config.ModelConfig,
        decode_batch_size: int,
        decode_num_worker: int,
        ctx_tokens: int | None = None,
        enable_chunked_prefill: bool = False,
    ) -> InferenceSummary:
        """
        Run disagg with given prefill/decode worker info

        Args:
            model_path (str): the model name
            runtime_config (RuntimeConfig): the runtime config
            prefill_model_config (ModelConfig): the prefill model config
            prefill_batch_size (int): the prefill batch size
            prefill_num_worker (int): the number of prefill workers
            decode_model_config (ModelConfig): the decode model config
            decode_batch_size (int): the decode batch size
            decode_num_worker (int): the number of decode workers
            ctx_tokens (int | None): max_num_batched_tokens budget for the
                prefill engine.  When set and smaller than
                ``prefill_batch_size * isl``, the effective prefill batch
                size is reduced to ``max(1, ctx_tokens // isl)`` to model
                chunked-prefill behaviour where only one (or a few) requests
                can be processed per scheduling round.
            enable_chunked_prefill (bool): When True (and ``ctx_tokens`` is
                not explicitly provided), default ``ctx_tokens`` to 2048
                (vLLM's typical chunked-prefill budget). When False, skip
                the chunked-prefill recompute. Explicit ``ctx_tokens``
                always wins.

        Returns:
            InferenceSummary: the summary of the inference result
        """
        prefill_model = models.get_model(model_path, prefill_model_config, self._prefill_backend.name.value)
        decode_model = models.get_model(model_path, decode_model_config, self._decode_backend.name.value)
        prefill_sess = InferenceSession(
            model=prefill_model, database=self._prefill_database, backend=self._prefill_backend
        )
        decode_sess = InferenceSession(model=decode_model, database=self._decode_database, backend=self._decode_backend)

        def _parallel_sig(mc: config.ModelConfig) -> tuple:
            """Hashable parallel-dims signature for cache keys."""
            return (
                int(mc.tp_size),
                int(mc.pp_size),
                int(mc.attention_dp_size),
                int(mc.moe_tp_size or 1),
                int(mc.moe_ep_size or 1),
            )

        def _cached_static(
            side: str,
            sess: InferenceSession,
            model_cfg: config.ModelConfig,
            rt_cfg: config.RuntimeConfig,
            mode: str,
            scale: float,
        ) -> InferenceSummary:
            """Run ``sess.run_static``, memoizing on (parallel, mode, bs, ...).

            The cache is enabled only when ``self._static_run_cache`` is set
            (typically by ``find_best_disagg_result_under_constraints`` for
            the duration of one picker call). Returns a shallow-copied
            summary with a copied DataFrame so chunked-prefill / eff-bs
            callers can ``set_summary_df`` without polluting the cache.
            """
            cache = self._static_run_cache
            if cache is None:
                return sess.run_static(mode=mode, runtime_config=rt_cfg, latency_correction_scale=scale)
            key = (
                side,
                _parallel_sig(model_cfg),
                mode,
                int(rt_cfg.batch_size),
                int(rt_cfg.isl),
                int(rt_cfg.osl),
                int(rt_cfg.prefix),
                float(scale),
            )
            cached = cache.get(key)
            if cached is None:
                cached = sess.run_static(mode=mode, runtime_config=rt_cfg, latency_correction_scale=scale)
                cache[key] = cached
            clone = copy.copy(cached)
            df = cached.get_summary_df()
            if df is not None:
                clone.set_summary_df(df.copy())
            return clone

        use_queue_model = self._prefill_backend.use_queue_model

        # Honour explicit ctx_tokens; otherwise default from enable_chunked_prefill.
        if ctx_tokens is None and use_queue_model:
            ctx_tokens = 2048 if enable_chunked_prefill else None

        # Compute effective prefill batch size based on ctx_tokens budget.
        # In disagg, max_num_batched_tokens limits how many tokens the
        # prefill engine can process per scheduling step.
        isl = runtime_config.isl
        effective_prefill_bs = prefill_batch_size
        if use_queue_model and ctx_tokens is not None and ctx_tokens > 0 and isl > 0:
            max_concurrent = max(1, ctx_tokens // isl)
            effective_prefill_bs = min(prefill_batch_size, max_concurrent)

        prefill_runtime_config = copy.deepcopy(runtime_config)
        prefill_runtime_config.batch_size = effective_prefill_bs
        # Run prefill WITHOUT the correction scale so that t_prefill
        # fed into the queueing simulation reflects raw silicon latency.
        # The correction (KV-transfer overhead) is applied only to the
        # final reported TTFT, not to the effective-BS calculation.
        prefill_summary = _cached_static(
            "prefill", prefill_sess, prefill_model_config,
            prefill_runtime_config, "static_ctx", 1.0,
        )

        # When chunked prefill applies (ctx_tokens < isl), recompute the
        # prefill latency using per-chunk simulation instead of the single-pass
        # approximation from run_static.  This properly models the growing
        # attention KV length across chunks.
        # _run_chunked_context_phase is vLLM-specific (chunked prefill semantics
        # vary by backend), so only call it if the backend provides it.
        # Only applied when the queueing model is enabled (AICONFIG_TTFT_QUEUE_MODEL=1).
        if (
            use_queue_model
            and ctx_tokens is not None
            and ctx_tokens > 0
            and ctx_tokens < isl
            and hasattr(self._prefill_backend, "_run_chunked_context_phase")
        ):
            chunked_cache = self._chunked_prefill_cache
            chunked_key = (
                _parallel_sig(prefill_model_config), int(effective_prefill_bs),
                int(isl), int(runtime_config.prefix), int(ctx_tokens),
            )
            chunked_latency_dict = chunked_cache.get(chunked_key) if chunked_cache is not None else None
            if chunked_latency_dict is None:
                chunked_latency_dict, _, _ = self._prefill_backend._run_chunked_context_phase(
                    prefill_model, self._prefill_database, prefill_runtime_config,
                    batch_size=effective_prefill_bs, isl=isl,
                    ctx_tokens=ctx_tokens, prefix=runtime_config.prefix,
                )
                if chunked_cache is not None:
                    chunked_cache[chunked_key] = chunked_latency_dict
            # No latency correction here -- the KV-transfer overhead
            # is applied only to the final reported TTFT.

            chunked_total_ms = sum(chunked_latency_dict.values())

            # Patch the prefill summary DataFrame with corrected TTFT
            summary_df = prefill_summary.get_summary_df().copy()
            summary_df["ttft"] = chunked_total_ms
            summary_df["context_latency"] = chunked_total_ms
            summary_df["request_latency"] = chunked_total_ms  # static_ctx has no gen
            # Recalculate throughput metrics
            if chunked_total_ms > 0:
                global_bs = float(summary_df["global_bs"].iloc[0])
                pp = float(summary_df["pp"].iloc[0])
                seq_s = global_bs / chunked_total_ms * 1000 * pp
                tp = float(summary_df["tp"].iloc[0])
                dp = float(summary_df["dp"].iloc[0])
                summary_df["seq/s"] = seq_s
                summary_df["seq/s/gpu"] = seq_s / (tp * pp * dp)
                summary_df["tokens/s"] = seq_s * 1  # static_ctx: 1 first token
                summary_df["tokens/s/gpu"] = seq_s * 1 / (tp * pp * dp)
            prefill_summary.set_summary_df(summary_df)
            # Also update per-op context latency dict
            prefill_summary.set_context_latency_dict(chunked_latency_dict)

        # Cap decode BS at KV cache capacity before running decode.
        kv_cache = self._max_kv_slots_cache
        kv_key = (_parallel_sig(decode_model_config), int(isl), int(runtime_config.osl))
        max_kv_slots = kv_cache.get(kv_key) if kv_cache is not None else None
        if max_kv_slots is None:
            max_kv_slots = self._compute_max_kv_slots(
                decode_model, self._decode_database,
                isl=isl, osl=runtime_config.osl,
            )
            if kv_cache is not None:
                kv_cache[kv_key] = max_kv_slots
        # Decoder max concurrent for queueing simulation: vllm reserves
        # blocks for the full sequence (ISL + OSL), so the hard limit on
        # concurrent sequences per decode worker equals max_kv_slots.
        #
        # The queueing simulation operates at *per-prefill-worker* scope
        # (it is driven by lc = decode_concurrency / prefill_num_worker), so
        # the local decode load it sees is also per-prefill-worker.  The cap
        # must be expressed on the SAME footing: how many decode slots back a
        # single prefill worker's stream.  Decode capacity is a property of the
        # DECODE workers (``max_kv_slots * decode_num_worker``); the prefill
        # workers feed that shared pool, so per prefill worker the available
        # decode capacity is the pool divided by ``prefill_num_worker``.
        decoder_max_concurrent = max(1, int(max_kv_slots * max(decode_num_worker, 1)))
        kv_capped_dbs = min(decode_batch_size, max_kv_slots)

        decode_runtime_config = copy.deepcopy(runtime_config)
        decode_runtime_config.batch_size = kv_capped_dbs
        # When queueing model is enabled, the effective-BS re-run below
        # handles decode unsaturation, so skip the legacy correction scale.
        # When disabled, use the configured scale (default 1.08).
        decode_scale = 1.0 if use_queue_model else self._decode_latency_correction_scale
        decode_summary = _cached_static(
            "decode", decode_sess, decode_model_config,
            decode_runtime_config, "static_gen", decode_scale,
        )

        # Compute effective decode BS: in disagg, finished-decode slots stay
        # empty while their replacements go through the prefill queue.
        # The average decode occupancy is lower than the configured BS.
        # Use the ORIGINAL requested concurrency (not KV-capped) because
        # the benchmark sends decode_batch_size concurrent requests -- all
        # of them compete for the prefill queue even if only max_kv_slots
        # can decode simultaneously.
        # Only applied when the queueing model is enabled.
        if use_queue_model:
            prefill_summary_df = prefill_summary.get_summary_df()
            t_prefill = float(prefill_summary_df["ttft"].iloc[0])
            decode_summary_df = decode_summary.get_summary_df()
            tpot = float(decode_summary_df["tpot"].iloc[0])
            osl = int(decode_summary_df["osl"].iloc[0])
            t_decode = tpot * max(osl - 1, 0)
            original_concurrency = decode_batch_size * decode_num_worker

            eff_dbs = self._prefill_backend.compute_effective_decode_bs(
                original_concurrency, prefill_num_worker,
                t_decode_ms=t_decode, t_prefill_ms=t_prefill,
                prefill_bs=effective_prefill_bs,
                decoder_max_concurrent=decoder_max_concurrent,
            )
            eff_dbs_per_worker = eff_dbs / max(decode_num_worker, 1)
            # Effective decode BS can't exceed KV cache capacity
            eff_dbs_per_worker = min(eff_dbs_per_worker, kv_capped_dbs)

            # Re-run decode at eff bs for accurate TPOT; restore the requested
            # bs on the summary so (d)bs reflects what the user deploys.
            if eff_dbs_per_worker < kv_capped_dbs * 0.95:
                eff_dbs_rounded = max(1, round(eff_dbs_per_worker))
                decode_runtime_eff = copy.deepcopy(runtime_config)
                decode_runtime_eff.batch_size = eff_dbs_rounded
                decode_summary = _cached_static(
                    "decode", decode_sess, decode_model_config,
                    decode_runtime_eff, "static_gen", decode_scale,
                )
                decode_summary_df = decode_summary.get_summary_df().copy()
                decode_pp = int(decode_summary_df["pp"].iloc[0])
                decode_dp = int(decode_summary_df["dp"].iloc[0])
                decode_summary_df["bs"] = decode_batch_size
                decode_summary_df["global_bs"] = decode_batch_size * decode_dp
                decode_summary_df["concurrency"] = decode_batch_size * decode_pp * decode_dp
                decode_summary.set_summary_df(decode_summary_df)

        # Use the ORIGINAL requested decode concurrency for the TTFT
        # queueing simulation, not the KV-capped/effective BS.  The benchmark
        # sends decode_batch_size concurrent requests regardless of KV cap;
        # the KV cap only limits how many can decode simultaneously, but all
        # requests still compete for the prefill queue.
        original_decode_concurrency = decode_batch_size * decode_num_worker

        # Check if decode concurrency exceeds KV capacity with buffer.
        kv_capacity_exceeded = _check_kv_capacity_exceeded(
            original_decode_concurrency, decode_num_worker, max_kv_slots
        )

        disagg_summary_df = self._get_disagg_summary_df(
            prefill_summary.get_summary_df(),
            prefill_num_worker,
            decode_summary.get_summary_df(),
            decode_num_worker,
            decode_concurrency_override=original_decode_concurrency,
            decoder_max_concurrent=decoder_max_concurrent,
            max_kv_slots=max_kv_slots,
            kv_capacity_exceeded=kv_capacity_exceeded,
        )

        disagg_summary = InferenceSummary(runtime_config=runtime_config)

        # Always set OOM (True or False) so downstream check_oom() does not
        # warn about an unset memory status.  prefill/decode static summaries
        # have _is_oom populated by run_static -> set_memory_and_check_oom.
        prefill_oom = prefill_summary.check_oom()
        decode_oom = decode_summary.check_oom()
        disagg_summary.set_oom(bool(prefill_oom or decode_oom))

        # Flag configs exceeding KV capacity so they are excluded from frontier.
        if kv_capacity_exceeded:
            disagg_summary.set_kv_cache_oom(True)

        disagg_summary.set_summary_df(disagg_summary_df)

        # Carry per-op latency breakdowns from prefill/decode static runs
        per_ops_data = {}
        per_ops_source = {}
        prefill_ctx_latency = prefill_summary.get_context_latency_dict()
        if prefill_ctx_latency:
            per_ops_data["prefill"] = dict(prefill_ctx_latency)
        prefill_ctx_source = prefill_summary.get_context_source_dict()
        if prefill_ctx_source:
            per_ops_source["prefill"] = dict(prefill_ctx_source)
        decode_gen_latency = decode_summary.get_generation_latency_dict()
        if decode_gen_latency:
            per_ops_data["decode"] = dict(decode_gen_latency)
        decode_gen_source = decode_summary.get_generation_source_dict()
        if decode_gen_source:
            per_ops_source["decode"] = dict(decode_gen_source)
        if per_ops_data:
            disagg_summary.set_per_ops_data(per_ops_data)
        if per_ops_source:
            disagg_summary.set_per_ops_source(per_ops_source)

        return disagg_summary

    def get_worker_candidates(
        self,
        model_path: str,
        model_config: config.ModelConfig,
        parallel_config_list: list[tuple[int, int, int, int, int]],
        b_list: list[int] | range,
        runtime_config: config.RuntimeConfig,
        mode: str,
        latency_correction_scale: float = 1.0,
    ) -> pd.DataFrame:
        """Get all worker candidates for a given search space.

        It enumerates all (parallel_config, batch_size) combinations,
        runs static inference, and returns a DataFrame with columns from
        :data:`common.ColumnsStatic`.

        Args:
            model_path: HuggingFace model ID or local path.
            model_config: Model configuration (quant modes etc.).
            parallel_config_list: List of (tp, pp, dp, moe_tp, moe_ep) tuples.
            b_list: Batch sizes to sweep.
            runtime_config: Runtime config (isl, osl, etc.).
            mode: ``"static_ctx"`` for prefill or ``"static_gen"`` for decode.
            latency_correction_scale: Multiplicative correction applied to
                latencies (default 1.0).

        Returns:
            DataFrame with one row per (parallel_config, batch_size) that fits
            in memory.

        Raises:
            RuntimeError: If no valid results are found for any config.
        """
        summary_df = pd.DataFrame(columns=common.ColumnsStatic)
        exceptions: list[Exception] = []
        all_configs_oom = True

        for parallel_config in parallel_config_list:
            tp_size, pp_size, dp_size, moe_tp_size, moe_ep_size = parallel_config
            logger.debug(
                "Getting candidate workers with parallel config: tp=%d, pp=%d, dp=%d, moe_tp=%d, moe_ep=%d",
                tp_size,
                pp_size,
                dp_size,
                moe_tp_size,
                moe_ep_size,
            )

            try:
                overwritten_model_config = copy.deepcopy(model_config)
                overwritten_model_config.pp_size = pp_size
                overwritten_model_config.tp_size = tp_size
                overwritten_model_config.moe_tp_size = moe_tp_size
                overwritten_model_config.moe_ep_size = moe_ep_size
                overwritten_model_config.attention_dp_size = dp_size
                model = models.get_model(
                    model_path=model_path,
                    model_config=overwritten_model_config,
                    backend_name=self._prefill_backend.name.value,
                )
                if mode == "static_ctx":
                    sess = InferenceSession(
                        model=model,
                        database=self._prefill_database,
                        backend=self._prefill_backend,
                    )
                else:
                    sess = InferenceSession(
                        model=model,
                        database=self._decode_database,
                        backend=self._decode_backend,
                    )

                for b in b_list:
                    overwritten_runtime_config = copy.deepcopy(runtime_config)
                    overwritten_runtime_config.batch_size = b
                    summary = sess.run_static(
                        mode=mode,
                        runtime_config=overwritten_runtime_config,
                        latency_correction_scale=latency_correction_scale,
                    )
                    if not summary.check_oom():
                        all_configs_oom = False
                        summary_df = pd.concat(
                            [summary_df, summary.get_summary_df()],
                            axis=0,
                            ignore_index=True,
                        )
                    else:  # larger b will always OOM
                        break
            except Exception as e:
                logger.warning(
                    "Error getting candidate workers with parallel config: "
                    "tp=%d, pp=%d, dp=%d, moe_tp=%d, moe_ep=%d; "
                    "skipping this combination. Error: %s",
                    tp_size,
                    pp_size,
                    dp_size,
                    moe_tp_size,
                    moe_ep_size,
                    e,
                )
                exceptions.append(e)
                continue
        if summary_df.empty:
            if exceptions:
                raise RuntimeError(
                    f"No results found for any parallel configuration. Showing last exception: {exceptions[-1]}"
                ) from exceptions[-1]
            if all_configs_oom:
                raise RuntimeError(
                    "No results found: the model does not fit in GPU memory for any parallel "
                    "configuration. Try increasing --total-gpus, using a quantized model, or "
                    "using a system with more VRAM per GPU."
                )
            raise NoFeasibleConfigError(
                "No results found for any parallel configuration. No configuration satisfied the "
                "TTFT/TPOT or request-latency constraints. Try relaxing --ttft, --tpot, or "
                "--request_latency (e.g., higher ttft/tpot or higher request_latency)."
            )
        return summary_df

    def _pick_autoscale(
        self,
        prefill_summary_df: pd.DataFrame,
        decode_summary_df: pd.DataFrame,
        runtime_config: config.RuntimeConfig,
        disagg_summary: InferenceSummary,
        target_ttft: float | None = None,
        target_tpot: float | None = None,
        top_n: int = 5,
    ) -> InferenceSummary:
        """Pick best prefill and decode engines independently for autoscaling.

        Delegates to :func:`aiconfigurator.sdk.picking.pick_autoscale` and
        wraps the result in an ``InferenceSummary``.
        """
        from aiconfigurator.sdk.picking import pick_autoscale

        if target_ttft is None:
            target_ttft = runtime_config.ttft

        if target_tpot is None:
            tpot_values = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
            target_tpot = max(tpot_values)

        result = pick_autoscale(
            prefill_df=prefill_summary_df,
            decode_df=decode_summary_df,
            target_ttft=target_ttft,
            target_tpot=target_tpot,
            top_n=top_n,
            ttft_correction_fn=self._prefill_backend.compute_ttft_correction_factor,
        )

        disagg_summary_df = result["best_config_df"]
        if not disagg_summary_df.empty:
            disagg_summary.set_summary_df(disagg_summary_df)
        return disagg_summary

    # optimization
    def find_best_disagg_result_under_constraints(
        self,
        model_path: str,
        runtime_config: config.RuntimeConfig,
        prefill_model_config: config.ModelConfig,
        prefill_parallel_config_list: list[tuple[int, int, int, int, int]],
        prefill_max_num_tokens: int,
        prefill_num_worker_list: list[int],
        decode_model_config: config.ModelConfig,
        decode_parallel_config_list: list[tuple[int, int, int, int, int]],
        decode_max_num_tokens: int,
        decode_num_worker_list: list[int],
        num_gpu_list: list[int] | None,
        max_prefill_gpus: int | None = None,
        max_decode_gpus: int | None = None,
        require_same_tp: bool = False,
        autoscale: bool = False,
        target_tpot: float | None = None,
        enable_chunked_prefill: bool = False,
    ) -> InferenceSummary | None:
        """
        Run disagg with given constraints
        1. get all summary df, which matches the constraints
        2. find best config under constraints, call match scales to get the best scale
        3. call a func to get disagg_summary_df (this is shared by run_disgg func)
        4. return summary
        5. several empirical values:
            - 0.7 is the threshold to filter decode workers, because the performance of
              decode workers is much lower than prefill workers
            - 5 is the top k to return for drawing pareto frontier of each tpot

        Args:
            model_path (str): the model name
            runtime_config (RuntimeConfig): the runtime config
            prefill_model_config (ModelConfig): the prefill model config
            prefill_parallel_config_list (List[Tuple[int, int, int, int, int]]):
                the prefill parallel config list
            prefill_max_num_tokens (int): the prefill max num tokens
            prefill_num_worker_list (List[int]): the prefill num worker list
            decode_model_config (ModelConfig): the decode model config
            decode_parallel_config_list (List[Tuple[int, int, int, int, int]]):
                the decode parallel config list
            decode_max_num_tokens (int): the decode max num tokens
            decode_num_worker_list (List[int]): the decode num worker list
            num_gpu_list (Optional[List[int]]): the num gpu list
            enable_chunked_prefill (bool): forwarded to ``run_disagg`` for
                each candidate combo; when False the chunked-prefill path
                is skipped (single-pass prefill).

        Returns:
            Optional[InferenceSummary]: the summary of the inference result, contains all the
                possible disagg config and perf that matches SLA.
        """

        if max_prefill_gpus is not None and max_prefill_gpus <= 0:
            raise ValueError(f"max_prefill_gpus must be a positive integer, got {max_prefill_gpus}")
        if max_decode_gpus is not None and max_decode_gpus <= 0:
            raise ValueError(f"max_decode_gpus must be a positive integer, got {max_decode_gpus}")

        # minor perf optimization: convert num_gpu_list to a set to speed up lookup
        num_gpu_set = set[int](num_gpu_list) if num_gpu_list else set()

        # Enable session-level caches for the heavy primitives ``run_disagg``
        # invokes (run_static, _compute_max_kv_slots, _run_chunked_context_phase).
        # These share results across (P, D) variations of the same
        # (parallel, bs) combo. Scoped to this picker call.
        self._static_run_cache = {}
        self._max_kv_slots_cache = {}
        self._chunked_prefill_cache = {}

        # ``run_disagg`` is deterministic in (parallel_sig, bs, num_worker,
        # enable_chunked_prefill) for a fixed (model_path, runtime_config).
        # The picker's outer loop iterates ~50 (ttft, tpot) constraint pairs
        # and re-evaluates the same (P, D, p_bs, d_bs) combos; cache results
        # so each unique combo only runs once.
        _run_disagg_cache: dict = {}

        def _cached_run_disagg(
            p_parallel_sig: tuple,
            p_bs: int,
            p_workers: int,
            d_parallel_sig: tuple,
            d_bs: int,
            d_workers: int,
            chunked: bool,
            prefill_worker: dict,
            decode_worker: dict,
        ) -> InferenceSummary:
            """Memoized ``run_disagg`` call keyed on the projection inputs
            that vary across the picker's loops. ``model_path`` and
            ``runtime_config`` are constant across calls and excluded from
            the key. ``prefill_worker`` / ``decode_worker`` are passed only
            so the model configs can be rebuilt on a cache miss.
            """
            key = (p_parallel_sig, p_bs, p_workers, d_parallel_sig, d_bs, d_workers, chunked)
            if key in _run_disagg_cache:
                return _run_disagg_cache[key]
            p_mc = _model_config_with_parallel(prefill_model_config, prefill_worker)
            d_mc = _model_config_with_parallel(decode_model_config, decode_worker)
            summary = self.run_disagg(
                model_path=model_path,
                runtime_config=runtime_config,
                prefill_model_config=p_mc,
                prefill_batch_size=p_bs,
                prefill_num_worker=p_workers,
                decode_model_config=d_mc,
                decode_batch_size=d_bs,
                decode_num_worker=d_workers,
                enable_chunked_prefill=chunked,
            )
            _run_disagg_cache[key] = summary
            return summary

        @functools.lru_cache(maxsize=8192)
        def _match_workers(
            prefill_throughput: float,
            prefill_gpus: int,
            decode_throughput: float,
            decode_gpus: int,
            rate_matching_prefill_degradation_factor: float,
            rate_matching_decode_degradation_factor: float,
        ) -> tuple[int, int]:
            """
            Match the prefill and decode workers, return the best prefill and decode num worker
            """
            prefill_opt_num_worker, decode_opt_num_worker = -1, -1
            throughput_per_gpu_max = 0
            for decode_num_worker in decode_num_worker_list:
                for prefill_num_worker in prefill_num_worker_list:
                    num_gpu = prefill_gpus * prefill_num_worker + decode_gpus * decode_num_worker

                    # if num_gpu_set is empty, we don't have any constraint on the number of gpus
                    # if num_gpu_set is not empty, we only consider the gpus that are in the set
                    if len(num_gpu_set) > 0 and num_gpu not in num_gpu_set:
                        continue

                    # per-pool GPU budget for hetero disagg
                    if max_prefill_gpus is not None and max_decode_gpus is not None:
                        if prefill_gpus * prefill_num_worker > max_prefill_gpus:
                            continue
                        if decode_gpus * decode_num_worker > max_decode_gpus:
                            continue

                    prefill_throughput_corrected = (
                        prefill_throughput * prefill_num_worker * rate_matching_prefill_degradation_factor
                    )
                    decode_throughput_corrected = (
                        decode_throughput * decode_num_worker * rate_matching_decode_degradation_factor
                    )

                    # criteria 1, try to make prefill_throughput larger than decode_throughput
                    # otherwise, decode bs cannot be achieved and decode throughput cannot be
                    # achieved as well.
                    # if prefill_throughput < decode_throughput:
                    #    continue

                    # criteria 2, try to make the throughput per gpu larger
                    throughput_per_gpu = min(prefill_throughput_corrected, decode_throughput_corrected) / num_gpu

                    if throughput_per_gpu > throughput_per_gpu_max:
                        throughput_per_gpu_max = throughput_per_gpu
                        prefill_opt_num_worker, decode_opt_num_worker = (
                            prefill_num_worker,
                            decode_num_worker,
                        )

            return prefill_opt_num_worker, decode_opt_num_worker

        def _find_best_result_under_constraints(
            ttft: float,
            tpot: float,
            prefill_summary_df: pd.DataFrame,
            decode_summary_df: pd.DataFrame,
            return_top_k: int,
            num_gpu_list: list[int] | None,
            rate_matching_prefill_degradation_factor: float,
            rate_matching_decode_degradation_factor: float,
            require_same_tp: bool = False,
        ) -> InferenceSummary:
            """
            Find the best result under constraints
            """

            # 1. we categorize the decode summary
            #    df into different categories based on parallelism (we can use the parallel key in
            #    the df). do the rate matching and sort the result by category - throughput.
            # 2. for prefill, follow two rules: high throughput, if at same level, choose the one
            #    with small batchsize. add one func for correct ttft (we have some formula,
            #    just leave it blank for now)
            # 3. prefill/decode correction are already applied to workers.
            #    Additional correction can be a degradation factor for the final result during the
            #   rate matching process.
            # 4. rate matching. The prefill throughput should be 1.x larger than the decode
            #    throughput.
            #    "1.x" is an empirical value. Default is 1.1.

            # SLA pre-filter: prefill rows whose single-pass static ttft is
            # already above the constraint can be dropped.  After
            # rate-matching, the combo will go through ``run_disagg`` which
            # applies the queue-model TTFT correction (and chunked-prefill,
            # eff-bs TPOT re-run, KV capacity check) -- the post-projection
            # numbers are checked again before a combo is kept.
            prefill_candidates = prefill_summary_df[prefill_summary_df["ttft"] < ttft].copy()
            if len(prefill_candidates) == 0:
                logger.debug(f"No prefill worker candidates found for ttft {ttft}ms.")
                return None
            prefill_candidates = (
                prefill_candidates.sort_values(by=["seq/s/gpu", "global_bs"], ascending=[False, True])
                .reset_index(drop=True)
                .head(MAX_PREFILL_WORKERS)
            )

            decode_candidates = decode_summary_df[
                (decode_summary_df["tpot"] < tpot * DECODE_FILTER_RATIO_MAX)
                & (decode_summary_df["tpot"] > tpot * DECODE_FILTER_RATIO_MIN)
            ].copy()
            if len(decode_candidates) == 0:
                logger.debug(f"No decode worker candidates found for tpot {tpot}ms.")
                return None

            all_category_results: list[dict] = []
            prefill_candidates_list = prefill_candidates.to_dict("records")

            for parallel_value, parallel_group in decode_candidates.groupby("parallel"):
                parallel_group_sorted = (
                    parallel_group.sort_values(by=["seq/s/gpu"], ascending=[False])
                    .reset_index(drop=True)
                    .head(MAX_DECODE_WORKERS_PER_CATEGORY)
                )

                decode_workers_list = parallel_group_sorted.to_dict("records")
                category_results: list[dict] = []
                for decode_worker in decode_workers_list:
                    decode_throughput = float(decode_worker["seq/s"])
                    decode_gpus = decode_worker["num_total_gpus"]
                    decode_bs = int(decode_worker.get("bs", 1))
                    for prefill_worker in prefill_candidates_list:
                        # For SGLang non-wideep disaggregated serving
                        # See: https://github.com/ai-dynamo/dynamo/issues/5870
                        if require_same_tp and prefill_worker["tp"] != decode_worker["tp"]:
                            continue
                        prefill_throughput = float(prefill_worker["seq/s"])
                        prefill_gpus = prefill_worker["num_total_gpus"]
                        prefill_num_worker, decode_num_worker = _match_workers(
                            prefill_throughput=prefill_throughput,
                            prefill_gpus=prefill_gpus,
                            decode_throughput=decode_throughput,
                            decode_gpus=decode_gpus,
                            rate_matching_prefill_degradation_factor=rate_matching_prefill_degradation_factor,
                            rate_matching_decode_degradation_factor=rate_matching_decode_degradation_factor,
                        )
                        if prefill_num_worker == -1 or decode_num_worker == -1:
                            continue

                        # Project this (P, D, p_bs, d_bs) combo through the
                        # SAME pipeline ``cli estimate`` uses (run_disagg):
                        # chunked-prefill TTFT, queueing correction (or
                        # legacy 1.8x when the queueing model is off),
                        # eff-bs decode TPOT re-run, KV-capacity check.
                        p_parallel_sig = (
                            int(prefill_worker["tp"]), int(prefill_worker["pp"]),
                            int(prefill_worker["dp"]), int(prefill_worker["moe_tp"]),
                            int(prefill_worker["moe_ep"]),
                        )
                        d_parallel_sig = (
                            int(decode_worker["tp"]), int(decode_worker["pp"]),
                            int(decode_worker["dp"]), int(decode_worker["moe_tp"]),
                            int(decode_worker["moe_ep"]),
                        )
                        p_bs = int(prefill_worker.get("bs", 1))
                        summary = _cached_run_disagg(
                            p_parallel_sig, p_bs, prefill_num_worker,
                            d_parallel_sig, decode_bs, decode_num_worker,
                            enable_chunked_prefill,
                            prefill_worker, decode_worker,
                        )
                        if summary.check_oom() or summary.check_kv_cache_oom():
                            continue
                        result_df = summary.get_summary_df()
                        if result_df is None or result_df.empty:
                            continue
                        disagg_dict = result_df.iloc[0].to_dict()

                        # Re-apply SLA filter on the projected numbers --
                        # queue-corrected TTFT can exceed the raw single-pass
                        # value, and eff-bs TPOT can be lower than the row's
                        # raw tpot.  The pre-filter only used raw values.
                        if disagg_dict.get("ttft", 0.0) >= ttft:
                            continue
                        if disagg_dict.get("tpot", 0.0) >= tpot:
                            continue

                        category_results.append(disagg_dict)

                if category_results:
                    # only return the best one for each category
                    best_result = max(category_results, key=lambda x: (x["tokens/s/gpu"], -x["num_total_gpus"]))
                    all_category_results.append(best_result)
                else:
                    logger.debug(f"No matched result for decode parallel {parallel_value}.")

            if not all_category_results:
                logger.debug("No disagg summary found after applying constraints.")
                return None

            disagg_summary_df = pd.DataFrame(all_category_results, columns=common.ColumnsDisagg).round(3)
            disagg_summary_df = (
                disagg_summary_df.sort_values(by=["tokens/s/gpu"], ascending=[False])
                .head(return_top_k)
                .reset_index(drop=True)
            )
            return disagg_summary_df
            # _find_best_result_under_constraints() ends here

        # start, get all possible p/d servers
        if decode_max_num_tokens < 1:
            logger.warning("decode_max_num_tokens is less than 1, set to 1")
            decode_max_num_tokens = 1
        decode_batch_size_list_default = (
            list(range(1, 16, 1)) + list(range(16, 32, 2)) + list(range(32, 128, 4)) + list(range(128, 512, 8)) + [512]
        )
        if decode_max_num_tokens > max(decode_batch_size_list_default):
            decode_batch_size_range = decode_batch_size_list_default + [decode_max_num_tokens]
        else:
            decode_batch_size_range = [i for i in decode_batch_size_list_default if i <= decode_max_num_tokens]

        if prefill_max_num_tokens < runtime_config.isl:
            logger.warning("prefill_max_num_tokens is less than runtime_config.isl, set to runtime_config.isl")
            prefill_max_num_tokens = runtime_config.isl

        max_prefill_batch_size = prefill_max_num_tokens // runtime_config.isl
        prefill_batch_size_range = range(1, max_prefill_batch_size + 1)

        # initialize disagg summary
        disagg_summary = InferenceSummary(runtime_config=runtime_config)
        disagg_summary_df = pd.DataFrame(columns=common.ColumnsDisagg)
        disagg_summary.set_summary_df(disagg_summary_df)

        # find prefill and decode workers
        # When queueing model is enabled, effective-BS re-run handles decode
        # unsaturation, so use scale=1.0 for decode candidates.
        # When disabled, use the configured scale (default 1.08).
        _use_qm = self._prefill_backend.use_queue_model
        _decode_scale = 1.0 if _use_qm else self._decode_latency_correction_scale
        prefill_summary_df = self.get_worker_candidates(
            model_path=model_path,
            model_config=prefill_model_config,
            parallel_config_list=prefill_parallel_config_list,
            b_list=prefill_batch_size_range,
            runtime_config=runtime_config,
            mode="static_ctx",
            latency_correction_scale=self._prefill_latency_correction_scale,
        )
        decode_summary_df = self.get_worker_candidates(
            model_path=model_path,
            model_config=decode_model_config,
            parallel_config_list=decode_parallel_config_list,
            b_list=decode_batch_size_range,
            runtime_config=runtime_config,
            mode="static_gen",
            latency_correction_scale=_decode_scale,
        )
        if len(prefill_summary_df) == 0 or len(decode_summary_df) == 0:
            logger.debug(f"No prefill or decode workers found for {model_path} with given configs.")
            return disagg_summary

        # ----- autoscale mode: pick P and D independently, no rate matching -----
        if autoscale:
            return self._pick_autoscale(
                prefill_summary_df=prefill_summary_df,
                decode_summary_df=decode_summary_df,
                runtime_config=runtime_config,
                disagg_summary=disagg_summary,
                target_tpot=target_tpot,
            )

        # find best result under constraints
        constraint_pairs: list[tuple[float, float]] = []
        if runtime_config.request_latency is not None and runtime_config.request_latency > 0:
            constraint_pairs = enumerate_ttft_tpot_constraints(
                runtime_config.osl,
                runtime_config.request_latency,
                runtime_config.ttft,
            )
            if not constraint_pairs:
                logger.debug(
                    "No ttft/tpot constraints derived for request_latency=%s in disagg optimization.",
                    runtime_config.request_latency,
                )
        else:
            tpot_values = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
            constraint_pairs = [(runtime_config.ttft, tpot) for tpot in tpot_values]

        for ttft_constraint, tpot_constraint in constraint_pairs:
            logger.debug(
                "Finding best result under constraints for ttft=%sms, tpot=%sms...",
                ttft_constraint,
                tpot_constraint,
            )
            filtered_disagg_summary_df = _find_best_result_under_constraints(
                ttft=ttft_constraint,
                tpot=tpot_constraint,
                prefill_summary_df=prefill_summary_df,
                decode_summary_df=decode_summary_df,
                return_top_k=5,
                num_gpu_list=num_gpu_list,
                rate_matching_prefill_degradation_factor=self._rate_matching_prefill_degradation_factor,
                rate_matching_decode_degradation_factor=self._rate_matching_decode_degradation_factor,
                require_same_tp=require_same_tp,
            )
            if filtered_disagg_summary_df is not None:
                disagg_summary_df = pd.concat(
                    [disagg_summary_df, filtered_disagg_summary_df], axis=0, ignore_index=True
                )
        if len(disagg_summary_df) == 0:
            logger.debug(f"No disagg result found for {model_path} with given constraints.")
            return disagg_summary

        disagg_summary_df = disagg_summary_df.drop_duplicates(ignore_index=True)

        # set final disagg summary
        disagg_summary.set_summary_df(disagg_summary_df)
        return disagg_summary

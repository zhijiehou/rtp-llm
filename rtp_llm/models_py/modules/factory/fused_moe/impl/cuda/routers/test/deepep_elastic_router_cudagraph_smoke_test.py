# type: ignore
"""Standalone CUDA Graph capture smoke test for DeepEpElasticRouter.

Validates the core value proposition of the A2 decode cudagraph path:

* ``(do_expand=False, do_cpu_sync=False)`` dispatch is capture-friendly:
  no D2H, no host-side ``cudaStreamSynchronize``, so
  ``torch.cuda.graph(g)`` captures the dispatch successfully.
* ``(do_expand=True,  do_cpu_sync=True)`` dispatch triggers an internal
  ``cudaStreamSynchronize`` (the price of populating
  ``handle.num_recv_tokens_per_expert_list`` as a Python list), which
  CUDA Graph capture forbids → capture MUST fail with a graph-state
  related ``RuntimeError`` (typically "capturing stream has unjoined
  work" from CUDA).

If the prefill capture ever starts succeeding (e.g. because DeepEPv2
silently changed the dispatch semantics) this test will fail loud — that
is by design.  Likewise, if the decode capture ever starts failing, the
entire premise of A2 (decode cudagraph) is broken and needs immediate
attention.

This test stays separate from ``deepep_elastic_router_test.py`` because
(a) it doesn't fit the prepare/finalize matrix shape, and (b) it
deliberately disables every debug / NaN-guard path that itself does
``.item()`` / ``.cpu()`` (those paths defeat capture for any layout, so
including them would obscure the capture-success signal).
"""
import logging
import os

import torch
import torch.distributed
import torch.multiprocessing as mp

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPMode,
    DeepEPWrapper,
    init_deepep_wrapper,
)
from rtp_llm.ops import MoeConfig, NcclCommConfig, ParallelismConfig, RuntimeConfig
from rtp_llm.test.utils.port_util import PortManager

NUM_TOKEN_PER_RANK = 64
HIDDEN_SIZE = 7168
TOPK = 8
NUM_EXPERTS = 128


def _set_env(do_expand: bool, do_cpu_sync: bool) -> None:
    os.environ["USE_DEEPEP_ELASTIC"] = "1"
    os.environ["DEEPEP_ELASTIC_DO_EXPAND"] = "1" if do_expand else "0"
    os.environ["DEEPEP_ELASTIC_DO_CPU_SYNC"] = "1" if do_cpu_sync else "0"
    os.environ.setdefault("DEEPEP_ELASTIC_ALLOW_HYBRID", "0")
    os.environ.setdefault("EP_DISABLE_GIN", "1")
    os.environ.setdefault("EP_SUPPRESS_NCCL_CHECK", "1")
    # Disable every host-sync debug path so capture-success signal isn't
    # masked by an unrelated D2H.
    os.environ["DEEPEP_ELASTIC_NAN_GUARD"] = "0"
    os.environ["DEEPEP_ELASTIC_DEBUG_ANOMALY"] = "0"
    os.environ["DEEPEP_ELASTIC_DUMP_HANDLE"] = "0"
    os.environ["DEEPEP_ELASTIC_CROSS_RANK_DIFF"] = "0"


def _clear_env() -> None:
    for key in (
        "USE_DEEPEP_ELASTIC",
        "DEEPEP_ELASTIC_DO_EXPAND",
        "DEEPEP_ELASTIC_DO_CPU_SYNC",
        "DEEPEP_ELASTIC_ALLOW_HYBRID",
        "EP_DISABLE_GIN",
        "EP_SUPPRESS_NCCL_CHECK",
        "DEEPEP_ELASTIC_NAN_GUARD",
        "DEEPEP_ELASTIC_DEBUG_ANOMALY",
        "DEEPEP_ELASTIC_DUMP_HANDLE",
        "DEEPEP_ELASTIC_CROSS_RANK_DIFF",
    ):
        os.environ.pop(key, None)


def _init_buffer(rank, world_size, do_expand, do_cpu_sync, nccl_port):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(world_size)
    )
    _set_env(do_expand, do_cpu_sync)

    model_config = ModelConfig()
    model_config.attn_config.head_num = 2
    model_config.attn_config.size_per_head = 128
    model_config.num_layers = 2
    model_config.max_seq_len = 2048
    model_config.vocab_size = 500000
    model_config.moe_k = TOPK
    model_config.expert_num = NUM_EXPERTS
    model_config.hidden_size = HIDDEN_SIZE

    base_port = nccl_port + 11
    nccl_comm_config = NcclCommConfig(
        nccl_ip="127.0.0.1",
        tp_nccl_port=base_port - 2,
        dp_tp_nccl_port=base_port - 10,
        ffn_tp_nccl_port=base_port - 5,
    )

    moe_config = MoeConfig()
    moe_config.use_deepep_low_latency = False
    moe_config.use_deepep_internode = False

    runtime_config = RuntimeConfig()
    runtime_config.max_generate_batch_size = NUM_TOKEN_PER_RANK
    moe_config.ll_num_max_token = NUM_TOKEN_PER_RANK

    parallelism_config = ParallelismConfig()
    parallelism_config.tp_size = 1
    parallelism_config.tp_rank = 0
    parallelism_config.ep_size = world_size
    parallelism_config.ep_rank = rank
    parallelism_config.dp_size = world_size
    parallelism_config.dp_rank = rank
    parallelism_config.local_rank = rank
    parallelism_config.world_size = world_size
    parallelism_config.world_rank = rank
    parallelism_config.local_world_size = world_size

    torch.cuda.set_device(rank)
    torch.set_default_device(f"cuda:{rank}")

    init_distributed_environment(
        parallelism_config=parallelism_config,
        nccl_comm_config=nccl_comm_config,
        nccl_init_port=base_port - 11,
        backend="nccl",
        timeout=60,
    )

    py_env = PyEnvConfigs()
    py_env.parallelism_config = parallelism_config
    py_env.moe_config = moe_config
    py_env.runtime_config = runtime_config
    py_env.concurrency_config.concurrency_limit = NUM_TOKEN_PER_RANK
    engine_config = EngineConfig(
        parallelism_config=py_env.parallelism_config,
        runtime_config=py_env.runtime_config,
        nccl_comm_config=nccl_comm_config,
        server_config=py_env.server_config,
        pd_sep_config=py_env.pd_separation_config,
        concurrency_config=py_env.concurrency_config,
        fmha_config=py_env.fmha_config,
        kv_cache_config=py_env.kv_cache_config,
        profiling_debug_logging_config=py_env.profiling_debug_logging_config,
        hw_kernel_config=py_env.py_hw_kernel_config,
        device_resource_config=py_env.device_resource_config,
        moe_config=py_env.moe_config,
        model_specific_config=py_env.model_specific_config,
        sp_config=py_env.sp_config,
        cache_store_config=py_env.cache_store_config,
        misc_config=py_env.misc_config.misc_config,
        arpc_config=py_env.arpc_config,
        grpc_config=py_env.grpc_config,
        load_config=py_env.load_config,
    )
    init_deepep_wrapper(engine_config, model_config)
    wrapper = DeepEPWrapper.get_instance(DeepEPWrapper._instance._config)
    assert wrapper.mode == DeepEPMode.ELASTIC
    return wrapper.elastic_buffer


def _do_dispatch(buffer, x, topk_ids, topk_weights, do_expand, do_cpu_sync):
    recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
        x=x,
        topk_idx=topk_ids,
        topk_weights=topk_weights,
        num_experts=NUM_EXPERTS,
        expert_alignment=1,
        do_expand=do_expand,
        do_cpu_sync=do_cpu_sync,
        async_with_compute_stream=True,
    )
    if event is not None and getattr(event, "event", None) is not None:
        event.current_stream_wait()
    return recv_x, recv_topk_idx, recv_topk_weights, handle


def _run_capture_attempt(
    rank,
    do_expand,
    do_cpu_sync,
    expect_capture_succeeds,
    world_size,
    nccl_port,
):
    buffer = _init_buffer(
        rank, world_size, do_expand, do_cpu_sync, nccl_port
    )
    try:
        torch.manual_seed(1234 + rank)
        torch.cuda.manual_seed(1234 + rank)

        x = torch.randn(
            (NUM_TOKEN_PER_RANK, HIDDEN_SIZE), dtype=torch.bfloat16
        ).cuda()
        topk_ids = (
            torch.rand(NUM_TOKEN_PER_RANK, NUM_EXPERTS)
            .topk(TOPK, dim=-1, largest=True)[1]
            .to(torch.int64)
            .cuda()
        )
        topk_weights = (
            torch.ones((NUM_TOKEN_PER_RANK, TOPK), dtype=torch.float32) / TOPK
        ).cuda()

        # Warmup: 3 dispatches outside capture so internal state stabilises.
        for _ in range(3):
            _do_dispatch(
                buffer, x, topk_ids, topk_weights, do_expand, do_cpu_sync
            )
            torch.cuda.synchronize()

        # One more warmup on the side stream (PyTorch's recommended pattern
        # for CUDA Graph capture).
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _do_dispatch(
                buffer, x, topk_ids, topk_weights, do_expand, do_cpu_sync
            )
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        capture_succeeded = False
        capture_error_msg = None
        try:
            with torch.cuda.graph(g, stream=side_stream):
                _do_dispatch(
                    buffer, x, topk_ids, topk_weights, do_expand, do_cpu_sync
                )
            capture_succeeded = True
        except Exception as e:
            capture_error_msg = f"{type(e).__name__}: {e}"
            try:
                del g
            except Exception:
                pass

        layout = f"do_expand={do_expand} do_cpu_sync={do_cpu_sync}"
        print(
            f"[CGCAPTURE] rank={rank} {layout} "
            f"capture_succeeded={capture_succeeded} "
            f"expected={expect_capture_succeeds} "
            f"err={capture_error_msg if not capture_succeeded else 'None'}",
            flush=True,
        )

        if expect_capture_succeeds:
            assert capture_succeeded, (
                f"[CGCAPTURE FAIL] rank={rank} {layout} capture should "
                f"succeed but failed: {capture_error_msg}"
            )
        else:
            assert not capture_succeeded, (
                f"[CGCAPTURE FAIL] rank={rank} {layout} capture should "
                f"have failed (cudaStreamSynchronize inside (True, True) "
                f"dispatch is forbidden inside CUDA Graph capture) but "
                f"succeeded — either the dispatch behavior changed or the "
                f"test environment is silently allowing forbidden sync ops."
            )
    finally:
        try:
            DeepEPWrapper.reset()
        except Exception:
            pass
        try:
            destroy_distributed_environment()
        except Exception:
            pass
        _clear_env()


def _spawn_capture(
    rank,
    do_expand,
    do_cpu_sync,
    expect_capture_succeeds,
    world_size,
    nccl_port,
):
    _run_capture_attempt(
        rank,
        do_expand,
        do_cpu_sync,
        expect_capture_succeeds,
        world_size,
        nccl_port,
    )


def _run_capture(do_expand, do_cpu_sync, expect_capture_succeeds):
    port_manager = PortManager()
    ports, locks = port_manager.get_consecutive_ports(1)
    nccl_port = ports[0]
    world_size = 2
    try:
        label = "decode (F,F)" if not do_cpu_sync else "prefill (T,T)"
        logging.info(
            "deepep_elastic_router_cudagraph_smoke_test: %s "
            "expect_capture_succeeds=%s world_size=%s",
            label,
            expect_capture_succeeds,
            world_size,
        )
        mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
            _spawn_capture,
            args=(
                do_expand,
                do_cpu_sync,
                expect_capture_succeeds,
                world_size,
                nccl_port,
            ),
            nprocs=world_size,
            join=True,
        )
    finally:
        for lock in locks:
            lock.__exit__(None, None, None)


def test_decode_cudagraph_capture_succeeds():
    """(do_expand=False, do_cpu_sync=False) dispatch must be capture-friendly."""
    _run_capture(do_expand=False, do_cpu_sync=False, expect_capture_succeeds=True)


def test_prefill_cudagraph_capture_fails():
    """(do_expand=True, do_cpu_sync=True) dispatch must fail capture.

    The (T,T) path internally does ``cudaStreamSynchronize`` so it can
    populate ``handle.num_recv_tokens_per_expert_list`` as a Python list.
    CUDA Graph capture forbids host-side sync ops. If this assertion ever
    starts failing (capture unexpectedly succeeds for prefill), DeepEPv2
    has changed dispatch semantics and the entire (True, True) → (False,
    False) split needs re-evaluation.
    """
    _run_capture(do_expand=True, do_cpu_sync=True, expect_capture_succeeds=False)


if __name__ == "__main__":
    test_decode_cudagraph_capture_succeeds()
    test_prefill_cudagraph_capture_fails()
    print("[CGCAPTURE] ALL CHECKS PASSED")

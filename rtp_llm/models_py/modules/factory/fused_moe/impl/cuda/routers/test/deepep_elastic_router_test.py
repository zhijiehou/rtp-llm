# type: ignore
"""Unit tests for ``DeepEpElasticRouter`` (USE_DEEPEP_ELASTIC=1 path).

Only one layout is supported post-fix:

* ``do_expand=True, do_cpu_sync=True`` → 2D Contiguous ``[ΣN_e, hidden]``.

The 3D Batched (``do_cpu_sync=False``) path was removed because DeepEPv2
leaves ``recv_x`` compact 2D at offset 0 in that mode, so a downstream
``.view(E_local, M_max, hidden)`` would read uninitialised memory. See
``集成方案/BLOCKERS.md`` and the ship-2D decision note.

The matrix is therefore:

* ``bf16`` (no quant) and ``fp8_per_block``
* layout is fixed (``do_expand=True``, ``do_cpu_sync=True``); both env
  vars are still parsed but only the supported combination runs the
  positive matrix.

``USE_DEEPEP_LOW_LATENCY`` is *not* consulted on the elastic path.

Modeled on ``deepep_low_latency_router_test.py``: spawn world_size
workers, initialise distributed + DeepEPv2 ``ElasticBuffer`` singleton,
then exercise ``router.prepare`` → fake-execute (identity) →
``router.finalize`` and verify the combined output approximates the
original input within the appropriate tolerance.

Also includes two negative cases asserting that
``DEEPEP_ELASTIC_DO_EXPAND=0`` and ``DEEPEP_ELASTIC_DO_CPU_SYNC=0`` are
both rejected at router construction.
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
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_elastic_router import (
    DeepEpElasticRouter,
)
from rtp_llm.ops import MoeConfig, NcclCommConfig, ParallelismConfig, RuntimeConfig
from rtp_llm.test.utils.numeric_util import per_token_cast_back
from rtp_llm.test.utils.port_util import PortManager

NUM_TOKEN_PER_RANK = 64
HIDDEN_SIZE = 7168
TOPK = 8
NUM_EXPERTS = 128


def _set_elastic_env(do_expand: bool, do_cpu_sync: bool) -> None:
    os.environ["USE_DEEPEP_ELASTIC"] = "1"
    os.environ["DEEPEP_ELASTIC_DO_EXPAND"] = "1" if do_expand else "0"
    os.environ["DEEPEP_ELASTIC_DO_CPU_SYNC"] = "1" if do_cpu_sync else "0"


def _clear_elastic_env() -> None:
    for key in (
        "USE_DEEPEP_ELASTIC",
        "DEEPEP_ELASTIC_DO_EXPAND",
        "DEEPEP_ELASTIC_DO_CPU_SYNC",
    ):
        os.environ.pop(key, None)


def _init_router(
    rank: int,
    use_fp8: bool,
    do_expand: bool,
    do_cpu_sync: bool,
    parallelism_config: ParallelismConfig,
    nccl_port: int,
):
    world_size = parallelism_config.world_size
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world_size))
    os.environ["ACCL_DISPATCH_NUM_WARP_GROUPS"] = "4"
    os.environ["ACCL_COMBINE_NUM_WARP_GROUPS"] = "4"
    os.environ["ACCL_LOW_LATENCY_OPTIMIZE"] = "1"
    os.environ["ACCL_TOPO_FIX"] = "1"
    os.environ["ACCL_LOAD_BALANCE"] = "1"
    # Required gates for DeepEpElasticRouter — must be set BEFORE
    # init_deepep_wrapper so the wrapper picks the ELASTIC mode and the
    # elastic-layout flags propagate into DeepepWrapperConfig.
    _set_elastic_env(do_expand=do_expand, do_cpu_sync=do_cpu_sync)

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
    # USE_DEEPEP_LOW_LATENCY is intentionally NOT consulted by the elastic
    # path post-fix — leave it False to assert the elastic router stays
    # decoupled from the legacy gate.
    moe_config.use_deepep_low_latency = False
    moe_config.use_deepep_internode = False

    runtime_config = RuntimeConfig()
    runtime_config.max_generate_batch_size = NUM_TOKEN_PER_RANK
    moe_config.ll_num_max_token = NUM_TOKEN_PER_RANK

    config = MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )

    torch.cuda.set_device(parallelism_config.local_rank)
    torch.set_default_device(f"cuda:{parallelism_config.local_rank}")
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

    wrapper = DeepEPWrapper.get_instance(
        DeepEPWrapper._instance._config  # type: ignore[union-attr]
    )
    assert wrapper.mode == DeepEPMode.ELASTIC, (
        f"expected DeepEPMode.ELASTIC, got {wrapper.mode}; "
        "is USE_DEEPEP_ELASTIC=1 exported before init_deepep_wrapper?"
    )

    router = DeepEpElasticRouter(
        config,
        FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn if use_fp8 else None,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=[128, 128] if use_fp8 else None,
        ),
    )
    return config, router


def _destroy_router(router: DeepEpElasticRouter):
    del router
    DeepEPWrapper.reset()
    destroy_distributed_environment()
    # Leave the env clean for the next sub-test invocation in this process.
    _clear_elastic_env()


def _run_one(
    rank: int,
    use_fp8: bool,
    parallelism_config: ParallelismConfig,
    nccl_port: int,
):
    # Only do_expand=True, do_cpu_sync=True (2D Contiguous) is supported.
    config, router = _init_router(
        rank,
        use_fp8,
        do_expand=True,
        do_cpu_sync=True,
        parallelism_config=parallelism_config,
        nccl_port=nccl_port,
    )

    # Sanity: the router latched the env-driven flags.
    assert router._do_expand is True, (
        f"router._do_expand should be True, got {router._do_expand}"
    )
    assert router._do_cpu_sync is True, (
        f"router._do_cpu_sync should be True, got {router._do_cpu_sync}"
    )

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)

    num_token_per_rank = NUM_TOKEN_PER_RANK
    hidden_size = HIDDEN_SIZE
    num_experts = NUM_EXPERTS
    num_topk = TOPK

    hidden_states = torch.randn(
        (num_token_per_rank, hidden_size), dtype=torch.bfloat16
    ).cuda()
    topk_ids = torch.rand(num_token_per_rank, num_experts).topk(
        num_topk, dim=-1, largest=True
    )[1].cuda()
    topk_weights = (
        torch.ones((num_token_per_rank, num_topk), dtype=torch.float32) / num_topk
    ).cuda()

    # ---- prepare ----
    payload = router.prepare(
        hidden_states,
        None,
        None,
        topk_weights,
        topk_ids,
    )
    expert_x = payload.expert_x
    assert expert_x is not None, "prepare() returned empty expert_x"
    assert payload.expert_tokens_meta is not None

    # 2D Contiguous: tight [ΣN_e, hidden] layout — expert_num_tokens_cpu
    # must be available so executors can slice per-expert ranges.
    assert (
        payload.expert_tokens_meta.expert_num_tokens_cpu is not None
    ), "2D Contiguous prepare() must populate expert_num_tokens_cpu"
    assert expert_x.dim() == 2, (
        f"2D Contiguous expert_x should have rank 2, got shape {expert_x.shape}"
    )

    # ---- fake expert execution: identity (dequant if needed) ----
    if use_fp8 and payload.expert_x_scale is not None:
        # 2D Contiguous output: [ΣN_e, K] flat
        fused = per_token_cast_back(expert_x, payload.expert_x_scale)
    else:
        fused = (
            expert_x.to(torch.bfloat16)
            if expert_x.dtype != torch.bfloat16
            else expert_x
        )

    # ---- finalize ----
    extra_finalize_args = {"original_num_tokens": num_token_per_rank}
    combined_x = router.finalize(
        CombineForwardPayload(fused_expert_output=fused),
        payload.expert_topk_weights,
        payload.expert_topk_ids,
        False,
        extra_finalize_args,
    )

    assert combined_x.shape == hidden_states.shape, (
        f"shape mismatch after finalize: got {combined_x.shape}, "
        f"expected {hidden_states.shape}"
    )
    assert combined_x.dtype == hidden_states.dtype, (
        f"dtype mismatch: got {combined_x.dtype}, expected {hidden_states.dtype}"
    )
    # Identity expert with uniform weights → combined ≈ original on the
    # hottest dimensions. Loose tolerance — this test gates wiring, not
    # numerical fidelity (that lives in the deepep_test.py kernel test).
    torch.testing.assert_close(
        combined_x[:, :128],
        hidden_states[:, :128],
        atol=2e-1,
        rtol=2e-1,
    )

    _destroy_router(router)


def _run_negative(
    rank: int,
    do_expand: bool,
    do_cpu_sync: bool,
    parallelism_config: ParallelismConfig,
    nccl_port: int,
):
    """Generic negative case: expects router construction to AssertionError."""
    raised = False
    try:
        _init_router(
            rank,
            use_fp8=False,
            do_expand=do_expand,
            do_cpu_sync=do_cpu_sync,
            parallelism_config=parallelism_config,
            nccl_port=nccl_port,
        )
    except AssertionError:
        raised = True
    finally:
        # Best-effort cleanup if init partially succeeded.
        try:
            DeepEPWrapper.reset()
        except Exception:
            pass
        try:
            destroy_distributed_environment()
        except Exception:
            pass
        _clear_elastic_env()

    assert raised, (
        f"do_expand={do_expand}, do_cpu_sync={do_cpu_sync} should fail-closed "
        "in DeepEpElasticRouter.__init__, but no AssertionError was raised"
    )


def _spawn_wrapper(
    rank: int,
    use_fp8: bool,
    world_size: int,
    test_tp_size: int,
    nccl_port: int,
):
    dp_size = world_size // test_tp_size
    ep_size = world_size

    parallelism_config = ParallelismConfig()
    parallelism_config.tp_size = test_tp_size
    parallelism_config.tp_rank = rank % test_tp_size
    parallelism_config.ep_size = ep_size
    parallelism_config.ep_rank = rank % ep_size
    parallelism_config.dp_size = dp_size
    parallelism_config.dp_rank = rank // test_tp_size
    parallelism_config.local_rank = rank
    parallelism_config.world_size = world_size
    parallelism_config.world_rank = rank
    parallelism_config.local_world_size = world_size
    _run_one(rank, use_fp8, parallelism_config, nccl_port)


def _spawn_negative_wrapper(
    rank: int,
    do_expand: bool,
    do_cpu_sync: bool,
    world_size: int,
    test_tp_size: int,
    nccl_port: int,
):
    dp_size = world_size // test_tp_size
    ep_size = world_size

    parallelism_config = ParallelismConfig()
    parallelism_config.tp_size = test_tp_size
    parallelism_config.tp_rank = rank % test_tp_size
    parallelism_config.ep_size = ep_size
    parallelism_config.ep_rank = rank % ep_size
    parallelism_config.dp_size = dp_size
    parallelism_config.dp_rank = rank // test_tp_size
    parallelism_config.local_rank = rank
    parallelism_config.world_size = world_size
    parallelism_config.world_rank = rank
    parallelism_config.local_world_size = world_size
    _run_negative(rank, do_expand, do_cpu_sync, parallelism_config, nccl_port)


def test_deepep_elastic_router():
    port_manager = PortManager()
    ports, locks = port_manager.get_consecutive_ports(1)
    nccl_port = ports[0]

    world_size = 2
    test_tp_sizes = [1, 2]

    # bf16 + fp8 × 2D Contiguous only. do_cpu_sync=False (3D Batched)
    # is fail-closed at router __init__ (see BLOCKERS.md: DeepEPv2 recv_x
    # is compact 2D in that mode, not per-expert padded), so the positive
    # matrix is restricted to the supported layout.
    try:
        for use_fp8 in (True, False):
            for test_tp_size in test_tp_sizes:
                logging.info(
                    "test_deepep_elastic_router: use_fp8=%s layout=2D-Contiguous "
                    "test_tp_size=%s world_size=%s",
                    use_fp8,
                    test_tp_size,
                    world_size,
                )
                mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
                    _spawn_wrapper,
                    args=(
                        use_fp8,
                        world_size,
                        test_tp_size,
                        nccl_port,
                    ),
                    nprocs=world_size,
                    join=True,
                )
    finally:
        for lock in locks:
            lock.__exit__(None, None, None)


def _run_reject_test(do_expand: bool, do_cpu_sync: bool, label: str):
    port_manager = PortManager()
    ports, locks = port_manager.get_consecutive_ports(1)
    nccl_port = ports[0]

    world_size = 2
    test_tp_size = 2
    try:
        logging.info(
            "test_deepep_elastic_router_rejects_%s: world_size=%s test_tp_size=%s",
            label,
            world_size,
            test_tp_size,
        )
        mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
            _spawn_negative_wrapper,
            args=(do_expand, do_cpu_sync, world_size, test_tp_size, nccl_port),
            nprocs=world_size,
            join=True,
        )
    finally:
        for lock in locks:
            lock.__exit__(None, None, None)


def test_deepep_elastic_router_rejects_do_expand_false():
    """DEEPEP_ELASTIC_DO_EXPAND=0 must be rejected at router construction."""
    _run_reject_test(do_expand=False, do_cpu_sync=True, label="do_expand_false")


def test_deepep_elastic_router_rejects_do_cpu_sync_false():
    """DEEPEP_ELASTIC_DO_CPU_SYNC=0 must be rejected at router construction.

    3D Batched layout was removed because DeepEPv2 leaves recv_x compact 2D
    at offset 0 when do_cpu_sync=False — a downstream
    .view(E_local, M_max, hidden) would read uninitialised memory.
    """
    _run_reject_test(do_expand=True, do_cpu_sync=False, label="do_cpu_sync_false")


if __name__ == "__main__":
    test_deepep_elastic_router()
    test_deepep_elastic_router_rejects_do_expand_false()
    test_deepep_elastic_router_rejects_do_cpu_sync_false()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import signal
import time

from vllm import AsyncEngineArgs, envs
from vllm.entrypoints.launchers.launcher import setup_server
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.utils import launch_core_engines
from vllm.v1.executor import Executor
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus
from vllm.v1.utils import (
    APIServerProcessManager,
    RustFrontendProcessManager,
    wait_for_completion_or_failure,
)

logger = init_logger(__name__)


def run_multi_api_server(args: argparse.Namespace):
    assert not args.headless
    rust_frontend_path = (
        envs.VLLM_RUST_FRONTEND_PATH if envs.VLLM_USE_RUST_FRONTEND else None
    )
    num_api_servers: int = args.api_server_count
    assert num_api_servers > 0

    if rust_frontend_path and num_api_servers > 1:
        raise ValueError(
            "VLLM_RUST_FRONTEND_PATH does not support api_server_count > 1"
        )

    if num_api_servers > 1:
        setup_multiprocess_prometheus()

    shutdown_requested = False

    # Catch SIGTERM and SIGINT to allow graceful shutdown.
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.debug("Received %d signal.", signum)
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    listen_address, sock = setup_server(args, reuse_port=num_api_servers > 1)

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args._api_process_count = num_api_servers
    engine_args._api_process_rank = -1

    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    if num_api_servers > 1 and envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
        raise ValueError(
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING cannot be used with api_server_count > 1"
        )

    executor_class = Executor.get_class(vllm_config)
    log_stats = not engine_args.disable_log_stats

    parallel_config = vllm_config.parallel_config
    dp_rank = parallel_config.data_parallel_rank
    assert parallel_config.local_engines_only or dp_rank == 0

    api_server_manager: APIServerProcessManager | RustFrontendProcessManager | None = (
        None
    )

    from vllm.v1.engine.utils import get_engine_zmq_addresses

    # Defer port allocation to the child's bind() to avoid TOCTOU, except
    # for Rust front-end and Ray DP, which can't see the post-bind rebind
    # (CLI-arg subprocess / pickled-into-actor snapshot respectively) and
    # so pre-allocate driver-side -- reintroducing the original race only
    # there.
    is_ray_dp = parallel_config.data_parallel_backend == "ray"
    addresses = get_engine_zmq_addresses(
        vllm_config,
        num_api_servers,
        defer_api_server_ports=not (rust_frontend_path or is_ray_dp),
    )

    with launch_core_engines(
        vllm_config, executor_class, log_stats, addresses
    ) as engine_launch:
        local_engine_manager = engine_launch.engine_manager
        coordinator = engine_launch.coordinator
        addresses = engine_launch.addresses
        stats_update_address = (
            coordinator.get_stats_publish_address() if coordinator else None
        )

        if rust_frontend_path:
            if parallel_config.local_engines_only:
                expected_engine_start_index = parallel_config.data_parallel_rank
                expected_engine_count = parallel_config.data_parallel_size_local
            else:
                expected_engine_start_index = 0
                expected_engine_count = parallel_config.data_parallel_size
            # Start rust front-end process.
            api_server_manager = RustFrontendProcessManager(
                binary_path=rust_frontend_path,
                sock=sock,
                args=args,
                input_address=addresses.inputs[0],
                output_address=addresses.outputs[0],
                engine_start_index=expected_engine_start_index,
                engine_count=expected_engine_count,
                data_parallel_size=parallel_config.data_parallel_size,
                stats_update_address=stats_update_address,
            )
        else:
            # Start API server(s).
            api_server_manager = APIServerProcessManager(
                listen_address=listen_address,
                sock=sock,
                args=args,
                num_servers=num_api_servers,
                input_addresses=addresses.inputs,
                output_addresses=addresses.outputs,
                stats_update_address=stats_update_address,
                tensor_queue=engine_launch.tensor_queue,
            )

            if not is_ray_dp:
                # Forward each child's bound endpoints to the engine handshake
                # (runs on ``with`` exit). Skipped for Ray DP, where addresses
                # are pre-allocated above and Ray actors already hold them.
                actual_inputs, actual_outputs = (
                    api_server_manager.gather_actual_addresses()
                )
                addresses.inputs = actual_inputs
                addresses.outputs = actual_outputs

        # Set frontend processes to watch during engine startup.
        # If any of these processes exit before the engines are up, the engine startup
        # will be aborted with an error.
        engine_launch.watched_frontend_processes = api_server_manager.processes

    # Wait for API servers.
    try:
        wait_for_completion_or_failure(
            api_server_manager=api_server_manager,
            engine_manager=local_engine_manager,
            coordinator=coordinator,
        )
    finally:
        timeout = shutdown_by = None
        if shutdown_requested:
            timeout = vllm_config.shutdown_timeout
            shutdown_by = time.monotonic() + timeout
            logger.info("Waiting up to %d seconds for processes to exit", timeout)

        def to_timeout(deadline: float | None) -> float | None:
            return (
                deadline if deadline is None else max(deadline - time.monotonic(), 0.0)
            )

        api_server_manager.shutdown(timeout=timeout)
        if local_engine_manager:
            local_engine_manager.shutdown(timeout=to_timeout(shutdown_by))
        if coordinator:
            coordinator.shutdown(timeout=to_timeout(shutdown_by))

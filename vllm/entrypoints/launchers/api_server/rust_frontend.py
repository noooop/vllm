# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import signal
import time
import weakref
from typing import Any

from vllm import AsyncEngineArgs, envs
from vllm.entrypoints.launchers.launcher import setup_server
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.utils import launch_core_engines
from vllm.v1.executor import Executor
from vllm.v1.utils import (
    _shutdown_subprocesses,
    _SubprocessWrapper,
    wait_for_completion_or_failure,
)

logger = init_logger(__name__)


class RustFrontendProcessManager:
    """Manages a single Rust frontend subprocess.

    Launches the Rust vllm-rs binary in 'frontend' mode, passing the
    listening socket fd and ZMQ transport addresses. Provides the same
    interface as APIServerProcessManager for process monitoring.
    """

    def __init__(
        self,
        binary_path: str,
        sock: Any,
        args: argparse.Namespace,
        input_address: str,
        output_address: str,
        engine_start_index: int,
        engine_count: int,
        data_parallel_size: int,
        stats_update_address: str | None = None,
    ):
        import os
        import subprocess

        fd = sock.fileno()
        os.set_inheritable(fd, True)

        cmd = [
            binary_path,
            "frontend",
            "--listen-fd",
            str(fd),
            "--input-address",
            input_address,
            "--output-address",
            output_address,
            "--engine-start-index",
            str(engine_start_index),
            "--engine-count",
            str(engine_count),
            "--data-parallel-size",
            str(data_parallel_size),
        ]
        if stats_update_address is not None:
            cmd.extend(["--coordinator-address", stats_update_address])
        from vllm.entrypoints.serve.utils.api_utils import jsonify_non_default_args

        args_dict = jsonify_non_default_args(
            args,
            exclude={
                "api_server_count",
                # Python passes the bootstrapped engine range explicitly.
                "data_parallel_rank",
                "data_parallel_external_lb",
                "data_parallel_hybrid_lb",
            },
        )
        # The Rust `frontend` subcommand parses --args-json via serde_json,
        # which bypasses clap and therefore ignores any `#[arg(env = ...)]`
        # declarations on SharedRuntimeArgs fields. Forward the env-driven
        # values explicitly so VLLM_ENGINE_READY_TIMEOUT_S and
        # VLLM_HTTP_TIMEOUT_KEEP_ALIVE behave the same on both Python and Rust
        # frontends.
        args_dict["engine_ready_timeout_secs"] = envs.VLLM_ENGINE_READY_TIMEOUT_S
        args_dict["http_timeout_keep_alive"] = envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE
        args_json = json.dumps(args_dict, sort_keys=True)
        cmd.extend(["--args-json", args_json])

        # The subprocess needs the real values, but the log must not carry
        # credentials such as api_key or hf_token.
        from vllm.entrypoints.serve.utils.api_utils import redact_sensitive_args

        redacted_json = json.dumps(redact_sensitive_args(args_dict), sort_keys=True)
        logger.info("Launching Rust frontend: %s", " ".join(cmd[:-1] + [redacted_json]))
        self._proc = subprocess.Popen(cmd, pass_fds=(fd,))

        # Create a process wrapper with a sentinel fd for monitoring
        self.processes: list[_SubprocessWrapper] = [
            _SubprocessWrapper(self._proc, "RustFrontend")
        ]

        self._finalizer = weakref.finalize(self, _shutdown_subprocesses, self.processes)

    def shutdown(self, timeout: float | None = None) -> None:
        if self._finalizer.detach() is not None:
            _shutdown_subprocesses(self.processes, timeout=timeout)


def run_rust_frontend(args: argparse.Namespace):
    assert not args.headless
    rust_frontend_path = (
        envs.VLLM_RUST_FRONTEND_PATH if envs.VLLM_USE_RUST_FRONTEND else None
    )

    # rust_frontend only supports num_api_servers = 1 for now.
    num_api_servers = 1
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

    listen_address, sock = setup_server(args, reuse_port=False)

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args._api_process_count = num_api_servers
    engine_args._api_process_rank = -1

    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    executor_class = Executor.get_class(vllm_config)
    log_stats = not engine_args.disable_log_stats

    parallel_config = vllm_config.parallel_config
    dp_rank = parallel_config.data_parallel_rank
    assert parallel_config.local_engines_only or dp_rank == 0

    api_server_manager: RustFrontendProcessManager | None = None

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

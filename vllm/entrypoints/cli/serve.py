# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

import uvloop

import vllm.envs as envs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.launchers.api_server import (
    run_headless,
    run_multi_api_server,
    run_rust_frontend,
    run_single_api_server,
)
from vllm.entrypoints.launchers.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.launchers.dp_supervisor import run_dp_supervisor
from vllm.entrypoints.launchers.utils.setup_utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

DESCRIPTION = """Launch a local OpenAI-compatible API server to serve LLM
completions via HTTP. Defaults to Qwen/Qwen3-0.6B if no model is specified.

Search by using: `--help=<ConfigGroup>` to explore options by section (e.g.,
--help=ModelConfig, --help=Frontend)
  Use `--help=all` to show all available flags at once.
"""


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI."""

    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        if getattr(args, "grpc", False):
            from vllm.entrypoints.launchers.grpc_server import serve_grpc

            uvloop.run(serve_grpc(args))
            return

        rust_frontend_path = (
            envs.VLLM_RUST_FRONTEND_PATH if envs.VLLM_USE_RUST_FRONTEND else None
        )

        if args.headless:
            if args.api_server_count is not None and args.api_server_count > 0:
                raise ValueError(
                    f"--api-server-count={args.api_server_count} cannot be "
                    "used with --headless (no API servers are started in "
                    "headless mode)."
                )
            # Default to 0 in headless mode (no API servers)
            args.api_server_count = 0

        # Detect LB mode for defaulting api_server_count.
        # Multi-port: --data-parallel-multi-port-external-lb
        # External LB: --data-parallel-external-lb or --data-parallel-rank
        # Hybrid LB: --data-parallel-hybrid-lb or --data-parallel-start-rank
        is_external_lb = (
            args.data_parallel_external_lb or args.data_parallel_rank is not None
        )

        # If --data_parallel_multi_port_external_lb and --data_parallel_hybrid_lb
        # are unset, default to hybrid if --data-parallel-start-rank is set
        is_hybrid_lb = is_multi_port = False
        if (
            not args.data_parallel_hybrid_lb
            and not args.data_parallel_multi_port_external_lb
        ):
            is_hybrid_lb = args.data_parallel_start_rank is not None
        else:
            is_hybrid_lb = args.data_parallel_hybrid_lb
            is_multi_port = args.data_parallel_multi_port_external_lb

        if sum([is_multi_port, is_external_lb, is_hybrid_lb]) > 1:
            raise ValueError(
                "Cannot use more than one data parallel load balancing mode. "
                "Choose one of: --data-parallel-multi-port-external-lb, "
                "--data-parallel-external-lb (or --data-parallel-rank), "
                "--data-parallel-hybrid-lb (or --data-parallel-start-rank)."
            )

        # Default api_server_count if not explicitly set.
        # - Multi-port: 1 (supervisor spawns one server per local DP rank)
        # - Rust frontend: 1 (not applicable as it's multithreaded)
        # - External LB: 1 (external LB handles distribution)
        # - Hybrid LB: Use local DP size (internal LB for local ranks only)
        # - Internal LB: Use full DP size
        if args.api_server_count is None:
            if is_multi_port or is_external_lb or rust_frontend_path:
                args.api_server_count = 1
            elif is_hybrid_lb:
                args.api_server_count = args.data_parallel_size_local or 1
                if args.api_server_count > 1:
                    logger.info(
                        "Defaulting api_server_count to data_parallel_size_local "
                        "(%d) for hybrid LB mode.",
                        args.api_server_count,
                    )
            else:
                args.api_server_count = args.data_parallel_size
                if args.api_server_count > 1:
                    logger.info(
                        "Defaulting api_server_count to data_parallel_size (%d).",
                        args.api_server_count,
                    )
        elif rust_frontend_path and args.api_server_count > 1:
            logger.warning(
                "Ignoring --api-server-count=%d when using rust front-end process",
                args.api_server_count,
            )
            args.api_server_count = 1

        # Elastic EP currently only supports running with at most one API server.
        if getattr(args, "enable_elastic_ep", False) and args.api_server_count > 1:
            logger.warning(
                "Elastic EP only supports running with with at most one API server. "
                "Capping api_server_count from %d to 1.",
                args.api_server_count,
            )
            args.api_server_count = 1
        if rust_frontend_path:
            run_rust_frontend(args)
        elif is_multi_port:
            run_dp_supervisor(args)
        elif args.api_server_count < 1:
            run_headless(args)
        elif args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            args.api_server_count = None
            run_single_api_server(args)

    def validate(self, args: argparse.Namespace) -> None:
        validate_parsed_serve_args(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        serve_parser = subparsers.add_parser(
            self.name,
            help="Launch a local OpenAI-compatible API server to serve LLM "
            "completions via HTTP.",
            description=DESCRIPTION,
            usage="vllm serve [model_tag] [options]",
        )

        serve_parser = make_arg_parser(serve_parser)
        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import dataclasses
import multiprocessing
import os
import signal
from logging import Logger
from multiprocessing import forkserver
from string import Template
from typing import Any

from fastapi import FastAPI

from vllm import EngineArgs, envs
from vllm.logger import current_formatter_type, init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)


VLLM_SUBCMD_PARSER_EPILOG = (
    "For full list:            vllm {subcmd} --help=all\n"
    "For a section:            vllm {subcmd} --help=ModelConfig    (case-insensitive)\n"  # noqa: E501
    "For a flag:               vllm {subcmd} --help=max-model-len  (_ or - accepted)\n"  # noqa: E501
    "Documentation:            https://docs.vllm.ai\n"
)


def log_version_and_model(lgr: Logger, version: str, model_name: str) -> None:
    formatter = None if envs.VLLM_DISABLE_LOG_LOGO else current_formatter_type(lgr)
    if formatter is None:
        message = "vLLM server version %s, serving model %s"
    else:
        logo_template = Template(
            "\n       ${w}█     █     █▄   ▄█${r}\n"
            " ${o}▄▄${r} ${b}▄█${r} ${w}█     █     █ ▀▄▀ █${r}  version ${w}%s${r}\n"
            "  ${o}█${r}${b}▄█▀${r} ${w}█     █     █     █${r}  model   ${w}%s${r}\n"
            "   ${b}▀▀${r}  ${w}▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀${r}\n"
        )
        colors = {
            "w": "\033[1m",  # bold, default foreground
            "o": "\033[93m",  # orange
            "b": "\033[94m",  # blue
            "r": "\033[0m",  # reset
        }
        if formatter != "color":
            # monochrome logo (no ansi escape codes)
            colors = dict.fromkeys(colors, "")

        message = logo_template.substitute(colors)

    lgr.info(message, version, model_name)


def cli_env_setup():
    # The safest multiprocessing method is `spawn`, as the default `fork` method
    # is not compatible with some accelerators. The default method will be
    # changing in future versions of Python, so we should use it explicitly when
    # possible.
    #
    # We only set it here in the CLI entrypoint, because changing to `spawn`
    # could break some existing code using vLLM as a library. `spawn` will cause
    # unexpected behavior if the code is not protected by
    # `if __name__ == "__main__":`.
    #
    # References:
    # - https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # - https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    # - https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    # - https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders
    if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
        logger.debug("Setting VLLM_WORKER_MULTIPROC_METHOD to 'spawn'")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def setup_forkserver():
    """Pre-import heavy modules in the forkserver process (idempotent)."""
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        # The executor is expected to be mp.
        # Pre-import heavy modules in the forkserver process
        logger.debug("Setup forkserver with pre-imports")
        multiprocessing.set_start_method("forkserver")
        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
        forkserver.ensure_running()
        logger.debug("Forkserver setup complete!")


def setup_interrupt_handler():
    """Install a pre-uvicorn SIGINT/SIGTERM handler that raises KeyboardInterrupt.

    This is a one-shot, process-level side effect.  It is replaced by
    ``loop.add_signal_handler`` once ``_install_signal_handlers`` runs
    (after uvicorn has been built), so it only matters for the window
    between socket bind and ``server.serve()`` starting.
    """

    def _interrupt_init(*_) -> None:
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGINT, _interrupt_init)
    signal.signal(signal.SIGTERM, _interrupt_init)


def init_parser_plugin(args: argparse.Namespace):
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)


def log_routes(app: FastAPI) -> None:
    """Log the routes exposed by ``app`` for operator convenience."""
    logger.info("Available routes are:")
    for route in app.routes:
        path = getattr(route, "path", None)
        if path is None:
            continue

        methods = getattr(route, "methods", None)
        if methods:
            logger.info("Route: %s, Methods: %s", path, ", ".join(sorted(methods)))
            continue

        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None:
            name = getattr(endpoint, "__name__", type(endpoint).__name__)
            logger.info("Route: %s, Endpoint: %s", path, name)


def get_non_default_args(args: argparse.Namespace | EngineArgs) -> dict[str, Any]:
    from vllm.entrypoints.launchers.cli_args import make_arg_parser

    non_default_args = {}

    # Handle Namespace
    if isinstance(args, argparse.Namespace):
        parser = make_arg_parser(FlexibleArgumentParser())
        for arg, default in vars(parser.parse_args([])).items():
            if default != getattr(args, arg):
                non_default_args[arg] = getattr(args, arg)

    # Handle EngineArgs instance
    elif isinstance(args, EngineArgs):
        default_args = EngineArgs(model=args.model)  # Create default instance
        for field in dataclasses.fields(args):
            current_val = getattr(args, field.name)
            default_val = getattr(default_args, field.name)
            if current_val != default_val:
                non_default_args[field.name] = current_val
        if default_args.model != EngineArgs.model:
            non_default_args["model"] = default_args.model
    else:
        raise TypeError(
            "Unsupported argument type. Must be Namespace or EngineArgs instance."
        )

    return non_default_args


def _jsonify_arg_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            key: _jsonify_arg_value(val)
            for key, val in dataclasses.asdict(value).items()
        }
    if isinstance(value, dict):
        return {str(key): _jsonify_arg_value(val) for key, val in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonify_arg_value(item) for item in value]
    if (model_dump := getattr(value, "model_dump", None)) is not None:
        return _jsonify_arg_value(model_dump(mode="json"))
    if (to_dict := getattr(value, "dict", None)) is not None:
        return _jsonify_arg_value(to_dict())
    return repr(value)


def jsonify_non_default_args(
    args: argparse.Namespace | EngineArgs,
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    non_default_args = get_non_default_args(args)
    if exclude is not None:
        for key in exclude:
            non_default_args.pop(key, None)

    return {key: _jsonify_arg_value(value) for key, value in non_default_args.items()}


# Fields whose values must never be logged verbatim.
_SENSITIVE_ARG_FIELDS = frozenset({"api_key", "hf_token", "watermark_config"})


def redact_sensitive_args(args: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `args` with sensitive values redacted for logging."""
    if not any(key in _SENSITIVE_ARG_FIELDS for key in args):
        return args
    return {
        key: ("***" if key in _SENSITIVE_ARG_FIELDS else value)
        for key, value in args.items()
    }

def log_non_default_args(args: argparse.Namespace | EngineArgs):
    non_default_args = get_non_default_args(args)
    logger.info("non-default args: %s", redact_sensitive_args(non_default_args))

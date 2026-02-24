"""Command-line interface for the benchmarking pipeline.

Supports two modes:
1. Legacy argparse mode: python -m benchmarking.cli run --benchmark rexvqa --provider medrax ...
2. Hydra mode: python -m benchmarking.cli benchmark=rexvqa provider=medrax ...
"""

import argparse
import sys
import hydra
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .llm_providers.base import LLMProvider
from .benchmarks import *
from .runner import BenchmarkRunner, BenchmarkRunConfig


def create_benchmark(benchmark_name: str, data_dir: str, **kwargs) -> Benchmark:
    """Create a benchmark based on the benchmark name.

    Args:
        benchmark_name (str): Name of the benchmark
        data_dir (str): Directory containing benchmark data
        **kwargs: Additional configuration parameters

    Returns:
        Benchmark: The configured benchmark
    """
    benchmark_map = {
        "rexvqa": ReXVQABenchmark,
        "chestagentbench": ChestAgentBenchBenchmark,
        "mimic_cxr": MimicCXRBenchmark,
    }

    if benchmark_name not in benchmark_map:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. Available: {list(benchmark_map.keys())}"
        )

    benchmark_class = benchmark_map[benchmark_name]
    return benchmark_class(data_dir, **kwargs)


def create_llm_provider(
    provider_type: str, model_name: str, system_prompt: str, **kwargs
) -> LLMProvider:
    """Create an LLM provider based on the model name and type.

    Args:
        provider_type (str): Type of provider (openai, google, openrouter, medrax, medgemma)
        model_name (str): Name of the model
        system_prompt (str): System prompt identifier to load from file
        **kwargs: Additional configuration parameters

    Returns:
        LLMProvider: The configured LLM provider
    """
    # Lazy imports to avoid slow startup
    if provider_type == "openai":
        from .llm_providers.openai_provider import OpenAIProvider

        provider_class = OpenAIProvider
    elif provider_type == "google":
        from .llm_providers.google_provider import GoogleProvider

        provider_class = GoogleProvider
    elif provider_type == "openrouter":
        from .llm_providers.openrouter_provider import OpenRouterProvider

        provider_class = OpenRouterProvider
    elif provider_type == "medrax":
        from .llm_providers.medrax_provider import MedRAXProvider

        provider_class = MedRAXProvider
    elif provider_type == "medgemma":
        from .llm_providers.medgemma_provider import MedGemmaProvider

        provider_class = MedGemmaProvider
    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. Available: openai, google, openrouter, medrax, medgemma"
        )

    return provider_class(model_name, system_prompt, **kwargs)


# =============================================================================
# Hydra-based CLI
# =============================================================================


def run_benchmark_from_config(cfg: DictConfig) -> None:
    """Run a benchmark using Hydra configuration.

    Args:
        cfg (DictConfig): Hydra configuration object
    """
    print(
        f"Running benchmark: {cfg.benchmark.name} with provider: {cfg.provider.type}, model: {cfg.provider.model}"
    )
    try:
        hydra_run_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        hydra_run_dir = (
            OmegaConf.select(cfg, "hydra.runtime.output_dir")
            or OmegaConf.select(cfg, "hydra.run.dir")
        )
    run_id_from_hydra = Path(hydra_run_dir).name if hydra_run_dir else None
    output_dir = Path(hydra_run_dir).parent if hydra_run_dir else Path(cfg.runner.output_dir)

    # Create benchmark
    benchmark_cfg = OmegaConf.to_container(cfg.benchmark, resolve=True) or {}
    benchmark_kwargs = {
        key: value
        for key, value in benchmark_cfg.items()
        if key not in {"name", "data_dir"} and value is not None
    }
    benchmark = create_benchmark(
        benchmark_name=cfg.benchmark.name, data_dir=cfg.benchmark.data_dir, **benchmark_kwargs
    )

    # Create LLM provider kwargs
    provider_kwargs = {
        "temperature": cfg.provider.temperature,
        "top_p": cfg.provider.top_p,
        "max_tokens": cfg.provider.max_tokens,
    }

    # Add MedRAX-specific kwargs if present
    if cfg.provider.type == "medrax":
        if OmegaConf.select(cfg, "provider.tools") is not None:
            # Convert OmegaConf list to Python list
            provider_kwargs["tools"] = OmegaConf.to_container(cfg.provider.tools, resolve=True)
        if OmegaConf.select(cfg, "provider.rag") is not None:
            # Convert OmegaConf dict to Python dict
            provider_kwargs["rag"] = OmegaConf.to_container(cfg.provider.rag, resolve=True)
        if OmegaConf.select(cfg, "provider.model_dir") is not None:
            provider_kwargs["model_dir"] = cfg.provider.model_dir
        if OmegaConf.select(cfg, "provider.temp_dir") is not None:
            provider_kwargs["temp_dir"] = cfg.provider.temp_dir
        if OmegaConf.select(cfg, "provider.prompt_file") is not None:
            provider_kwargs["prompt_file"] = cfg.provider.prompt_file
    else:
        # Allow prompt_file override for other providers (even if unused)
        if OmegaConf.select(cfg, "provider.prompt_file") is not None:
            provider_kwargs["prompt_file"] = cfg.provider.prompt_file

    llm_provider = create_llm_provider(
        provider_type=cfg.provider.type,
        model_name=cfg.provider.model,
        system_prompt=cfg.provider.system_prompt,
        **provider_kwargs,
    )

    # Create runner config
    config = BenchmarkRunConfig(
        benchmark_name=cfg.benchmark.name,
        provider_name=cfg.provider.type,
        model_name=cfg.provider.model,
        output_dir=str(output_dir),
        max_questions=cfg.benchmark.max_questions,
        temperature=cfg.provider.temperature,
        top_p=cfg.provider.top_p,
        max_tokens=cfg.provider.max_tokens,
        concurrency=cfg.runner.concurrency,
        random_seed=cfg.benchmark.random_seed,
        system_prompt=cfg.provider.system_prompt,
        run_id=run_id_from_hydra,
    )

    # Run benchmark
    runner = BenchmarkRunner(config)
    summary = runner.run_benchmark(benchmark, llm_provider)
    print(summary)


@hydra.main(version_base=None, config_path="config", config_name="config")
def hydra_main(cfg: DictConfig):
    """Hydra CLI entry point.

    Usage:
        # Use defaults
        python -m benchmarking.cli

        # Override benchmark and provider
        python -m benchmarking.cli benchmark=chestagentbench provider=google

        # Override specific values
        python -m benchmarking.cli provider.model=gpt-4o runner.concurrency=4

        # Override MedRAX tools
        python -m benchmarking.cli provider=medrax 'provider.tools=[MedGemmaVQATool,ChestXRayReportGeneratorTool]'
    """
    # Print resolved config for debugging
    print("=" * 60)
    print("Resolved Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    try:
        run_benchmark_from_config(cfg)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


# =============================================================================
# Legacy argparse-based CLI
# =============================================================================


def run_benchmark_command(args) -> None:
    """Run a benchmark using argparse arguments (legacy mode)."""
    print(
        f"Running benchmark: {args.benchmark} with provider: {args.provider}, model: {args.model}"
    )

    # Create benchmark
    benchmark_kwargs = {}
    benchmark_kwargs["max_questions"] = args.max_questions
    benchmark_kwargs["random_seed"] = args.random_seed
    benchmark = create_benchmark(
        benchmark_name=args.benchmark, data_dir=args.data_dir, **benchmark_kwargs
    )

    # Create LLM provider
    provider_kwargs = {}
    provider_kwargs["temperature"] = args.temperature
    provider_kwargs["top_p"] = args.top_p
    provider_kwargs["max_tokens"] = args.max_tokens
    llm_provider = create_llm_provider(
        provider_type=args.provider,
        model_name=args.model,
        system_prompt=args.system_prompt,
        **provider_kwargs,
    )

    # Create runner config
    config = BenchmarkRunConfig(
        benchmark_name=args.benchmark,
        provider_name=args.provider,
        model_name=args.model,
        output_dir=args.output_dir,
        max_questions=args.max_questions,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        random_seed=args.random_seed,
        system_prompt=args.system_prompt,
    )

    # Run benchmark
    runner = BenchmarkRunner(config)
    summary = runner.run_benchmark(benchmark, llm_provider)
    print(summary)


def argparse_main():
    """Legacy argparse CLI entry point."""
    parser = argparse.ArgumentParser(description="MedRAX Benchmarking Pipeline (Legacy Mode)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run benchmark command
    run_parser = subparsers.add_parser("run", help="Run a benchmark evaluation")
    run_parser.add_argument(
        "--benchmark",
        required=True,
        choices=["rexvqa", "chestagentbench", "mimic_cxr"],
        help="Benchmark dataset: rexvqa, chestagentbench, or mimic_cxr",
    )
    run_parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "google", "openrouter", "medrax", "medgemma"],
        help="LLM provider to use",
    )
    run_parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., gpt-4o, gpt-4.1-2025-04-14, gemini-2.5-pro)",
    )
    run_parser.add_argument(
        "--system-prompt",
        required=True,
        choices=["MEDICAL_ASSISTANT", "CHESTAGENTBENCH_PROMPT", "MEDGEMMA_PROMPT"],
        help="System prompt: MEDICAL_ASSISTANT (general) or CHESTAGENTBENCH_PROMPT (benchmarks)",
    )
    run_parser.add_argument(
        "--data-dir", required=True, help="Directory containing benchmark data files"
    )
    run_parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )
    run_parser.add_argument(
        "--max-questions", type=int, help="Maximum number of questions to process (default: all)"
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="Model temperature for response generation (default: 0.7)",
    )
    run_parser.add_argument(
        "--top-p", type=float, default=0.95, help="Top-p nucleus sampling parameter (default: 0.95)"
    )
    run_parser.add_argument(
        "--max-tokens",
        type=int,
        default=5000,
        help="Maximum tokens per model response (default: 5000)",
    )
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of datapoints to process in parallel (default: 1)",
    )
    run_parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for shuffling benchmark data (enables reproducible runs, default: 42)",
    )

    run_parser.set_defaults(func=run_benchmark_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


# =============================================================================
# Main entry point - detects which mode to use
# =============================================================================


def main():
    """Main entry point that detects whether to use argparse or Hydra mode.

    - If first argument is 'run', use legacy argparse mode
    - Otherwise, use Hydra mode
    """
    # Check if we should use legacy argparse mode
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        argparse_main()
    else:
        # Avoid GlobalHydra re-init errors when invoked multiple times in one process
        try:
            from hydra.core.global_hydra import GlobalHydra

            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
        except Exception:
            pass
        hydra_main()


if __name__ == "__main__":
    main()

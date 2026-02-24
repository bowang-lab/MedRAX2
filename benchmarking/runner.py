"""Main test runner for benchmarking pipeline."""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from .llm_providers import LLMProvider, LLMRequest, LLMResponse
from .benchmarks import Benchmark, BenchmarkDataPoint


@dataclass
class BenchmarkResult:
    """Result of running a benchmark on a single data point."""

    data_point_id: str
    question: str
    model_answer: str
    correct_answer: str
    is_correct: bool
    duration: float
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    chunk_history: Optional[Any] = None
    tool_execution_trace: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkRunConfig:
    """Configuration for a benchmark run."""

    benchmark_name: str
    provider_name: str
    model_name: str
    output_dir: str
    max_questions: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 5000
    concurrency: int = 1
    run_id: Optional[str] = None
    random_seed: Optional[int] = None
    system_prompt: Optional[str] = None


class BenchmarkRunner:
    """Main class for running benchmarks against LLM providers."""

    def __init__(self, config: BenchmarkRunConfig):
        """Initialize the benchmark runner.

        Args:
            config (BenchmarkRunConfig): Configuration for the benchmark run
        """
        self.config = config
        self.results = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_name: Optional[str] = None
        self.prompt_file: Optional[str] = None
        self.prompt_content: Optional[str] = None

        # Generate unique run ID
        self.run_id = config.run_id or f"{config.benchmark_name}_{config.provider_name}_{config.model_name}_{config.max_questions}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()
        self.logger.info(f"Initialized benchmark runner with ID: {self.run_id}")
        self.logger.info(f"Run ID from Hydra: {self.config.run_id}")

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        # Create logger
        self.logger = logging.getLogger(f"benchmark_runner_{self.run_id}")
        self.logger.setLevel(logging.INFO)
        # Clear existing handlers to avoid duplicate logs when reusing the logger name
        self.logger.handlers.clear()

        # Log to both file and console for traceability
        log_file = self.run_dir / f"benchmark_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def run_benchmark(
        self,
        benchmark: Benchmark,
        llm_provider: LLMProvider,
    ) -> Dict[str, Any]:
        """Run a benchmark against an LLM provider.
        
        Args:
            benchmark (Benchmark): The benchmark to run
            llm_provider (LLMProvider): The LLM provider to test
            
        Returns:
            Dict[str, Any]: Summary of benchmark results
        """
        self.logger.info(f"Starting benchmark run: {self.run_id}")
        self.logger.info(f"Benchmark: {benchmark}")
        self.logger.info(f"Provider: {llm_provider.provider_name}")
        self.logger.info(f"Model: {llm_provider.model_name}")
        # Capture prompt details for logging
        self.prompt_name = getattr(llm_provider, "prompt_name", None)
        self.prompt_file = getattr(llm_provider, "prompt_file", None)
        self.prompt_content = getattr(llm_provider, "system_prompt", None)

        # Test provider connection
        if not llm_provider.test_connection():
            self.logger.error("LLM provider connection test failed")
            return {"error": "LLM provider connection test failed"}
        
        # Initialize counters
        processed = 0
        correct = 0
        total_duration = 0.0
        
        # Determine concurrency
        max_workers = max(1, int(getattr(self.config, "concurrency", 1) or 1))
        
        # Process data points in parallel using a bounded thread pool
        with tqdm(total=len(benchmark), desc="Processing questions") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(self._process_data_point, dp, llm_provider): idx
                    for idx, dp in enumerate(benchmark)
                }
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        self.logger.error(f"Error processing data point {idx}: {e}")
                        result = BenchmarkResult(
                            data_point_id=f"error_{idx}",
                            question="",
                            model_answer="",
                            correct_answer="",
                            is_correct=False,
                            duration=0.0,
                            error=str(e),
                        )
                    
                    # Update counters
                    processed += 1
                    if result.is_correct:
                        correct += 1
                    total_duration += result.duration
                    
                    # Add to results and persist immediately
                    self.results.append(result)
                    self._save_individual_result(result)
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Periodic logging
                    if processed % 10 == 0:
                        accuracy = (correct / processed) * 100
                        avg_duration = total_duration / processed if processed > 0 else 0.0
                        self.logger.info(
                            f"Progress: {processed}/{len(benchmark)} | "
                            f"Accuracy: {accuracy:.2f}% | "
                            f"Avg Duration: {avg_duration:.2f}s"
                        )
        
        # Save final results
        summary = self._save_final_results(benchmark)

        self.logger.info(f"Benchmark run completed: {self.run_id}")
        self.logger.info(f"Summary: {summary}")

        return summary

    def _process_data_point(
        self, data_point: BenchmarkDataPoint, llm_provider: LLMProvider
    ) -> BenchmarkResult:
        """Process a single data point.
        
        Args:
            data_point (BenchmarkDataPoint): The data point to process
            llm_provider (LLMProvider): The LLM provider to use
            
        Returns:
            BenchmarkResult: Result of processing the data point
        """
        start_time = time.time()
        
        try:
            # Create request for LLM
            request = LLMRequest(text=data_point.text, images=data_point.images)

            # Get response from LLM
            response: LLMResponse = llm_provider.generate_response(request)

            # Extract and normalize answer.
            extracted_answer = self._extract_answer(response.content)
            is_multi_select = bool((data_point.metadata or {}).get("multi_select")) or (
                (data_point.metadata or {}).get("label_mode") == "allow_multiple_findings"
            )
            model_answer = self._normalize_answer(extracted_answer, multi_select=is_multi_select)
            correct_answer = self._normalize_answer(
                data_point.correct_answer or "", multi_select=is_multi_select
            )
            is_correct = model_answer == correct_answer
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Return result
            return BenchmarkResult(
                data_point_id=data_point.id,
                question=data_point.text,
                model_answer=model_answer,
                correct_answer=correct_answer,
                is_correct=is_correct,
                duration=duration,
                usage=response.usage,
                chunk_history=response.chunk_history,
                tool_execution_trace=response.tool_execution_trace,
                metadata={
                    "data_point_metadata": data_point.metadata,
                    "raw_response": response.content,
                    "extracted_answer_raw": extracted_answer,
                    "normalized_model_answer": model_answer,
                    "normalized_correct_answer": correct_answer,
                    "multi_select_eval": is_multi_select,
                },
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return BenchmarkResult(
                data_point_id=data_point.id,
                question=data_point.text,
                model_answer="",
                correct_answer=data_point.correct_answer or "",
                is_correct=False,
                duration=duration,
                error=str(e),
                chunk_history=None,
                tool_execution_trace=None,
                metadata={"data_point_metadata": data_point.metadata},
            )

    def _extract_answer(self, response_text: str) -> str:
        """Extract the answer from the model response.
        
        Args:
            response_text (str): The full response text from the model
            
        Returns:
            str: The extracted answer
        """
        if not response_text:
            return ""

        boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", response_text, flags=re.IGNORECASE)
        for boxed in boxed_matches:
            parsed = self._parse_answer_tokens(boxed)
            if parsed:
                return parsed

        answer_patterns = [
            r"(?:final\s*answer|answer)\s*[:\-]\s*([A-Za-z0-9](?:\s*(?:,|/|;|\band\b)\s*[A-Za-z0-9])*)",
            r"(?:option|options)\s*[:\-]\s*([A-Za-z0-9](?:\s*(?:,|/|;|\band\b)\s*[A-Za-z0-9])*)",
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, response_text, flags=re.IGNORECASE)
            if match:
                parsed = self._parse_answer_tokens(match.group(1))
                if parsed:
                    return parsed

        return ""

    def _parse_answer_tokens(self, value: str) -> str:
        """Parse one or more answer tokens (letters/numbers) into normalized CSV format."""
        if not value:
            return ""

        normalized = value.upper()
        normalized = normalized.replace("AND", ",")
        normalized = normalized.replace("/", ",")
        normalized = normalized.replace("|", ",")
        normalized = normalized.replace(";", ",")

        tokens = re.split(r"[^A-Z0-9]+", normalized)

        parsed_tokens = set()
        for token in tokens:
            if not token:
                continue
            if token.isdigit():
                parsed_tokens.add(str(int(token)))
                continue
            if token.isalpha() and 1 <= len(token) <= 2:
                parsed_tokens.add(token)

        ordered_tokens = sorted(
            parsed_tokens,
            key=lambda t: (0, int(t)) if t.isdigit() else (1, t),
        )
        return ",".join(ordered_tokens)

    def _normalize_answer(self, value: str, multi_select: bool) -> str:
        """Normalize extracted answers for single-choice and multi-select benchmarks."""
        parsed = self._parse_answer_tokens(value)
        if parsed:
            if multi_select:
                return parsed
            return parsed.split(",")[0]
        return (value or "").strip()

    def _save_individual_result(self, result: BenchmarkResult) -> None:
        """Save a single result to its own JSON file.
        
        Args:
            result (BenchmarkResult): The result to save
        """
        # Sanitize data_point_id for filename (remove invalid characters)
        safe_id = re.sub(r"[^\w\-_.]", "_", result.data_point_id)

        # Create run_id directory and individual_results subdirectory
        individual_results_dir = self.run_dir / "individual_results"
        individual_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with benchmark name and data point ID
        filename = f"{self.config.benchmark_name}_{safe_id}.json"
        result_file = individual_results_dir / filename
        
        # Convert result to serializable format
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "data_point_id": result.data_point_id,
            "question": result.question,
            "model_answer": result.model_answer,
            "correct_answer": result.correct_answer,
            "is_correct": result.is_correct,
            "duration": result.duration,
            "usage": result.usage,
            "error": result.error,
            "chunk_history": result.chunk_history,
            "tool_execution_trace": result.tool_execution_trace,
            "metadata": result.metadata,
        }
        
        # Save to file
        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)

    def _save_final_results(self, benchmark: Benchmark) -> Dict[str, Any]:
        """Save final results and return summary.
        
        Args:
            benchmark (Benchmark): The benchmark that was run
            
        Returns:
            Dict[str, Any]: Summary of results
        """
        # Create run_id directory and final_results subdirectory
        final_results_dir = self.run_dir / "final_results"
        final_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = final_results_dir / f"{self.run_id}_results.json"
        
        # Convert results to serializable format for final file
        results_data = []
        for result in self.results:
            results_data.append(
                {
                    "data_point_id": result.data_point_id,
                    "question": result.question,
                    "model_answer": result.model_answer,
                    "correct_answer": result.correct_answer,
                    "is_correct": result.is_correct,
                    "duration": result.duration,
                    "usage": result.usage,
                    "error": result.error,
                    "tool_execution_trace": result.tool_execution_trace,
                    "metadata": result.metadata,
                }
            )

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        
        # Calculate summary statistics
        total_questions = len(self.results)
        correct_answers = sum(1 for r in self.results if r.is_correct)
        total_duration = sum(r.duration for r in self.results)

        accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        usage_summary = self._aggregate_usage_metrics()

        # Create summary
        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "benchmark_name": self.config.benchmark_name,
                "provider_name": self.config.provider_name,
                "model_name": self.config.model_name,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
                "system_prompt_name": self.config.system_prompt or self.prompt_name,
            },
            "benchmark_info": {
                "total_size": len(benchmark),
                "processed_questions": total_questions,
            },
            "results": {
                "accuracy": accuracy,
                "correct_answers": correct_answers,
                "total_questions": total_questions,
                "total_duration": total_duration,
                "avg_duration_per_question": (
                    total_duration / total_questions if total_questions > 0 else 0
                ),
            },
            "usage": usage_summary,
            "results_file": str(results_file),
        }
        
        prompt_log = {
            "name": self.prompt_name or self.config.system_prompt,
            "file": self.prompt_file,
            "content": self.prompt_content,
        }
        if any(prompt_log.values()):
            summary["prompt"] = prompt_log

        # Save summary
        summary_file = final_results_dir / f"{self.run_id}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _aggregate_usage_metrics(self) -> Dict[str, Any]:
        """Aggregate token and request usage across all results."""
        token_totals: Dict[str, float] = {}
        llm_requests_total = 0
        questions_with_usage = 0

        for result in self.results:
            usage = result.usage or {}
            if not usage:
                continue

            questions_with_usage += 1
            llm_requests_total += int(usage.get("llm_requests") or 0)

            # Nested token structure (e.g., {"tokens": {"input": ..., "output": ...}})
            nested_tokens = usage.get("tokens")
            if isinstance(nested_tokens, dict):
                for key, value in nested_tokens.items():
                    if isinstance(value, (int, float)):
                        token_totals[key] = token_totals.get(key, 0) + value

            # Flat token keys used by some providers
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
            ):
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    token_totals[key] = token_totals.get(key, 0) + value

        avg_tokens_per_question = {}
        if questions_with_usage > 0:
            avg_tokens_per_question = {
                key: value / questions_with_usage for key, value in token_totals.items()
            }

        calculated_total_tokens = token_totals.get("total_tokens", 0)
        if calculated_total_tokens == 0 and token_totals:
            calculated_total_tokens = (
                token_totals.get("input", 0)
                + token_totals.get("output", 0)
                + token_totals.get("reasoning", 0)
            )
            if calculated_total_tokens == 0:
                calculated_total_tokens = (
                    token_totals.get("prompt_tokens", 0)
                    + token_totals.get("completion_tokens", 0)
                )

        return {
            "questions_with_usage": questions_with_usage,
            "total_llm_requests": llm_requests_total,
            "token_totals": token_totals,
            "calculated_total_tokens": calculated_total_tokens,
            "avg_tokens_per_question": avg_tokens_per_question,
        }

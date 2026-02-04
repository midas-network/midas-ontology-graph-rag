#!/usr/bin/env python3
"""
CLI interface for AnalysisManager.

Usage:
  python run_analysis.py                    # Run with defaults
  python run_analysis.py --examples 3       # Use 3 examples
  python run_analysis.py --gpu              # Use remote GPU
  python run_analysis.py --baseline-only    # Run baseline only
"""
import argparse
import sys
from analysis_manager import AnalysisManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run context value analysis on LLM prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run full analysis with defaults
  %(prog)s --examples 3             # Use 3 few-shot examples
  %(prog)s --gpu                    # Use remote GPU server
  %(prog)s --baseline-only          # Run only baseline test
  %(prog)s --no-ontology            # Skip ontology context test
  %(prog)s --model llama3.1:latest  # Use different model
        """
    )

    parser.add_argument(
        '--ontology',
        default=None,
        help="Path to OWL ontology file (default: ontologies/midas_data/midas-data.owl or midas-data.owl)"
    )

    parser.add_argument(
        "--papers",
        default="./data/modeling_papers.json",
        help="Path to modeling papers JSON (default: ./data/modeling_papers.json)"
    )

    parser.add_argument(
        "--abstract",
        default="data/fred-abstract.txt",
        help="Path to abstract text file (default: data/fred-abstract.txt)"
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=2,
        help="Number of few-shot examples to use (default: 2)"
    )

    parser.add_argument(
        "--model",
        default="llama3.2:latest",
        help="LLM model name (default: llama3.2:latest)"
    )

    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="LLM API host URL (default: http://localhost:11434)"
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use remote GPU server (shortcut for --host http://gpu-n29:41091)"
    )

    parser.add_argument(
        "--output",
        default="analysis_report.json",
        help="Output path for JSON report (default: analysis_report.json)"
    )

    parser.add_argument(
        "--comparison",
        default="responses_comparison.txt",
        help="Output path for response comparison (default: responses_comparison.txt)"
    )

    # Test selection options
    test_group = parser.add_argument_group("test selection")
    test_group.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only baseline test"
    )
    test_group.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline test"
    )
    test_group.add_argument(
        "--no-ontology",
        action="store_true",
        help="Skip ontology test"
    )
    test_group.add_argument(
        "--no-examples",
        action="store_true",
        help="Skip examples test"
    )
    test_group.add_argument(
        "--no-full",
        action="store_true",
        help="Skip full context test"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle GPU shortcut
    host = args.host
    if args.gpu:
        host = "http://gpu-n29:41091"
        print(f"Using remote GPU: {host}")

    # Create manager
    manager = AnalysisManager(
        ontology_path=args.ontology,
        papers_path=args.papers,
        abstract_path=args.abstract,
        num_examples=args.examples,
        llm_model=args.model,
        llm_host=host
    )

    # Determine which tests to run
    if args.baseline_only:
        print("Running baseline test only...\n")
        manager.load_data()
        result = manager.run_single_test(manager.run_baseline)

        print(f"\n{'=' * 80}")
        print("RESULT")
        print(f"{'=' * 80}")
        print(f"Prompt size: {result.prompt_size:,} characters")
        print(f"Response size: {result.response_size:,} characters")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        print(f"\nResponse:\n{result.response}")

    else:
        # Run selected tests
        print("Running analysis...\n")
        manager.load_data()

        results = {}

        if not args.no_baseline:
            result = manager.run_single_test(manager.run_baseline)
            results[result.name] = result

        if not args.no_ontology:
            result = manager.run_single_test(manager.run_with_ontology)
            results[result.name] = result

        if not args.no_examples:
            result = manager.run_single_test(manager.run_with_examples)
            results[result.name] = result

        if not args.no_full:
            result = manager.run_single_test(manager.run_full_context)
            results[result.name] = result

        # Analyze and save
        recommendations = manager.analyze_results(results)
        manager.print_summary(results, recommendations)
        manager.save_report(results, recommendations, args.output)
        manager.save_responses_comparison(results, args.comparison)

        print(f"\n✓ Analysis complete!")
        print(f"  - Report saved to: {args.output}")
        print(f"  - Comparison saved to: {args.comparison}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


"""
Entry point for the Mood Machine AI System.

This is the main CLI interface that runs the full pipeline:
  - Rule-based mood analysis with negation and emoji handling
  - ML classification with TF-IDF + Logistic Regression
  - RAG-powered contextual explanations
  - Confidence scoring and ensemble predictions
  - Comprehensive logging

Usage:
    python main.py              # Run full demo + interactive mode
    python main.py --evaluate   # Run evaluation only
    python main.py --interactive # Interactive mode only
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from pipeline import MoodPipeline
from dataset import SAMPLE_POSTS, TRUE_LABELS


# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------

def setup_logging(log_dir: str = "logs") -> None:
    """
    Configure logging to both console and file.

    Logs are saved to logs/mood_machine_YYYYMMDD_HHMMSS.log
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mood_machine_{timestamp}.log")

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # File handler: detailed logs
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    # Console handler: info and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    logging.info("Logging initialized. Log file: %s", log_file)


# ---------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------

def run_evaluation(pipeline: MoodPipeline) -> dict:
    """Evaluate the pipeline on the labeled dataset."""
    print("\n" + "=" * 60)
    print("  📊 PIPELINE EVALUATION ON DATASET")
    print("=" * 60)

    evaluation = pipeline.evaluate(SAMPLE_POSTS, TRUE_LABELS)

    print(f"\nDataset: {evaluation['total']} posts")
    print(f"Accuracy: {evaluation['accuracy']:.0%} ({evaluation['correct']}/{evaluation['total']})")
    print(f"Average Confidence: {evaluation['average_confidence']:.2f}")
    print()

    for r in evaluation["results"]:
        marker = "✅" if r["correct"] else "❌"
        print(f'  {marker} "{r["text"]}"')
        print(f'     Predicted: {r["predicted_label"]} | True: {r["true_label"]} | Confidence: {r["confidence"]:.2f}')
        print()

    return evaluation


def run_demo(pipeline: MoodPipeline) -> None:
    """Run a few demo analyses with full explanations."""
    print("\n" + "=" * 60)
    print("  🎭 MOOD MACHINE DEMO")
    print("=" * 60)

    demo_texts = [
        "I love this class so much!",
        "Today was a terrible, awful, no-good day",
        "Feeling tired but kind of hopeful about tomorrow",
        "Bruh this is the worst exam I've ever taken 💀",
        "Can't stop smiling today 😊 everything went perfect",
    ]

    for text in demo_texts:
        result = pipeline.analyze(text)
        print(f'\n{"─" * 50}')
        print(f'Input: "{text}"')
        print(f'{"─" * 50}')
        print()
        print(result["explanation"])

        if result.get("ml"):
            print(f"\n  🤖 ML model says: {result['ml']['label']} "
                  f"(confidence: {result['ml']['confidence']:.2f})")

        print()


def run_interactive(pipeline: MoodPipeline) -> None:
    """Interactive mode: type text and get mood analysis."""
    print("\n" + "=" * 60)
    print("  🎤 INTERACTIVE MOOD MACHINE")
    print("=" * 60)
    print("\nType a sentence to analyze its mood.")
    print("Type 'quit' or press Enter on an empty line to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input == "" or user_input.lower() == "quit":
            print("Goodbye from the Mood Machine! 👋")
            break

        result = pipeline.analyze(user_input)
        print()
        print(result["explanation"])

        if result.get("ml"):
            print(f"\n  🤖 ML model: {result['ml']['label']} "
                  f"(confidence: {result['ml']['confidence']:.2f})")
        print()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mood Machine: AI-powered mood analysis system"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluation on the dataset only"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run interactive mode only"
    )
    parser.add_argument(
        "--no-rag", action="store_true",
        help="Disable RAG explanations"
    )
    parser.add_argument(
        "--no-ml", action="store_true",
        help="Disable ML classifier"
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging()

    print("🎭 Mood Machine — AI-Powered Mood Analysis System")
    print("=" * 50)
    print("Initializing pipeline...")

    # Initialize pipeline
    pipeline = MoodPipeline(
        use_rag=not args.no_rag,
        use_ml=not args.no_ml,
    )
    print("✅ Pipeline ready!\n")

    if args.evaluate:
        run_evaluation(pipeline)
    elif args.interactive:
        run_interactive(pipeline)
    else:
        # Full demo: evaluation + demo + interactive
        run_evaluation(pipeline)
        run_demo(pipeline)
        run_interactive(pipeline)


if __name__ == "__main__":
    main()

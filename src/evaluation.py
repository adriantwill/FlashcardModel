"""
Evaluation module for assessing flashcard quality.

This module provides metrics and quality assessment tools for evaluating
generated flashcards.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from evaluate import load
import numpy as np
from tqdm import tqdm


class FlashcardEvaluator:
    """
    Evaluator for flashcard generation quality.
    """

    def __init__(self):
        """Initialize evaluation metrics."""
        print("Loading evaluation metrics...")
        self.rouge = load('rouge')
        self.bleu = load('bleu')
        print("Metrics loaded successfully")

    def compute_rouge_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE and BLEU scores for generated flashcards.

        Args:
            predictions: List of generated texts
            references: List of reference texts

        Returns:
            Dictionary with ROUGE and BLEU scores
        """
        # Compute ROUGE scores
        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=references
        )

        # Compute BLEU score
        # BLEU expects list of references for each prediction
        bleu_score = self.bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )

        return {
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'bleu': bleu_score['bleu']
        }

    def evaluate_from_jsonl(
        self,
        predictions_file: str,
        references_file: str
    ) -> Dict[str, float]:
        """
        Evaluate generated flashcards against reference data.

        Expected format for JSONL files:
        - predictions: {"input": "...", "target": "Q: ... A: ..."}
        - references: Same format

        Args:
            predictions_file: Path to predictions JSONL
            references_file: Path to references JSONL

        Returns:
            Dictionary with evaluation metrics
        """
        # Load predictions
        predictions = []
        with open(predictions_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data['target'])

        # Load references
        references = []
        with open(references_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                references.append(data['target'])

        print(f"Evaluating {len(predictions)} examples...")

        # Compute metrics
        metrics = self.compute_rouge_bleu(predictions, references)

        return metrics

    def evaluate_flashcards(
        self,
        generated_flashcards: List[Dict[str, str]],
        reference_flashcards: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """
        Evaluate generated flashcards against references.

        Args:
            generated_flashcards: List of generated flashcard dicts
            reference_flashcards: List of reference flashcard dicts

        Returns:
            Dictionary with evaluation metrics
        """
        # Extract Q&A pairs
        gen_qa = [
            f"Q: {fc['question']} A: {fc['answer']}"
            for fc in generated_flashcards
        ]

        ref_qa = [
            f"Q: {fc['question']} A: {fc['answer']}"
            for fc in reference_flashcards
        ]

        # Compute metrics
        metrics = self.compute_rouge_bleu(gen_qa, ref_qa)

        return metrics

    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print evaluation metrics in a formatted way.

        Args:
            metrics: Dictionary of metric scores
        """
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"\nROUGE-1:  {metrics['rouge1']:.4f}")
        print(f"ROUGE-2:  {metrics['rouge2']:.4f}")
        print(f"ROUGE-L:  {metrics['rougeL']:.4f}")
        print(f"BLEU:     {metrics['bleu']:.4f}")
        print("\n" + "="*60)

        # Interpretation
        print("\nInterpretation:")
        if metrics['rougeL'] > 0.5:
            print("  ✓ Excellent performance (ROUGE-L > 0.5)")
        elif metrics['rougeL'] > 0.4:
            print("  ✓ Good performance (ROUGE-L > 0.4)")
        elif metrics['rougeL'] > 0.35:
            print("  ⚠ Decent baseline (ROUGE-L > 0.35)")
        else:
            print("  ✗ Below target (ROUGE-L < 0.35)")

    def sample_for_manual_review(
        self,
        flashcards: List[Dict[str, str]],
        n_samples: int = 50,
        random_seed: int = 42
    ) -> List[Dict[str, str]]:
        """
        Sample flashcards for manual quality review.

        Args:
            flashcards: List of flashcard dictionaries
            n_samples: Number of samples to extract
            random_seed: Random seed for reproducibility

        Returns:
            List of sampled flashcards
        """
        np.random.seed(random_seed)

        if len(flashcards) <= n_samples:
            return flashcards

        indices = np.random.choice(len(flashcards), n_samples, replace=False)
        samples = [flashcards[i] for i in indices]

        return samples

    def create_manual_review_template(
        self,
        flashcards: List[Dict[str, str]],
        output_path: str = "outputs/manual_review.csv"
    ):
        """
        Create a CSV template for manual quality review.

        The template includes flashcards and empty columns for scoring.

        Args:
            flashcards: List of flashcard dictionaries
            output_path: Path to save the review template
        """
        # Sample flashcards
        samples = self.sample_for_manual_review(flashcards)

        # Create DataFrame with review columns
        df = pd.DataFrame({
            'chunk_id': [fc.get('chunk_id', i) for i, fc in enumerate(samples)],
            'source_text': [fc['source_text'][:200] + '...' if len(fc['source_text']) > 200 else fc['source_text'] for fc in samples],
            'question': [fc['question'] for fc in samples],
            'answer': [fc['answer'] for fc in samples],
            'relevance_1_5': [''] * len(samples),
            'correctness_1_5': [''] * len(samples),
            'clarity_1_5': [''] * len(samples),
            'usefulness_1_5': [''] * len(samples),
            'notes': [''] * len(samples)
        })

        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"\n✓ Created manual review template: {output_path}")
        print(f"  {len(samples)} flashcards ready for review")
        print("\nScoring guide (1-5 scale):")
        print("  Relevance:   1=unrelated, 5=perfectly relevant")
        print("  Correctness: 1=wrong, 5=completely accurate")
        print("  Clarity:     1=confusing, 5=very clear")
        print("  Usefulness:  1=not useful, 5=very useful")

    def analyze_manual_review(
        self,
        review_file: str
    ) -> Dict[str, any]:
        """
        Analyze completed manual review scores.

        Args:
            review_file: Path to completed review CSV

        Returns:
            Dictionary with analysis results
        """
        df = pd.read_csv(review_file)

        # Calculate average scores
        score_columns = ['relevance_1_5', 'correctness_1_5', 'clarity_1_5', 'usefulness_1_5']

        # Convert to numeric, replacing empty strings with NaN
        for col in score_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate metrics
        results = {
            'total_reviewed': len(df),
            'avg_relevance': df['relevance_1_5'].mean(),
            'avg_correctness': df['correctness_1_5'].mean(),
            'avg_clarity': df['clarity_1_5'].mean(),
            'avg_usefulness': df['usefulness_1_5'].mean(),
            'pct_good_quality': (
                (df[score_columns].mean(axis=1) >= 3).sum() / len(df) * 100
            )
        }

        # Print results
        print("\n" + "="*60)
        print("MANUAL REVIEW ANALYSIS")
        print("="*60)
        print(f"\nTotal reviewed: {results['total_reviewed']}")
        print(f"\nAverage scores (1-5):")
        print(f"  Relevance:   {results['avg_relevance']:.2f}")
        print(f"  Correctness: {results['avg_correctness']:.2f}")
        print(f"  Clarity:     {results['avg_clarity']:.2f}")
        print(f"  Usefulness:  {results['avg_usefulness']:.2f}")
        print(f"\nGood quality (avg ≥ 3): {results['pct_good_quality']:.1f}%")
        print("="*60)

        return results


def main():
    """
    Example usage from command line.
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  1. Evaluate against test set:")
        print("     python evaluation.py evaluate <predictions.jsonl> <references.jsonl>")
        print("\n  2. Create manual review template:")
        print("     python evaluation.py review <flashcards.json>")
        print("\n  3. Analyze manual review:")
        print("     python evaluation.py analyze <review.csv>")
        sys.exit(1)

    command = sys.argv[1]
    evaluator = FlashcardEvaluator()

    if command == "evaluate":
        if len(sys.argv) < 4:
            print("Error: Need predictions and references files")
            sys.exit(1)

        predictions_file = sys.argv[2]
        references_file = sys.argv[3]

        metrics = evaluator.evaluate_from_jsonl(predictions_file, references_file)
        evaluator.print_metrics(metrics)

    elif command == "review":
        if len(sys.argv) < 3:
            print("Error: Need flashcards file")
            sys.exit(1)

        flashcards_file = sys.argv[2]

        # Load flashcards
        with open(flashcards_file, 'r') as f:
            flashcards = json.load(f)

        evaluator.create_manual_review_template(flashcards)

    elif command == "analyze":
        if len(sys.argv) < 3:
            print("Error: Need review CSV file")
            sys.exit(1)

        review_file = sys.argv[2]
        evaluator.analyze_manual_review(review_file)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()

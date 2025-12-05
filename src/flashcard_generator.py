"""
Flashcard generation module using fine-tuned FLAN-T5 model.

This module handles inference: generating question-answer pairs from text chunks.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd

from pdf_processor import process_pdf_to_chunks


class FlashcardGenerator:
    """
    Flashcard generator using a fine-tuned FLAN-T5 model.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the flashcard generator.

        Args:
            model_path: Path to the fine-tuned model directory
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        print(f"Loading model from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f"Model loaded on {self.device}")

    def generate_flashcard(
        self,
        source_text: str,
        max_length: int = 256,
        num_beams: int = 4,
        temperature: float = 0.7,
        no_repeat_ngram_size: int = 3
    ) -> str:
        """
        Generate a flashcard (Q&A) from source text.

        Args:
            source_text: Input paragraph/text
            max_length: Maximum length of generated output
            num_beams: Number of beams for beam search
            temperature: Sampling temperature (higher = more creative)
            no_repeat_ngram_size: Prevent repetition of n-grams

        Returns:
            Generated text in format "Q: ... A: ..."
        """
        # Tokenize input
        inputs = self.tokenizer(
            source_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                temperature=temperature,
                do_sample=False,  # Deterministic for consistency
                no_repeat_ngram_size=no_repeat_ngram_size
            )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

    def parse_flashcard(self, generated_text: str) -> Dict[str, str]:
        """
        Parse generated text into question and answer components.

        Expects format: "Q: {question} A: {answer}"

        Args:
            generated_text: Model output

        Returns:
            Dictionary with 'question' and 'answer' keys
        """
        # Try to extract Q: and A: parts using regex
        match = re.search(r'Q:\s*(.+?)\s*A:\s*(.+)', generated_text, re.DOTALL)

        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
        else:
            # Fallback: try splitting on first '?'
            if '?' in generated_text:
                parts = generated_text.split('?', 1)
                question = parts[0].strip() + '?'
                answer = parts[1].strip() if len(parts) > 1 else ""
            else:
                # Couldn't parse - return as-is
                question = generated_text
                answer = ""

        return {
            'question': question,
            'answer': answer,
            'raw_output': generated_text
        }

    def generate_from_chunks(
        self,
        chunks: List[Dict[str, any]],
        show_progress: bool = True
    ) -> List[Dict[str, any]]:
        """
        Generate flashcards from a list of text chunks.

        Args:
            chunks: List of chunk dictionaries (from pdf_processor)
            show_progress: Whether to show progress bar

        Returns:
            List of flashcard dictionaries with questions and answers
        """
        flashcards = []

        iterator = tqdm(chunks, desc="Generating flashcards") if show_progress else chunks

        for chunk in iterator:
            # Generate flashcard
            generated = self.generate_flashcard(chunk['source_text'])

            # Parse into Q&A
            parsed = self.parse_flashcard(generated)

            # Combine with chunk metadata
            flashcard = {
                'chunk_id': chunk.get('chunk_id', None),
                'source_text': chunk['source_text'],
                'question': parsed['question'],
                'answer': parsed['answer'],
                'raw_output': parsed['raw_output'],
                'word_count': chunk.get('word_count', len(chunk['source_text'].split())),
                'pdf_file': chunk.get('pdf_file', None)
            }

            flashcards.append(flashcard)

        return flashcards

    def process_pdf_to_flashcards(
        self,
        pdf_path: str,
        output_format: str = 'json',
        output_dir: str = 'outputs',
        min_words: int = 30,
        max_words: int = 200
    ) -> str:
        """
        Complete pipeline: PDF → flashcards → save to file.

        Args:
            pdf_path: Path to input PDF
            output_format: Output format ('json' or 'csv')
            output_dir: Directory to save output
            min_words: Minimum words per chunk
            max_words: Maximum words per chunk

        Returns:
            Path to output file
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"PROCESSING PDF: {pdf_path.name}")
        print(f"{'='*60}\n")

        # Step 1: Extract and chunk PDF
        print("[1/3] Extracting and chunking PDF...")
        chunks = process_pdf_to_chunks(str(pdf_path), min_words, max_words)

        if not chunks:
            print("No chunks extracted. Exiting.")
            return None

        # Step 2: Generate flashcards
        print(f"\n[2/3] Generating flashcards from {len(chunks)} chunks...")
        flashcards = self.generate_from_chunks(chunks)

        # Step 3: Save output
        print(f"\n[3/3] Saving flashcards...")

        output_filename = f"flashcards_{pdf_path.stem}.{output_format}"
        output_path = output_dir / output_filename

        if output_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(flashcards, f, indent=2, ensure_ascii=False)
        elif output_format == 'csv':
            df = pd.DataFrame(flashcards)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        print(f"\n✓ Saved {len(flashcards)} flashcards to {output_path}")
        print(f"\n{'='*60}")

        return str(output_path)


def main():
    """
    Example usage from command line.
    """
    import sys

    if len(sys.argv) < 3:
        print("Usage: python flashcard_generator.py <model_path> <pdf_path> [output_format]")
        print("\nExample:")
        print("  python flashcard_generator.py models/flan-t5-flashcard-v1 document.pdf json")
        sys.exit(1)

    model_path = sys.argv[1]
    pdf_path = sys.argv[2]
    output_format = sys.argv[3] if len(sys.argv) > 3 else 'json'

    # Initialize generator
    generator = FlashcardGenerator(model_path)

    # Process PDF
    output_path = generator.process_pdf_to_flashcards(
        pdf_path,
        output_format=output_format
    )

    # Show sample flashcards
    if output_path:
        print("\nSample flashcards:")
        print("-" * 60)

        if output_format == 'json':
            with open(output_path, 'r') as f:
                flashcards = json.load(f)
        else:
            df = pd.read_csv(output_path)
            flashcards = df.to_dict('records')

        for i, fc in enumerate(flashcards[:3], 1):
            print(f"\nFlashcard {i}:")
            print(f"Q: {fc['question']}")
            print(f"A: {fc['answer']}")


if __name__ == "__main__":
    main()

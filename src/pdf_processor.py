"""
PDF processing module for extracting and chunking text.

This module handles extracting text from PDFs and splitting it into
appropriate chunks for flashcard generation.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
import pdfplumber
from tqdm import tqdm


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file using pdfplumber.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text as a single string
    """
    text = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

    return text


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing noise and normalizing whitespace.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r' +', ' ', text)

    # Remove standalone page numbers (lines with only digits)
    text = re.sub(r'\n\d+\n', '\n', text)

    # Normalize line breaks (multiple newlines → double newline for paragraphs)
    text = re.sub(r'\n\n+', '\n\n', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text_by_paragraphs(
    text: str,
    min_words: int = 30,
    max_words: int = 200
) -> List[str]:
    """
    Split text into paragraph chunks suitable for flashcard generation.

    Strategy:
    1. Split by double newlines (paragraph boundaries)
    2. Filter out chunks that are too short
    3. Split chunks that are too long by sentences

    Args:
        text: Cleaned text from PDF
        min_words: Minimum words per chunk (filter out shorter)
        max_words: Maximum words per chunk (split longer)

    Returns:
        List of text chunks
    """
    # Split by paragraph boundaries
    paragraphs = re.split(r'\n\n+', text)

    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        words = para.split()
        word_count = len(words)

        # Skip very short paragraphs
        if word_count < min_words:
            continue

        # If paragraph is within bounds, add it
        if word_count <= max_words:
            chunks.append(para)
        else:
            # Split long paragraphs by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)

            current_chunk = ""
            current_words = 0

            for sent in sentences:
                sent_words = len(sent.split())

                # If adding this sentence keeps us under max, add it
                if current_words + sent_words <= max_words:
                    current_chunk += sent + " "
                    current_words += sent_words
                else:
                    # Save current chunk if it meets minimum
                    if current_words >= min_words:
                        chunks.append(current_chunk.strip())

                    # Start new chunk with this sentence
                    current_chunk = sent + " "
                    current_words = sent_words

            # Add remaining text if it meets minimum
            if current_words >= min_words:
                chunks.append(current_chunk.strip())

    return chunks


def process_pdf_to_chunks(
    pdf_path: str,
    min_words: int = 30,
    max_words: int = 200
) -> List[Dict[str, any]]:
    """
    Complete pipeline: PDF → cleaned chunks ready for flashcard generation.

    Args:
        pdf_path: Path to PDF file
        min_words: Minimum words per chunk
        max_words: Maximum words per chunk

    Returns:
        List of dictionaries with chunk metadata:
        [
            {
                'chunk_id': 0,
                'source_text': "...",
                'word_count': 150,
                'pdf_file': "path/to/file.pdf"
            },
            ...
        ]
    """
    pdf_path = Path(pdf_path)

    print(f"Processing PDF: {pdf_path.name}")

    # Step 1: Extract text
    print("  [1/3] Extracting text...")
    raw_text = extract_text_from_pdf(str(pdf_path))

    if not raw_text:
        print(f"  Warning: No text extracted from {pdf_path.name}")
        return []

    # Step 2: Clean text
    print("  [2/3] Cleaning text...")
    cleaned_text = clean_text(raw_text)

    # Step 3: Chunk into paragraphs
    print("  [3/3] Chunking text...")
    chunks = chunk_text_by_paragraphs(cleaned_text, min_words, max_words)

    # Step 4: Create structured output
    processed_chunks = []
    for idx, chunk in enumerate(chunks):
        processed_chunks.append({
            'chunk_id': idx,
            'source_text': chunk,
            'word_count': len(chunk.split()),
            'pdf_file': str(pdf_path)
        })

    print(f"  ✓ Extracted {len(processed_chunks)} chunks from {pdf_path.name}")

    return processed_chunks


def process_multiple_pdfs(
    pdf_dir: str,
    min_words: int = 30,
    max_words: int = 200
) -> Dict[str, List[Dict]]:
    """
    Process multiple PDF files from a directory.

    Args:
        pdf_dir: Directory containing PDF files
        min_words: Minimum words per chunk
        max_words: Maximum words per chunk

    Returns:
        Dictionary mapping PDF filenames to their chunks
    """
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return {}

    print(f"Found {len(pdf_files)} PDF files to process\n")

    all_chunks = {}

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        chunks = process_pdf_to_chunks(pdf_file, min_words, max_words)
        all_chunks[pdf_file.name] = chunks

    total_chunks = sum(len(chunks) for chunks in all_chunks.values())
    print(f"\nTotal chunks extracted: {total_chunks}")

    return all_chunks


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        chunks = process_pdf_to_chunks(pdf_path)

        print(f"\nSample chunks:")
        for chunk in chunks[:3]:
            print(f"\n--- Chunk {chunk['chunk_id']} ({chunk['word_count']} words) ---")
            print(chunk['source_text'][:200] + "...")
    else:
        print("Usage: python pdf_processor.py <path_to_pdf>")
        print("\nOr use as a module:")
        print("  from pdf_processor import process_pdf_to_chunks")
        print("  chunks = process_pdf_to_chunks('document.pdf')")

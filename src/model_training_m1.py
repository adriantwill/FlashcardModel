"""
M1-Optimized Training Script for FLAN-T5 Flashcard Generator

This script is optimized for Apple Silicon (M1/M2/M3) Macs.
It uses Metal Performance Shaders (MPS) for GPU acceleration.

Usage:
    # Quick test (10-20 minutes)
    python3 src/model_training_m1.py --max_train_samples 1000 --num_epochs 1

    # Full training (8-12 hours overnight)
    python3 src/model_training_m1.py --num_epochs 3

    # Resume from checkpoint
    python3 src/model_training_m1.py --resume_from_checkpoint ./models/.../checkpoint-3000
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from evaluate import load
import numpy as np
import argparse
from pathlib import Path
import json

# Detect device
if torch.backends.mps.is_available():
    device = "mps"
    device_name = "Apple Silicon GPU (MPS)"
elif torch.cuda.is_available():
    device = "cuda"
    device_name = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
else:
    device = "cpu"
    device_name = "CPU"

print(f"\n{'='*60}")
print(f"DEVICE: {device_name}")
print(f"{'='*60}\n")


def main(args):
    """Main training function."""

    # Verify data exists
    data_dir = Path(args.data_dir)
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_file}\n"
            f"Run: python3 src/data_preparation.py first"
        )

    # Load tokenizer and model
    print("Loading FLAN-T5-base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    print(f"✓ Model loaded ({model.num_parameters():,} parameters)\n")

    # Load datasets
    print("Loading datasets...")
    train_dataset = load_dataset('json', data_files=str(train_file), split='train')
    val_dataset = load_dataset('json', data_files=str(val_file), split='train')

    # Limit dataset size for quick experiments
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
        print(f"✓ Using {len(train_dataset)} training samples (limited for quick iteration)\n")
    else:
        print(f"✓ Using full training set ({len(train_dataset):,} samples)\n")

    print(f"Validation samples: {len(val_dataset):,}\n")

    # Tokenization function
    def preprocess_function(examples):
        """Tokenize inputs and targets."""
        # Tokenize inputs (source text)
        inputs = tokenizer(
            examples['input'],
            max_length=args.max_input_length,
            truncation=True,
            padding='max_length'
        )

        # Tokenize targets (Q&A)
        targets = tokenizer(
            examples['target'],
            max_length=args.max_target_length,
            truncation=True,
            padding='max_length'
        )

        # Model needs labels for training
        inputs['labels'] = targets['input_ids']

        return inputs

    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )

    tokenized_val = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation"
    )
    print("✓ Tokenization complete\n")

    # Load evaluation metrics
    print("Loading evaluation metrics...")
    rouge = load('rouge')
    bleu = load('bleu')

    def compute_metrics(eval_pred):
        """Compute ROUGE and BLEU metrics during evaluation."""
        predictions, labels = eval_pred

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE
        rouge_scores = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )

        # Compute BLEU
        bleu_score = bleu.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )

        return {
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'bleu': bleu_score['bleu']
        }

    print("✓ Metrics loaded\n")

    # Training arguments (M1-optimized)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),

        # Batch sizes (optimized for 16GB M1 Pro)
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,  # Eval needs less memory
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Learning rate and schedule
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        lr_scheduler_type='linear',
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,

        # MPS doesn't fully support fp16 yet
        # Keep fp16=False for stability on M1
        fp16=False,

        # Evaluation and checkpointing
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=3,  # Keep only 3 most recent checkpoints
        load_best_model_at_end=True,
        metric_for_best_model='rougeL',

        # Logging
        logging_steps=args.logging_steps,
        logging_dir=str(output_dir / 'logs'),
        report_to='none',  # Disable wandb/tensorboard

        # Generation settings for evaluation
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=4,
    )

    # Data collator (handles batching and padding)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100  # -100 is ignored in loss computation
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Print training info
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Device: {device_name}")
    print(f"Model: {args.model_name}")
    print(f"Training samples: {len(tokenized_train):,}")
    print(f"Validation samples: {len(tokenized_val):,}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size per device: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # Estimate training time
    steps_per_epoch = len(tokenized_train) // (args.batch_size * args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_epochs
    print(f"\nEstimated steps: {total_steps:,}")
    if device == "mps":
        time_estimate_hours = total_steps * 1.5 / 3600  # Rough estimate
        print(f"Estimated time on M1: {time_estimate_hours:.1f}-{time_estimate_hours*1.5:.1f} hours")
    print()

    # Resume from checkpoint if specified
    last_checkpoint = None
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.resume_from_checkpoint)
        if checkpoint_path.exists():
            last_checkpoint = str(checkpoint_path)
            print(f"✓ Resuming from checkpoint: {last_checkpoint}\n")
        else:
            print(f"⚠ Checkpoint not found: {checkpoint_path}")
            print("Starting from scratch...\n")

    # Start training
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)
    print()

    try:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        # Training complete
        print("\n" + "="*60)
        print("✓ TRAINING COMPLETE!")
        print("="*60)
        print(f"Final training loss: {train_result.training_loss:.4f}")
        print()

        # Save final model
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(str(output_dir))
        print(f"✓ Model saved to: {output_dir}\n")

        # Save training stats
        stats_file = output_dir / "training_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({
                'device': device_name,
                'training_samples': len(tokenized_train),
                'validation_samples': len(tokenized_val),
                'epochs': args.num_epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'final_train_loss': float(train_result.training_loss),
                'total_steps': total_steps,
            }, f, indent=2)
        print(f"✓ Stats saved to: {stats_file}\n")

        # Final evaluation on validation set
        print("="*60)
        print("FINAL EVALUATION")
        print("="*60)
        metrics = trainer.evaluate()

        print(f"\nROUGE-1:  {metrics['eval_rouge1']:.4f}")
        print(f"ROUGE-2:  {metrics['eval_rouge2']:.4f}")
        print(f"ROUGE-L:  {metrics['eval_rougeL']:.4f}")
        print(f"BLEU:     {metrics['eval_bleu']:.4f}")
        print()

        # Interpretation
        if metrics['eval_rougeL'] > 0.5:
            print("✓ Excellent performance (ROUGE-L > 0.5)")
        elif metrics['eval_rougeL'] > 0.4:
            print("✓ Good performance (ROUGE-L > 0.4)")
        elif metrics['eval_rougeL'] > 0.35:
            print("⚠ Decent baseline (ROUGE-L > 0.35)")
        else:
            print("✗ Below target (ROUGE-L < 0.35)")

        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print(f"1. Test generation: python3 src/flashcard_generator.py {output_dir} <pdf_file> json")
        print(f"2. Evaluate quality: Manually review generated flashcards")
        print(f"3. Iterate: Adjust hyperparameters and retrain if needed")
        print("="*60 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print(f"Latest checkpoint saved in: {output_dir}")
        print(f"Resume with: --resume_from_checkpoint {output_dir}")

    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"{type(e).__name__}: {e}")
        print(f"\nCheck DEBUGGING_GUIDE.md for common issues")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train FLAN-T5 for flashcard generation (M1-optimized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-base",
        help="HuggingFace model name"
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing train.jsonl and val.jsonl"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Limit training samples for quick iteration (e.g., 1000 for testing)"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=512,
        help="Maximum input sequence length"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="Maximum target sequence length"
    )

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size (4 works well on M1 Pro 16GB)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * this)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps"
    )

    # Checkpointing
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/flan-t5-flashcard-v1",
        help="Output directory for model and checkpoints"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every N steps"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint directory"
    )

    args = parser.parse_args()

    # Run training
    main(args)

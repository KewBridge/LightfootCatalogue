"""
Evaluation Script for Catalogue Transcription

This script evaluates the quality of catalogue transcription by comparing
predicted outputs against gold standard data using various metrics.
"""

import os
import argparse
import pandas as pd
import unicodedata
from typing import Tuple, Set
from jiwer import wer, cer
from rouge_score import rouge_scorer

from lightcat.utils import get_logger
logger = get_logger(__name__)


def clean_text(text, is_species: bool = False) -> str:
    """
    Clean and normalize text for comparison.
    
    Args:
        text: Input text to clean
        is_species: If True, only keep first 3 words for species names
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return text
    
    text = unicodedata.normalize("NFKD", str(text)).strip().lower().replace("\\", "")
    
    if is_species:
        return " ".join(text.split(" ")[:3])
    
    return text


def compute_set_metrics(gold_set: Set, eval_set: Set) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score for two sets.
    
    Args:
        gold_set: Gold standard set
        eval_set: Evaluation/predicted set
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    tp = len(gold_set & eval_set)
    fp = len(eval_set - gold_set)
    fn = len(gold_set - eval_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def evaluate_species_names(df_gold: pd.DataFrame, df_eval: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Evaluate species name extraction.
    
    Args:
        df_gold: Gold standard dataframe
        df_eval: Evaluation dataframe
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    species_gold = set(df_gold["species_name"].dropna().unique())
    species_eval = set(df_eval["species_name"].dropna().unique())
    
    return compute_set_metrics(species_gold, species_eval)


def evaluate_family_names(df_gold: pd.DataFrame, df_eval: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Evaluate family-species pair extraction.
    
    Args:
        df_gold: Gold standard dataframe
        df_eval: Evaluation dataframe
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    family_gold = set(tuple(x) for x in df_gold[["family_name", "species_name"]].dropna().values)
    family_eval = set(tuple(x) for x in df_eval[["family_name", "species_name"]].dropna().values)
    
    return compute_set_metrics(family_gold, family_eval)


def evaluate_descriptions(df_gold: pd.DataFrame, df_eval: pd.DataFrame) -> dict:
    """
    Evaluate description quality using multiple metrics.
    
    Args:
        df_gold: Gold standard dataframe
        df_eval: Evaluation dataframe
        
    Returns:
        Dictionary containing WER, CER, and ROUGE metrics
    """
    # Find matching (family, species) pairs
    family_gold = set(tuple(x) for x in df_gold[["family_name", "species_name"]].dropna().values)
    family_eval = set(tuple(x) for x in df_eval[["family_name", "species_name"]].dropna().values)
    pairs = family_gold & family_eval
    
    gold_desc = []
    eval_desc = []
    
    for pair in pairs:
        fam_, sp_ = pair[0], pair[1]
        
        desc_gold = df_gold[
            (df_gold["family_name"] == fam_) & (df_gold["species_name"] == sp_)
        ]["description"].values
        
        desc_eval = df_eval[
            (df_eval["family_name"] == fam_) & (df_eval["species_name"] == sp_)
        ]["description"].values
        
        if len(desc_gold) == 0 or len(desc_eval) == 0:
            continue
            
        desc_gold = desc_gold[0]
        desc_eval = desc_eval[0]
        
        if pd.isna(desc_gold) or pd.isna(desc_eval):
            continue
            
        gold_desc.append(desc_gold)
        eval_desc.append(desc_eval)
    
    if len(gold_desc) == 0:
        return {
            "wer": 0.0,
            "cer": 0.0,
            "rouge_precision": 0.0,
            "rouge_recall": 0.0,
            "rouge_f1": 0.0,
            "num_pairs": 0
        }
    
    # Compute WER and CER
    wer_score = wer(gold_desc, eval_desc)
    cer_score = cer(gold_desc, eval_desc)
    
    # Compute ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for g, e in zip(gold_desc, eval_desc):
        score = scorer.score(g, e)["rougeL"]
        precision_scores.append(score.precision)
        recall_scores.append(score.recall)
        f1_scores.append(score.fmeasure)
    
    return {
        "wer": wer_score,
        "cer": cer_score,
        "rouge_precision": sum(precision_scores) / len(precision_scores),
        "rouge_recall": sum(recall_scores) / len(recall_scores),
        "rouge_f1": sum(f1_scores) / len(f1_scores),
        "num_pairs": len(gold_desc)
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate catalogue transcription quality against gold standard"
    )
    parser.add_argument(
        "gold_csv",
        type=str,
        help="Path to gold standard CSV file"
    )
    parser.add_argument(
        "eval_csv",
        type=str,
        help="Path to evaluation/predicted CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as CSV"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input files
    if not os.path.exists(args.gold_csv):
        logger.error(f"Error: Gold standard file not found: {args.gold_csv}")
        return 1
    
    if not os.path.exists(args.eval_csv):
        logger.error(f"Error: Evaluation file not found: {args.eval_csv}")
        return 1
    
    # Load data
    logger.info(f"Loading gold standard: {args.gold_csv}")
    df_gold = pd.read_csv(args.gold_csv)
    
    logger.info(f"Loading evaluation data: {args.eval_csv}")
    df_eval = pd.read_csv(args.eval_csv)
    
    # Clean data
    logger.info("Cleaning and normalizing data...")
    species_clean = lambda x: clean_text(x, is_species=True)
    
    df_gold["family_name"] = df_gold["family_name"].map(clean_text)
    df_gold["species_name"] = df_gold["species_name"].map(species_clean)
    df_gold["description"] = df_gold["description"].map(clean_text)
    
    df_eval["family_name"] = df_eval["family_name"].map(clean_text)
    df_eval["species_name"] = df_eval["species_name"].map(species_clean)
    df_eval["description"] = df_eval["description"].map(clean_text)
    
    # Evaluate species names
    logger.info("Evaluating species names...")
    sp_precision, sp_recall, sp_f1 = evaluate_species_names(df_gold, df_eval)
    
    # Evaluate family-species pairs
    logger.info("Evaluating family-species pairs...")
    fam_precision, fam_recall, fam_f1 = evaluate_family_names(df_gold, df_eval)
    
    # Evaluate descriptions
    logger.info("Evaluating descriptions...")
    desc_metrics = evaluate_descriptions(df_gold, df_eval)
    
    # Print results
    logger.info("="*70)
    logger.info("EVALUATION RESULTS")
    logger.info("="*70)
    logger.info(f"Gold standard records: {len(df_gold)}")
    logger.info(f"Evaluation records: {len(df_eval)}")
    logger.info("")
    
    logger.info("--- Species Name Extraction ---")
    logger.info(f"Precision: {sp_precision:.4f}")
    logger.info(f"Recall:    {sp_recall:.4f}")
    logger.info(f"F1 Score:  {sp_f1:.4f}")
    logger.info("")
    logger.info("--- Family-Species Pair Extraction ---")
    logger.info(f"Precision: {fam_precision:.4f}")
    logger.info(f"Recall:    {fam_recall:.4f}")
    logger.info(f"F1 Score:  {fam_f1:.4f}")
    logger.info("")
    logger.info("--- Description Quality ---")
    logger.info(f"Matching pairs evaluated:   {desc_metrics['num_pairs']}")
    logger.info(f"WER (Word Error Rate):      {desc_metrics['wer']:.4f}")
    logger.info(f"CER (Character Error Rate): {desc_metrics['cer']:.4f}")
    logger.info(f"ROUGE-L Precision:          {desc_metrics['rouge_precision']:.4f}")
    logger.info(f"ROUGE-L Recall:             {desc_metrics['rouge_recall']:.4f}")
    logger.info(f"ROUGE-L F1:                 {desc_metrics['rouge_f1']:.4f}")
    logger.info("="*70)
    
    # Save results if requested
    if args.output:
        results_df = pd.DataFrame({
            "Metric": [
                "Species_Precision", "Species_Recall", "Species_F1",
                "Family_Precision", "Family_Recall", "Family_F1",
                "Description_WER", "Description_CER",
                "Description_ROUGE_Precision", "Description_ROUGE_Recall", "Description_ROUGE_F1",
                "Description_Pairs_Evaluated"
            ],
            "Value": [
                sp_precision, sp_recall, sp_f1,
                fam_precision, fam_recall, fam_f1,
                desc_metrics['wer'], desc_metrics['cer'],
                desc_metrics['rouge_precision'], desc_metrics['rouge_recall'], desc_metrics['rouge_f1'],
                desc_metrics['num_pairs']
            ]
        })
        
        results_df.to_csv(args.output, index=False)
        logger.info(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()

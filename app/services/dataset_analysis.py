import logging
import random
import math
from typing import Any, Dict, List, Optional
import os

import numpy as np
from datasets import load_dataset_builder, get_dataset_config_names, load_dataset, concatenate_datasets
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)

class DatasetAnalysisService:
    def __init__(self):
        self.embedding_model = None
        self.tokenizers = {} # Cache tokenizers
        # Simple local caching for dataset loading to avoid constant re-downloads during interleaving
        self._dataset_cache = {}

    def _get_embedding_model(self):
        if self.embedding_model is None:
            logger.info("Loading sentence-transformer model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.embedding_model

    def _get_tokenizer(self, target_model: str):
        if target_model not in self.tokenizers:
             try:
                 # Fast tokenizer if available, avoid auth errors
                 self.tokenizers[target_model] = AutoTokenizer.from_pretrained(target_model, token=os.environ.get("HF_TOKEN"))
             except Exception as e:
                 logger.warning(f"Could not load tokenizer for {target_model}: {e}. Falling back to default.")
                 self.tokenizers[target_model] = None
        return self.tokenizers[target_model]

    def _estimate_perplexity(self, texts: List[str], target_model: str) -> float:
        """
        Calculates a pseudo-perplexity or length penalty if local LM isn't viable.
        For a true 360 analysis, we simulate perplexity based on text complexity 
        (vocab richness) to avoid heavy local inference, representing 'data predictability'.
        """
        # Simulated perplexity metric based on vocabulary richness
        # In a real heavy prod environment, call an Inference API endpoint here.
        all_words = " ".join(texts).split()
        if not all_words: return 0.0
        vocab = set(all_words)
        # Type-token ratio as a proxy for complexity (inverted for "predictability/perplexity")
        ttr = len(vocab) / len(all_words) 
        # Scale to a standard PPL-like range (e.g., 5-50)
        simulated_ppl = max(5.0, min(50.0, 10 + (ttr * 50)))
        return float(simulated_ppl)

    def _calculate_holistic_score(self, metrics: Dict) -> Dict:
        """Generates a 0-100 rating on how suitable the dataset combo is."""
        score = 100.0
        deductions = []
        
        # 1. Diversity Penalty
        diversity = metrics.get('semantic_metrics', {}).get('diversity_score', 0.5)
        if diversity < 0.2:
            score -= 20
            deductions.append("Very low semantic diversity (repetitive data).")
        elif diversity < 0.4:
            score -= 10
            deductions.append("Below average semantic diversity.")
            
        # 2. Size Penalty
        total_rows = metrics.get('total_rows', 0)
        if total_rows < 1000:
            score -= 30
            deductions.append("Dataset is too small for effective generalized SLM fine-tuning.")
        elif total_rows < 10000:
            score -= 10
            deductions.append("Dataset is on the smaller side. Consider augmenting.")
            
        # 3. Format Penalty
        has_fmt = metrics.get('has_instruction_format')
        task_detected = metrics.get('detected_task_type', 'unknown')
        if not has_fmt:
             score -= 25
             deductions.append("Lacks clear Instruction/Output structure or alignment keys (chosen/rejected or ground_truth). May require formatting.")
        else:
             deductions.append(f"Format looks great! Detected alignment task format: {task_detected.upper()}.")

        # 4. Sequence Length Limit Penalty (Target Model)
        max_len = metrics.get('sequence_metrics', {}).get('max_length', 0)
        avg_len = metrics.get('sequence_metrics', {}).get('avg_length', 0)
        
        # Assume typical target SLMs want a mix. Too short is bad, too long might truncate
        if avg_len < 20:
             score -= 15
             deductions.append("Average sequence length is extremely short. May result in model generating terse responses.")
             
        # 5. Missing Characteristics Check
        missing = []
        # Simulate checking for 'code' or 'math' blocks
        if metrics.get('sequence_metrics', {}).get('avg_length', 0) < 100:
             missing.append("Long-form reasoning or essay generation data.")
        if diversity > 0.7:
             missing.append("Domain-specific focused data (data is very broad).")
             
        metrics['missing_characteristics'] = missing
        metrics['rating_score'] = max(0.0, min(100.0, score))
        metrics['rating_deductions'] = deductions
        
        return metrics

    async def analyze_datasets(self, dataset_configs: List[Dict], target_model: str) -> Dict[str, Any]:
        """
        Analyzes one or multiple Hugging Face datasets for fine-tuning suitability.
        dataset_configs: List of dicts like [{"id": "imdb", "config": "plain_text", "split": "train"}]
        """
        try:
            logger.info(f"Analyzing {len(dataset_configs)} datasets. Target model: {target_model}")
            
            combined_samples = []
            combined_info = []
            total_rows = 0
            
            # --- 1. Fetch & Interleave Datasets ---
            for cfg in dataset_configs:
                d_id = cfg.get("id")
                config_name = cfg.get("config", "default")
                if not config_name: config_name = "default"
                split = cfg.get("split", "train")
                
                # Fetch Configs if None
                if config_name == "default":
                     try:
                         configs = get_dataset_config_names(d_id)
                         config_name = configs[0] if configs else "default"
                     except Exception:
                         pass # fallback
                 
                try:
                    builder = load_dataset_builder(d_id, config_name)
                    info = builder.info
                    
                    if split not in info.splits:
                         split = list(info.splits.keys())[0] if info.splits else "train"
                         
                    split_info = info.splits.get(split)
                    split_rows = split_info.num_examples if split_info else 0
                    total_rows += split_rows
                    
                    combined_info.append({
                        "dataset_id": d_id,
                        "config": config_name,
                        "split": split,
                        "rows": split_rows
                    })
                    
                    # Stream sample
                    dataset_stream = load_dataset(d_id, config_name, split=split, streaming=True)
                    # Take proportional samples based on uniform sampling (simplification: 100 per dataset)
                    sample_size = min(100, split_rows if split_rows > 0 else 100)
                    samples = list(dataset_stream.take(sample_size))
                    
                    # Tag origin
                    for s in samples:
                         s['_source_dataset_'] = d_id
                    combined_samples.extend(samples)
                    
                except Exception as e:
                     logger.warning(f"Failed to load sub-dataset {d_id}: {e}")
                     return {"error": f"Failed to load dataset '{d_id}': {str(e)}"}
            
            if not combined_samples:
                return {"error": "All specified datasets appear to be empty or could not be loaded."}

            analysis_result = {
                "analyzed_datasets": combined_info,
                "total_rows": total_rows,
                "target_model": target_model,
                "recommendations": [],
                "semantic_metrics": {},
                "sequence_metrics": {}
            }

            # --- 2. Analyze Column Structure & Content across interleaved data ---
            columns = combined_samples[0].keys()
            analysis_result["columns"] = [c for c in columns if c != '_source_dataset_']
            
            text_columns = [col for col in columns if isinstance(combined_samples[0][col], str) and col != '_source_dataset_']
            
            has_instruction = any(col.lower() in ["instruction", "prompt", "question"] for col in columns)
            has_output = any(col.lower() in ["output", "response", "answer", "completion"] for col in columns)
            has_chosen = "chosen" in columns
            has_rejected = "rejected" in columns
            has_ground_truth = any(col.lower() in ["ground_truth", "rubric", "target"] for col in columns)

            detected_task = "sft"
            if has_chosen and has_rejected:
                 detected_task = "dpo"
                 analysis_result["has_instruction_format"] = True
                 analysis_result["recommendations"].append("Dataset has chosen & rejected preference splits. Perfectly formatted for DPO (Direct Preference Optimization).")
            elif has_ground_truth:
                 detected_task = "grpo"
                 analysis_result["has_instruction_format"] = True
                 analysis_result["recommendations"].append("Dataset has ground_truth rubrics. Perfect for GRPO (Group Relative Policy Optimization) or RLVR.")
            elif has_instruction and has_output:
                 detected_task = "sft"
                 analysis_result["has_instruction_format"] = True
                 analysis_result["recommendations"].append("Dataset has standard prompt/completion or question/answer format suitable for SFT (Supervised Fine-Tuning).")
            else:
                 detected_task = "pretrain"
                 analysis_result["has_instruction_format"] = False
                 analysis_result["recommendations"].append("Dataset lacks direct task key splits. Suitable for generic language model pre-training.")

            analysis_result["detected_task_type"] = detected_task

            # Identify target text for deep analysis
            target_col = None
            if text_columns:
                target_col = max(text_columns, key=lambda col: len(str(combined_samples[0][col])))
                texts = [s[target_col] for s in combined_samples if s[target_col]]
            else:
                 texts = []

            # --- 3. Sequence Length & Tokenization Analysis ---
            if texts and target_model:
                tokenizer = self._get_tokenizer(target_model)
                if tokenizer:
                    lengths = [len(tokenizer.encode(t)) for t in texts]
                else:
                    # Fallback approximation (words * ~1.3 tokens)
                    lengths = [int(len(t.split()) * 1.3) for t in texts]
                    
                analysis_result["sequence_metrics"] = {
                    "avg_length": float(np.mean(lengths)),
                    "max_length": int(np.max(lengths)),
                    "min_length": int(np.min(lengths)),
                    "estimated": tokenizer is None
                }

            # --- 4. Semantic Analysis & Clustering ---
            if texts:
                model = self._get_embedding_model()
                embeddings = model.encode(texts)
                
                # Diversity
                sim_matrix = cosine_similarity(embeddings)
                np.fill_diagonal(sim_matrix, np.nan) 
                avg_similarity = np.nanmean(sim_matrix)
                diversity_score = 1.0 - avg_similarity
                
                # Basic simulated outlier metrics
                # (Distance to centroid)
                centroid = np.mean(embeddings, axis=0)
                centroid_sim = cosine_similarity(embeddings, [centroid])
                outlier_ratio = np.sum(centroid_sim < 0.2) / len(texts) # arbitrarily low sim
                
                analysis_result["semantic_metrics"] = {
                    "analyzed_column": target_col,
                    "sample_size": len(texts),
                    "diversity_score": float(diversity_score),
                    "avg_similarity": float(avg_similarity),
                    "outlier_ratio": float(outlier_ratio)
                }
                
                # --- 5. Perplexity ---
                ppl = self._estimate_perplexity(texts, target_model)
                analysis_result["perplexity_score"] = ppl

            # --- 6. Holistic Rating & Missing Characteristics ---
            analysis_result = self._calculate_holistic_score(analysis_result)

            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing dataset combo: {e}")
            return {"error": str(e)}

dataset_analysis_service = DatasetAnalysisService()

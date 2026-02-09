"""
MelisCelikkolMaster.py

Master Thesis: Ambiguity Resolution Across German Words and Their Sign Correspondents

3rd December 2025

Melis Çelikkol

Institut für Computerlinguistik
Ruprecht-Karls-Universität Heidelberg

Supervisor: Professor Dr. Katja Markert
External Supervisor: Dr. Wei Zhao
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import warnings
import json
import yaml
from itertools import product
from tqdm import tqdm
from datetime import datetime
import zipfile
import pickle

warnings.filterwarnings('ignore')


# checkpoint manager

class CheckpointManager:
    """Manage checkpoints for long-running optimizations"""

    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, name, data):
        """Save checkpoint data"""
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f" Checkpoint saved: {name}")

    def load_checkpoint(self, name):
        """Load checkpoint data"""
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f" Checkpoint loaded: {name}")
            return data
        return None

    def checkpoint_exists(self, name):
        """Check if checkpoint exists"""
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        return os.path.exists(path)


# device configuration

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    return device

DEVICE = get_device()


# configuration and data loading

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file, create default if not existing"""
    if not os.path.exists(config_path):
        print(f"Config file not found, creating default: {config_path}")
        default_config = {
            'paths': {
                'word_video': '/content/ManualAnnotation_dataset.csv',
                'meaning_video': '/content/videoid_meaning_dataset_base.csv',
                'meaning_video_gt': '/content/videoid_meaning_dataset_gt.csv',
                'meaning_video_test': '/content/videoid_meaning_dataset_test.csv',
                'test_set': '/content/TestSet.csv',
                'pathway_ground_truth': '/content/pathway_ground_truth.csv',
                'pathway_ground_truth_test': '/content/pathway_ground_truth_test.csv',
                'results': 'thesis_results'
            },
            'hyperparameters': {
                'similarity_threshold_range': [0.65, 0.70, 0.75, 0.80],
                'top_k_semantic_range': [3, 5, 7],
                'batch_size': 64
            },
            'optimization': {
                'optimization_size': 100,
                'validation_size': 150,
                'random_state': 42,
            },
            'models': {
                'embedding_models': [
                    'aari1995/German_Semantic_STS_V2',
                    'sentence-transformers/all-MiniLM-L6-v2',
                    'T-Systems-onsite/german-roberta-sentence-transformer-v2',
                    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                ]
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        return default_config

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded from {config_path}")
    return config


def load_csv_with_encoding(file_path, data_type):
    """Load CSV with simple handling"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            df = pd.read_csv(file_path, encoding='latin-1')

        df.columns = df.columns.str.strip()
        print(f"Columns found: {list(df.columns)}")
        df.columns = df.columns.str.lower()

        if data_type == 'word_video':
            if 'word' in df.columns and 'video_id' in df.columns:
                print(f"Successfully loaded {data_type}: {len(df)} rows")
                return df
            else:
                raise RuntimeError(f"Missing required columns. Need 'word' and 'video_id'. Found: {list(df.columns)}")

        else:  # meaning_video
            if 'meaning' in df.columns and 'video_id' in df.columns:
                df.rename(columns={'meaning': 'Meaning'}, inplace=True)
                print(f"Successfully loaded {data_type}: {len(df)} rows")
                return df
            else:
                raise RuntimeError(f"Missing required columns. Need 'meaning' and 'video_id'. Found: {list(df.columns)}")

    except Exception as e:
        raise RuntimeError(f"Could not load {data_type} from {file_path}: {str(e)}")


def create_results_zip(results_dir='thesis_results', output_filename=None):
    """Create a zip file of the entire results directory"""
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"thesis_results_{timestamp}.zip"

    print(f"\nCreating zip archive: {output_filename}")

    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(results_dir))
                zipf.write(file_path, arcname)
                print(f"  Added: {arcname}")

    file_size = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"\nZip file created: {output_filename} ({file_size:.2f} MB)")

    return output_filename


# cross-modal predictor

class CrossModalPredictor:
    def __init__(self, word_video_path, meaning_video_path, embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
                 hyperparams=None, device=None, exclude_indices=None, ablation_mode='baseline'):
        self.model_name = embedding_model
        self.device = device if device else DEVICE
        self.ablation_mode = ablation_mode  # 'baseline', 'no_word', 'no_sentence'

        self.hyperparams = {
            'similarity_threshold': 0.70,
            'top_k_semantic': 5,
            'batch_size': 64
        }
        if hyperparams:
            self.hyperparams.update(hyperparams)

        print(f"Loading data files (ablation mode: {ablation_mode})...")
        self.df_word_video = load_csv_with_encoding(word_video_path, 'word_video')
        self.df_meaning_video = load_csv_with_encoding(meaning_video_path, 'meaning_video')

        if exclude_indices is not None:
            print(f"Excluding {len(exclude_indices)} examples from searchable data (optimization/validation set)")
            self.df_word_video = self.df_word_video[~self.df_word_video.index.isin(exclude_indices)]
            print(f"Remaining searchable examples: {len(self.df_word_video)}")

        self.validate_datasets()

        print(f"Loading model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model, device=self.device)

        self._generate_embeddings()
        self._analyze_data_characteristics()

    def validate_datasets(self):
        """Validate that datasets require structure and content"""
        print("Validating datasets...")

        assert len(self.df_word_video) > 0, "Empty word_video dataset"
        assert 'word' in self.df_word_video.columns, "Missing 'word' column"
        assert 'video_id' in self.df_word_video.columns, "Missing 'video_id' column"

        assert len(self.df_meaning_video) > 0, "Empty meaning_video dataset"
        assert 'Meaning' in self.df_meaning_video.columns, "Missing 'Meaning' column"
        assert 'video_id' in self.df_meaning_video.columns, "Missing 'video_id' column"

        word_valid_count = self.df_word_video['word'].notna().sum()
        meaning_valid_count = self.df_meaning_video['Meaning'].notna().sum()

        assert word_valid_count > 0, "No valid words"
        assert meaning_valid_count > 0, "No valid meanings"

        print(f"Validation passed: {word_valid_count} words, {meaning_valid_count} meanings")

    def _generate_embeddings(self):
        print(f"Generating embeddings (ablation mode: {self.ablation_mode})...")

        self.df_word_video['word'] = self.df_word_video['word'].astype(str).str.lower().str.strip()
        self.df_word_video['sentence'] = self.df_word_video['sentence'].fillna('').astype(str)

        self.df_word_video['is_no_match'] = (
            self.df_word_video['video_id'].isna() |
            (self.df_word_video['video_id'].astype(str).str.strip() == '') |
            (self.df_word_video['video_id'].astype(str).str.strip() == '0') |
            (self.df_word_video['video_id'] == 0)
        )

        # ABLATION 1: Modify context_text based on ablation mode
        if self.ablation_mode == 'baseline':
            # Normal: word + sentence
            self.df_word_video['context_text'] = self.df_word_video.apply(
                lambda row: f"{row['word']} {row['sentence']}", axis=1
            )
        elif self.ablation_mode == 'no_word':
            # Only sentence
            self.df_word_video['context_text'] = self.df_word_video['sentence']
        elif self.ablation_mode == 'no_sentence':
            # Only word
            self.df_word_video['context_text'] = self.df_word_video['word']
        else:
            raise ValueError(f"Unknown ablation mode: {self.ablation_mode}")

        self.df_word_video = self.df_word_video.dropna(subset=['context_text'])
        self.df_meaning_video = self.df_meaning_video.dropna(subset=['Meaning'])

        valid_entries = self.df_word_video[~self.df_word_video['is_no_match']]
        texts_wv = valid_entries['context_text'].tolist()

        self.df_word_video['embedding'] = pd.Series(dtype=object)

        if len(texts_wv) > 0:
            embeddings_wv = self.model.encode(
                texts_wv, batch_size=self.hyperparams['batch_size'],
                show_progress_bar=True, convert_to_numpy=True, device=self.device
            )
            for idx, emb in zip(valid_entries.index, embeddings_wv):
                self.df_word_video.at[idx, 'embedding'] = emb

        self.df_meaning_video['Meaning'] = self.df_meaning_video['Meaning'].fillna('').astype(str)
        texts_m = self.df_meaning_video['Meaning'].tolist()
        embeddings_m = self.model.encode(
            texts_m, batch_size=self.hyperparams['batch_size'],
            show_progress_bar=True, convert_to_numpy=True, device=self.device
        )
        self.df_meaning_video['embedding'] = [emb for emb in embeddings_m]

        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def _analyze_data_characteristics(self):
        no_match_count = self.df_word_video['is_no_match'].sum()
        valid_match_count = len(self.df_word_video) - no_match_count

        self.dataset_stats = {
            'total_word_video_pairs': len(self.df_word_video),
            'valid_matches': valid_match_count,
            'no_matches': no_match_count,
            'unique_words': self.df_word_video['word'].nunique(),
            'unique_videos_word': self.df_word_video['video_id'].nunique(),
            'unique_videos_meaning': self.df_meaning_video['video_id'].nunique(),
            'total_meanings': len(self.df_meaning_video)
        }
        print("Dataset characteristics:")
        for key, value in self.dataset_stats.items():
            print(f"  {key}: {value}")

    def exact_match_prediction(self, word, sentence):
        """Exact match: Direct lookup in word-video mappings"""
        word_lower = word.lower().strip()

        # Find exact matches in word-video dataset
        exact_matches = self.df_word_video[
            (self.df_word_video['word'] == word_lower) &
            (~self.df_word_video['is_no_match'])
        ].copy()

        if len(exact_matches) == 0:
            return {'success': False, 'predictions': []}

        # ABLATION 1: Modify query text based on ablation mode for confidence calculation
        if self.ablation_mode == 'baseline':
            query_text = f"{word_lower} {sentence.lower()}"
        elif self.ablation_mode == 'no_word':
            query_text = sentence.lower()
        elif self.ablation_mode == 'no_sentence':
            query_text = word_lower
        else:
            query_text = f"{word_lower} {sentence.lower()}"

        query_embedding = self.model.encode(query_text, convert_to_numpy=True, device=self.device)

        # Get unique video IDs from exact matches and calculate confidence
        predictions = []
        seen_videos = set()

        for _, row in exact_matches.iterrows():
            video_id = str(row['video_id'])
            if video_id not in seen_videos:
                seen_videos.add(video_id)

                # Calculate confidence using semantic similarity with the matched context
                context_embedding = row['embedding']
                confidence = cosine_similarity([query_embedding], [context_embedding])[0][0]

                predictions.append({
                    'video_id': video_id,
                    'confidence': float(confidence),
                    'method': 'exact_match',
                    'matched_word': word_lower,
                    'source_sentence': row['sentence']
                })

        return {
            'success': True,
            'predictions': sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        }

    def semantic_similarity_prediction(self, word, sentence):
        """Semantic similarity: Find signs through semantic understanding"""
        word_lower = word.lower().strip()

        # ABLATION 1: Modify query text based on ablation mode
        if self.ablation_mode == 'baseline':
            query_text = f"{word_lower} {sentence.lower()}"
        elif self.ablation_mode == 'no_word':
            query_text = sentence.lower()
        elif self.ablation_mode == 'no_sentence':
            query_text = word_lower
        else:
            query_text = f"{word_lower} {sentence.lower()}"

        query_embedding = self.model.encode(query_text, convert_to_numpy=True, device=self.device)

        all_predictions = []

        searchable_entries = self.df_word_video[self.df_word_video['embedding'].notna()]

        if len(searchable_entries) > 0:
            corpus_embeddings_wv = np.vstack(searchable_entries['embedding'].values)
            similarities_wv = cosine_similarity([query_embedding], corpus_embeddings_wv)[0]

            top_indices_wv = np.argsort(similarities_wv)[-self.hyperparams['top_k_semantic']:][::-1]
            for idx in top_indices_wv:
                if similarities_wv[idx] > self.hyperparams['similarity_threshold']:
                    row = searchable_entries.iloc[idx]
                    all_predictions.append({
                        'video_id': row['video_id'],
                        'confidence': float(similarities_wv[idx]),
                        'method': 'semantic_word_video',
                        'source_sentence': row['sentence'],
                        'source_word': row['word']
                    })

        corpus_embeddings_m = np.vstack(self.df_meaning_video['embedding'].values)
        similarities_m = cosine_similarity([query_embedding], corpus_embeddings_m)[0]

        top_indices_m = np.argsort(similarities_m)[-self.hyperparams['top_k_semantic']:][::-1]
        for idx in top_indices_m:
            if similarities_m[idx] > self.hyperparams['similarity_threshold']:
                row = self.df_meaning_video.iloc[idx]
                all_predictions.append({
                    'video_id': row['video_id'],
                    'confidence': float(similarities_m[idx]),
                    'method': 'semantic_meaning',
                    'source_meaning': row['Meaning']
                })

        return {
            'success': len(all_predictions) > 0,
            'predictions': sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
        }

    def predict(self, word, sentence, method='both'):
        if method == 'exact_match':
            return self.exact_match_prediction(word, sentence)
        elif method == 'semantic':
            return self.semantic_similarity_prediction(word, sentence)
        else:
            exact_result = self.exact_match_prediction(word, sentence)
            semantic_result = self.semantic_similarity_prediction(word, sentence)

            return {
                'exact_match': exact_result,
                'semantic': semantic_result,
                'word': word,
                'sentence': sentence
            }


# hyperparameter optimizer

class HyperparameterOptimizer:
    def __init__(self, word_video_path, meaning_video_path, results_dir='thesis_results'):
        self.word_video_path = word_video_path
        self.meaning_video_path = meaning_video_path
        self.results_dir = results_dir
        self.optimization_dir = os.path.join(results_dir, 'optimization')
        self.checkpoint_manager = CheckpointManager()

        os.makedirs(self.optimization_dir, exist_ok=True)

    def create_three_way_split(self, df_word_video, optimization_size=100, validation_size=150, random_state=42):
        """Create three-way split: Training pool + Optimization set + Validation set"""
        print("\n" + "♡☆" * 40)
        print("Undertaking three-way split...")
        print("♡☆" * 40)
        print("Split structure:")
        print("  1. Training Pool    - Searchable data for predictions")
        print("  2. Optimization Set - For hyperparameter tuning")
        print("  3. Validation Set   - For model comparison and analysis")
        print("Note: External test set will be loaded separately for final evaluation")

        df_word_video['is_no_match'] = (
            df_word_video['video_id'].isna() |
            (df_word_video['video_id'].astype(str).str.strip() == '') |
            (df_word_video['video_id'].astype(str).str.strip() == '0') |
            (df_word_video['video_id'] == 0)
        )

        match_cases = df_word_video[~df_word_video['is_no_match']].copy()
        no_match_cases = df_word_video[df_word_video['is_no_match']].copy()

        match_cases = match_cases.sample(frac=1, random_state=random_state).reset_index(drop=True)
        no_match_cases = no_match_cases.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Calculate splits (70% match, 30% no-match for each set)
        opt_matches = int(optimization_size * 0.7)
        opt_no_matches = optimization_size - opt_matches

        val_matches = int(validation_size * 0.7)
        val_no_matches = validation_size - val_matches

        # Ensure we don't exceed available data
        opt_matches = min(opt_matches, len(match_cases) // 5)
        opt_no_matches = min(opt_no_matches, len(no_match_cases) // 5)
        val_matches = min(val_matches, len(match_cases) // 4)
        val_no_matches = min(val_no_matches, len(no_match_cases) // 4)

        # Create optimization set
        opt_match_set = match_cases.iloc[:opt_matches]
        opt_nomatch_set = no_match_cases.iloc[:opt_no_matches]

        # Create validation set (after optimization set)
        val_match_set = match_cases.iloc[opt_matches:opt_matches + val_matches]
        val_nomatch_set = no_match_cases.iloc[opt_no_matches:opt_no_matches + val_no_matches]

        # Remaining data is training pool
        train_match_set = match_cases.iloc[opt_matches + val_matches:]
        train_nomatch_set = no_match_cases.iloc[opt_no_matches + val_no_matches:]

        def convert_to_eval_format(match_df, nomatch_df):
            data = []
            for _, row in match_df.iterrows():
                data.append({
                    'word': row['word'].lower(),
                    'sentence': row['sentence'],
                    'true_video_id': str(row['video_id'])
                })
            for _, row in nomatch_df.iterrows():
                data.append({
                    'word': row['word'].lower(),
                    'sentence': row['sentence'],
                    'true_video_id': 'NO_MATCH'
                })
            return data

        optimization_data = convert_to_eval_format(opt_match_set, opt_nomatch_set)
        validation_data = convert_to_eval_format(val_match_set, val_nomatch_set)

        print(f"\n Three-way split created:")
        print(f"  Training Pool:    {len(train_match_set) + len(train_nomatch_set)} examples (searchable)")
        print(f"  Optimization Set: {len(optimization_data)} examples (for hyperparameter tuning)")
        print(f"  Validation Set:   {len(validation_data)} examples (for model comparison)")

        return {
            'optimization': optimization_data,
            'optimization_indices': list(opt_match_set.index) + list(opt_nomatch_set.index),
            'validation': validation_data,
            'validation_indices': list(val_match_set.index) + list(val_nomatch_set.index),
            'train_indices': list(train_match_set.index) + list(train_nomatch_set.index)
        }

    def load_external_test_set(self, test_set_path):
        """Load external test set"""
        print("\n" + "♡☆" * 40)
        print("Generalization test set being loaded...")
        print("♡☆" * 40)

        if not os.path.exists(test_set_path):
            raise FileNotFoundError(f"Test set not found: {test_set_path}")

        df_test = load_csv_with_encoding(test_set_path, 'word_video')

        df_test['is_no_match'] = (
            df_test['video_id'].isna() |
            (df_test['video_id'].astype(str).str.strip() == '') |
            (df_test['video_id'].astype(str).str.strip() == '0') |
            (df_test['video_id'] == 0)
        )

        test_data = []
        for _, row in df_test.iterrows():
            if row['is_no_match']:
                test_data.append({
                    'word': row['word'].lower(),
                    'sentence': row['sentence'],
                    'true_video_id': 'NO_MATCH'
                })
            else:
                test_data.append({
                    'word': row['word'].lower(),
                    'sentence': row['sentence'],
                    'true_video_id': str(row['video_id'])
                })

        n_matches = len(df_test[~df_test['is_no_match']])
        n_no_matches = len(df_test[df_test['is_no_match']])

        print(f"\n External test set loaded: {len(test_data)} cases")
        print(f"  Matches: {n_matches} ({n_matches/len(test_data)*100:.1f}%)")
        print(f"  No-matches: {n_no_matches} ({n_no_matches/len(test_data)*100:.1f}%)")
        print("♡☆" * 40)

        return test_data

    def optimize_for_model(self, optimization_data, model_name, exclude_indices=None,
                          param_grid=None):
        """Optimize hyperparameters for a specific model using optimization set"""
        print("\n" + "♡☆" * 40)
        print(f"HYPERPARAMETER OPTIMIZATION FOR: {model_name}")
        print(f"Using dedicated optimization set ({len(optimization_data)} examples)")
        print("♡☆" * 40)

        checkpoint_name = f"gridsearch_{model_name.replace('/', '_')}"

        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_name)
        if checkpoint:
            print("\n   Resuming from checkpoint!")
            results = checkpoint['results']
            completed_configs = checkpoint['completed_configs']
            best_score = checkpoint['best_score']
            best_config = checkpoint['best_config']
            start_idx = len(results)
            print(f"  Already completed: {start_idx} configurations")
        else:
            results = []
            completed_configs = set()
            best_score = -float('inf')
            best_config = None
            start_idx = 0

        param_combinations = [
            dict(zip(param_grid.keys(), v))
            for v in product(*param_grid.values())
        ]

        total_configs = len(param_combinations)
        print(f"\n Total configurations: {total_configs}")
        print(f"   Remaining: {total_configs - start_idx}")

        pbar = tqdm(
            param_combinations[start_idx:],
            desc=f"Optimizing {model_name}",
            initial=start_idx,
            total=total_configs
        )

        for i, params in enumerate(pbar, start=start_idx):
            params_key = json.dumps(params, sort_keys=True)
            if params_key in completed_configs:
                continue

            try:
                scores = self.evaluate_hyperparams(
                    optimization_data, params, model_name,
                    exclude_indices=exclude_indices
                )

                result = {
                    'trial': i + 1,
                    'similarity_threshold': params['similarity_threshold'],
                    'top_k_semantic': params['top_k_semantic'],
                    'batch_size': params['batch_size'],
                    'exact_accuracy': scores['exact_accuracy'],
                    'semantic_accuracy': scores['semantic_accuracy'],
                    'improvement': scores['improvement']
                }
                results.append(result)
                completed_configs.add(params_key)

                if scores['semantic_accuracy'] > best_score:
                    best_score = scores['semantic_accuracy']
                    best_config = params.copy()
                    no_improvement_count = 0
                    pbar.set_postfix({
                        'best': f"{best_score:.2f}%",
                        'current': f"{scores['semantic_accuracy']:.2f}%"
                    })

                if (i + 1) % 3 == 0:
                    self.checkpoint_manager.save_checkpoint(checkpoint_name, {
                        'results': results,
                        'completed_configs': completed_configs,
                        'best_score': best_score,
                        'best_config': best_config
                    })


            except KeyboardInterrupt:
                print("\n\n  Interrupted by user!")
                print("  Saving progress...")
                self.checkpoint_manager.save_checkpoint(checkpoint_name, {
                    'results': results,
                    'completed_configs': completed_configs,
                    'best_score': best_score,
                    'best_config': best_config
                })
                print(" Progress saved. Run again to resume from this point.")
                raise

            except Exception as e:
                print(f"\n  Error in trial {i+1}: {str(e)}")
                continue

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('semantic_accuracy', ascending=False)

        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        df_results.to_csv(f'{self.optimization_dir}/hyperparameter_optimization_{safe_model_name}.csv', index=False)

        best_result = df_results.iloc[0]
        with open(f'{self.optimization_dir}/best_hyperparameters_{safe_model_name}.json', 'w') as f:
            json.dump({
                'model': model_name,
                'hyperparameters': best_config,
                'optimization_performance': {
                    'exact_accuracy': float(best_result['exact_accuracy']),
                    'semantic_accuracy': float(best_result['semantic_accuracy']),
                    'improvement': float(best_result['improvement'])
                },
                'total_trials': len(results),
                'data_leakage_prevention': exclude_indices is not None
            }, f, indent=2)

        print(f"\n Optimization complete for {model_name}!")
        print(f"  Best config: threshold={best_config['similarity_threshold']}, top_k={best_config['top_k_semantic']}")
        print(f"  Best accuracy: {best_score:.2f}%")

        return best_config, df_results

    def evaluate_hyperparams(self, test_data, hyperparams, model_name, exclude_indices=None, ablation_mode='baseline'):
        """Evaluate hyperparameters"""
        try:
            predictor = CrossModalPredictor(
                self.word_video_path,
                self.meaning_video_path,
                embedding_model=model_name,
                hyperparams=hyperparams,
                exclude_indices=exclude_indices,
                ablation_mode=ablation_mode
            )

            correct_exact = 0
            correct_semantic = 0

            for case in test_data:
                word = case['word']
                sentence = case['sentence']
                true_video = case['true_video_id']
                is_no_match = (true_video == 'NO_MATCH')

                exact_result = predictor.predict(word, sentence, method='exact_match')
                semantic_result = predictor.predict(word, sentence, method='semantic')

                if is_no_match:
                    if not exact_result['success']:
                        correct_exact += 1
                    if not semantic_result['success']:
                        correct_semantic += 1
                else:
                    if exact_result['success'] and str(exact_result['predictions'][0]['video_id']) == str(true_video):
                        correct_exact += 1
                    if semantic_result['success'] and str(semantic_result['predictions'][0]['video_id']) == str(true_video):
                        correct_semantic += 1

            exact_accuracy = (correct_exact / len(test_data)) * 100
            semantic_accuracy = (correct_semantic / len(test_data)) * 100

            del predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                'exact_accuracy': exact_accuracy,
                'semantic_accuracy': semantic_accuracy,
                'improvement': semantic_accuracy - exact_accuracy
            }

        except Exception as e:
            print(f"Error evaluating hyperparams: {str(e)}")
            return {
                'exact_accuracy': 0.0,
                'semantic_accuracy': 0.0,
                'improvement': 0.0,
                'error': str(e)
            }


# ablation test analysis (Test 1: Word/Sentence columns)

class AblationTestAnalyzer:
    def __init__(self, word_video_path, meaning_video_path, results_dir='thesis_results'):
        self.word_video_path = word_video_path
        self.meaning_video_path = meaning_video_path
        self.results_dir = results_dir
        self.ablation_dir = os.path.join(results_dir, 'ablation_test_1_columns')

        os.makedirs(self.ablation_dir, exist_ok=True)

    def run_ablation_test(self, validation_data, best_model_name, best_hyperparams,
                         pathway_df, exclude_indices):
        """Run ablation test with three configurations: baseline, no_word, no_sentence"""
        print("\n" + "♡☆" * 40)
        print("ABLATION TEST 1: Input Column Impact Analysis")
        print("♡☆" * 40)
        print(f"Model: {best_model_name}")
        print(f"Validation set: {len(validation_data)} examples")
        print("\nTesting three configurations:")
        print("  1. BASELINE: word + sentence (current successful setup)")
        print("  2. NO_WORD: sentence only")
        print("  3. NO_SENTENCE: word only")

        ablation_modes = ['baseline', 'no_word', 'no_sentence']
        all_results = {}

        for mode in ablation_modes:
            print(f"\n{'='*60}")
            print(f"Running ablation mode: {mode.upper()}")
            print(f"{'='*60}")

            # Create predictor with specific ablation mode
            predictor = CrossModalPredictor(
                self.word_video_path,
                self.meaning_video_path,
                embedding_model=best_model_name,
                hyperparams=best_hyperparams,
                exclude_indices=exclude_indices,
                ablation_mode=mode
            )

            # Evaluate
            results = self._evaluate_ablation_mode(
                validation_data,
                predictor,
                pathway_df,
                mode
            )

            all_results[mode] = results

            # Cleanup
            del predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Compare results
        self._generate_ablation_comparison(all_results, best_model_name)

        return all_results

    def _evaluate_ablation_mode(self, validation_data, predictor, pathway_df, mode):
        """Evaluate a single ablation mode"""
        results = []

        for case in tqdm(validation_data, desc=f"Evaluating {mode}"):
            word = case['word']
            sentence = case['sentence']
            true_video = case['true_video_id']
            is_no_match = (true_video == 'NO_MATCH')

            # Get pathway type
            pathway_info = pathway_df[pathway_df['word'] == word]
            if len(pathway_info) > 0:
                pathway_type = pathway_info['pathway_type'].iloc[0]
            else:
                pathway_type = 'unknown'

            # Get predictions
            exact_result = predictor.predict(word, sentence, method='exact_match')
            semantic_result = predictor.predict(word, sentence, method='semantic')

            # Evaluate correctness
            if is_no_match:
                exact_correct = not exact_result['success']
                semantic_correct = not semantic_result['success']
                exact_confidence = 0.0
                semantic_confidence = 0.0
            else:
                exact_correct = False
                exact_confidence = 0.0
                if exact_result['success']:
                    best_exact = exact_result['predictions'][0]
                    exact_correct = (str(best_exact['video_id']) == str(true_video))
                    exact_confidence = best_exact['confidence']

                semantic_correct = False
                semantic_confidence = 0.0
                if semantic_result['success']:
                    predictions = semantic_result['predictions']
                    max_confidence = predictions[0]['confidence']
                    for pred in predictions:
                        if pred['confidence'] == max_confidence:
                            if str(pred['video_id']) == str(true_video):
                                semantic_correct = True
                                semantic_confidence = pred['confidence']
                                break
                        else:
                            break

                    if not semantic_correct and len(predictions) > 0:
                        semantic_confidence = predictions[0]['confidence']

            results.append({
                'word': word,
                'pathway_type': pathway_type,
                'is_no_match': is_no_match,
                'exact_correct': exact_correct,
                'semantic_correct': semantic_correct,
                'exact_confidence': exact_confidence,
                'semantic_confidence': semantic_confidence
            })

        df_results = pd.DataFrame(results)

        # Save detailed results
        safe_mode = mode.replace('_', '')
        df_results.to_csv(
            f'{self.ablation_dir}/detailed_results_{safe_mode}.csv',
            index=False
        )

        # Calculate metrics
        metrics = {
            'mode': mode,
            'exact_accuracy': float(df_results['exact_correct'].mean() * 100),
            'semantic_accuracy': float(df_results['semantic_correct'].mean() * 100),
            'improvement': float((df_results['semantic_correct'].mean() -
                                 df_results['exact_correct'].mean()) * 100),
            'avg_exact_confidence': float(df_results['exact_confidence'].mean()),
            'avg_semantic_confidence': float(df_results['semantic_confidence'].mean()),
            'total_cases': len(df_results)
        }

        # Pathway-wise performance
        pathway_performance = df_results.groupby('pathway_type').agg({
            'exact_correct': 'mean',
            'semantic_correct': 'mean'
        }).round(4) * 100

        # Convert to the expected dictionary structure
        metrics['pathway_performance'] = pathway_performance.to_dict()

        return metrics

    def _generate_ablation_comparison(self, all_results, model_name):
        """Generate comparison tables and visualizations"""
        print(f"\n{'='*60}")
        print("ABLATION TEST 1 RESULTS SUMMARY")
        print(f"{'='*60}")

        # Overall comparison
        comparison_data = []
        for mode, metrics in all_results.items():
            comparison_data.append({
                'configuration': mode,
                'exact_accuracy': metrics['exact_accuracy'],
                'semantic_accuracy': metrics['semantic_accuracy'],
                'improvement': metrics['improvement'],
                'avg_semantic_confidence': metrics['avg_semantic_confidence']
            })

        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('semantic_accuracy', ascending=False)

        # Save overall comparison
        df_comparison.to_csv(f'{self.ablation_dir}/ablation_comparison_summary.csv', index=False)

        print("\nOverall Performance:")
        print(df_comparison.to_string(index=False))

        # Calculate impact metrics
        baseline_accuracy = all_results['baseline']['semantic_accuracy']
        no_word_accuracy = all_results['no_word']['semantic_accuracy']
        no_sentence_accuracy = all_results['no_sentence']['semantic_accuracy']

        word_impact = baseline_accuracy - no_word_accuracy
        sentence_impact = baseline_accuracy - no_sentence_accuracy

        impact_summary = {
            'baseline_semantic_accuracy': baseline_accuracy,
            'no_word_semantic_accuracy': no_word_accuracy,
            'no_sentence_semantic_accuracy': no_sentence_accuracy,
            'word_column_impact': word_impact,
            'sentence_column_impact': sentence_impact,
            'interpretation': {
                'word_impact': 'positive' if word_impact > 0 else 'negative',
                'sentence_impact': 'positive' if sentence_impact > 0 else 'negative',
                'more_important_column': 'word' if abs(word_impact) > abs(sentence_impact) else 'sentence'
            }
        }

        with open(f'{self.ablation_dir}/impact_analysis.json', 'w') as f:
            json.dump(impact_summary, f, indent=2)

        print(f"\n{'='*60}")
        print("Input Column Impact Analysis")
        print(f"{'='*60}")
        print(f"Baseline (word + sentence):     {baseline_accuracy:.2f}%")
        print(f"Without word column:            {no_word_accuracy:.2f}%")
        print(f"Without sentence column:        {no_sentence_accuracy:.2f}%")
        print(f"\nImpact of WORD column:          {word_impact:+.2f}%")
        print(f"Impact of SENTENCE column:      {sentence_impact:+.2f}%")
        print(f"\nMost important column: {impact_summary['interpretation']['more_important_column'].upper()}")

        # Pathway-wise comparison
        self._generate_pathway_comparison(all_results)

        print(f"\n Ablation test 1 complete")
        print(f"  Results saved to: {self.ablation_dir}/")

    def _generate_pathway_comparison(self, all_results):
        """Generate pathway-wise comparison across ablation modes"""
        pathway_comparisons = []

        for mode, metrics in all_results.items():
            if 'pathway_performance' in metrics:
                pathway_perf = metrics['pathway_performance']
                if 'exact_correct' in pathway_perf and 'semantic_correct' in pathway_perf:
                    for pathway_type in pathway_perf['exact_correct'].keys():
                        exact_acc = pathway_perf['exact_correct'][pathway_type]
                        semantic_acc = pathway_perf['semantic_correct'][pathway_type]
                        pathway_comparisons.append({
                            'configuration': mode,
                            'pathway_type': pathway_type,
                            'exact_accuracy': exact_acc,
                            'semantic_accuracy': semantic_acc,
                            'improvement': semantic_acc - exact_acc
                        })

        if pathway_comparisons:
            df_pathway = pd.DataFrame(pathway_comparisons)

            # Pivot for easier comparison
            pivot_semantic = df_pathway.pivot(
                index='pathway_type',
                columns='configuration',
                values='semantic_accuracy'
            ).round(2)

            pivot_semantic.to_csv(f'{self.ablation_dir}/pathway_comparison_semantic_accuracy.csv')

            print(f"\nPathway-wise Semantic Accuracy (%):")
            print(pivot_semantic.to_string())


# ablation test 2: Meaning dataset comparison

class MeaningDatasetAblationAnalyzer:
    def __init__(self, word_video_path, results_dir='thesis_results'):
        self.word_video_path = word_video_path
        self.results_dir = results_dir
        self.ablation_dir = os.path.join(results_dir, 'ablation_test_2_meaning_dataset')

        os.makedirs(self.ablation_dir, exist_ok=True)

    def run_meaning_dataset_ablation(self, validation_data, best_model_name, best_hyperparams,
                                     pathway_df, exclude_indices,
                                     meaning_video_base_path, meaning_video_gt_path):
        """Compare performance using base vs gt meaning datasets"""
        print("\n" + "♡☆" * 40)
        print("ABLATION TEST 2: Meaning Dataset Comparison")
        print("♡☆" * 40)
        print(f"Model: {best_model_name}")
        print(f"Validation set: {len(validation_data)} examples")
        print("\nComparing two meaning datasets:")
        print(f"  1. BASE: {meaning_video_base_path}")
        print(f"  2. GT:   {meaning_video_gt_path}")
        print("\nThis tests the impact of different meaning descriptions.")

        dataset_configs = [
            {'name': 'base', 'path': meaning_video_base_path},
            {'name': 'gt', 'path': meaning_video_gt_path}
        ]

        all_results = {}

        for config in dataset_configs:
            dataset_name = config['name']
            dataset_path = config['path']

            print(f"\n{'='*60}")
            print(f"Evaluating with {dataset_name.upper()} meaning dataset")
            print(f"{'='*60}")

            # Create predictor with specific meaning dataset
            predictor = CrossModalPredictor(
                self.word_video_path,
                dataset_path,
                embedding_model=best_model_name,
                hyperparams=best_hyperparams,
                exclude_indices=exclude_indices,
                ablation_mode='baseline'
            )

            # Evaluate
            results = self._evaluate_meaning_dataset(
                validation_data,
                predictor,
                pathway_df,
                dataset_name
            )

            all_results[dataset_name] = results

            # Cleanup
            del predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Compare results
        self._generate_dataset_comparison(all_results, best_model_name)

        return all_results

    def _evaluate_meaning_dataset(self, validation_data, predictor, pathway_df, dataset_name):
        """Evaluate using a specific meaning dataset"""
        results = []

        for case in tqdm(validation_data, desc=f"Evaluating {dataset_name}"):
            word = case['word']
            sentence = case['sentence']
            true_video = case['true_video_id']
            is_no_match = (true_video == 'NO_MATCH')

            # Get pathway type
            pathway_info = pathway_df[pathway_df['word'] == word]
            if len(pathway_info) > 0:
                pathway_type = pathway_info['pathway_type'].iloc[0]
            else:
                pathway_type = 'unknown'

            # Get predictions
            exact_result = predictor.predict(word, sentence, method='exact_match')
            semantic_result = predictor.predict(word, sentence, method='semantic')

            # Evaluate correctness
            if is_no_match:
                exact_correct = not exact_result['success']
                semantic_correct = not semantic_result['success']
                exact_confidence = 0.0
                semantic_confidence = 0.0
            else:
                exact_correct = False
                exact_confidence = 0.0
                if exact_result['success']:
                    best_exact = exact_result['predictions'][0]
                    exact_correct = (str(best_exact['video_id']) == str(true_video))
                    exact_confidence = best_exact['confidence']

                semantic_correct = False
                semantic_confidence = 0.0
                if semantic_result['success']:
                    predictions = semantic_result['predictions']
                    max_confidence = predictions[0]['confidence']
                    for pred in predictions:
                        if pred['confidence'] == max_confidence:
                            if str(pred['video_id']) == str(true_video):
                                semantic_correct = True
                                semantic_confidence = pred['confidence']
                                break
                        else:
                            break

                    if not semantic_correct and len(predictions) > 0:
                        semantic_confidence = predictions[0]['confidence']

            results.append({
                'word': word,
                'pathway_type': pathway_type,
                'is_no_match': is_no_match,
                'exact_correct': exact_correct,
                'semantic_correct': semantic_correct,
                'exact_confidence': exact_confidence,
                'semantic_confidence': semantic_confidence
            })

        df_results = pd.DataFrame(results)

        # Save detailed results
        df_results.to_csv(
            f'{self.ablation_dir}/detailed_results_{dataset_name}.csv',
            index=False
        )

        # Calculate metrics
        metrics = {
            'dataset': dataset_name,
            'exact_accuracy': float(df_results['exact_correct'].mean() * 100),
            'semantic_accuracy': float(df_results['semantic_correct'].mean() * 100),
            'improvement': float((df_results['semantic_correct'].mean() -
                                 df_results['exact_correct'].mean()) * 100),
            'avg_exact_confidence': float(df_results['exact_confidence'].mean()),
            'avg_semantic_confidence': float(df_results['semantic_confidence'].mean()),
            'total_cases': len(df_results)
        }

        # Pathway-wise performance
        pathway_performance = df_results.groupby('pathway_type').agg({
            'exact_correct': 'mean',
            'semantic_correct': 'mean'
        }).round(4) * 100

        metrics['pathway_performance'] = pathway_performance.to_dict()

        return metrics

    def _generate_dataset_comparison(self, all_results, model_name):
        """Generate comparison between base and gt datasets"""
        print(f"\n{'='*60}")
        print("Ablation Test 2 Results - Summary")
        print(f"{'='*60}")

        # Overall comparison
        comparison_data = []
        for dataset_name, metrics in all_results.items():
            comparison_data.append({
                'meaning_dataset': dataset_name,
                'exact_accuracy': metrics['exact_accuracy'],
                'semantic_accuracy': metrics['semantic_accuracy'],
                'improvement': metrics['improvement'],
                'avg_semantic_confidence': metrics['avg_semantic_confidence']
            })

        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('semantic_accuracy', ascending=False)

        # Save overall comparison
        df_comparison.to_csv(f'{self.ablation_dir}/dataset_comparison_summary.csv', index=False)

        print("\nOverall Performance by Meaning Dataset:")
        print(df_comparison.to_string(index=False))

        # Calculate impact metrics
        base_accuracy = all_results['base']['semantic_accuracy']
        gt_accuracy = all_results['gt']['semantic_accuracy']

        dataset_impact = gt_accuracy - base_accuracy

        impact_summary = {
            'base_semantic_accuracy': base_accuracy,
            'gt_semantic_accuracy': gt_accuracy,
            'gt_vs_base_difference': dataset_impact,
            'better_dataset': 'gt' if dataset_impact > 0 else 'base',
            'interpretation': {
                'impact': 'positive' if dataset_impact > 0 else 'negative',
                'magnitude': abs(dataset_impact),
                'conclusion': f"GT dataset performs {abs(dataset_impact):.2f}% {'better' if dataset_impact > 0 else 'worse'} than BASE"
            }
        }

        with open(f'{self.ablation_dir}/dataset_impact_analysis.json', 'w') as f:
            json.dump(impact_summary, f, indent=2)

        print(f"\n{'='*60}")
        print("Meaning Dataset Impact Analysis")
        print(f"{'='*60}")
        print(f"BASE dataset semantic accuracy: {base_accuracy:.2f}%")
        print(f"GT dataset semantic accuracy:   {gt_accuracy:.2f}%")
        print(f"\nDifference (GT - BASE):         {dataset_impact:+.2f}%")
        print(f"Better performing dataset:      {impact_summary['better_dataset'].upper()}")
        print(f"\nConclusion: {impact_summary['interpretation']['conclusion']}")

        # Pathway-wise comparison
        self._generate_pathway_dataset_comparison(all_results)

        print(f"\n Ablation test 2 complete")
        print(f"  Results saved to: {self.ablation_dir}/")

    def _generate_pathway_dataset_comparison(self, all_results):
        """Generate pathway-wise comparison across datasets"""
        pathway_comparisons = []

        for dataset_name, metrics in all_results.items():
            if 'pathway_performance' in metrics:
                pathway_perf = metrics['pathway_performance']
                if 'exact_correct' in pathway_perf and 'semantic_correct' in pathway_perf:
                    for pathway_type in pathway_perf['exact_correct'].keys():
                        exact_acc = pathway_perf['exact_correct'][pathway_type]
                        semantic_acc = pathway_perf['semantic_correct'][pathway_type]
                        pathway_comparisons.append({
                            'dataset': dataset_name,
                            'pathway_type': pathway_type,
                            'exact_accuracy': exact_acc,
                            'semantic_accuracy': semantic_acc,
                            'improvement': semantic_acc - exact_acc
                        })

        if pathway_comparisons:
            df_pathway = pd.DataFrame(pathway_comparisons)

            # Pivot for easier comparison
            pivot_semantic = df_pathway.pivot(
                index='pathway_type',
                columns='dataset',
                values='semantic_accuracy'
            ).round(2)

            pivot_semantic.to_csv(f'{self.ablation_dir}/pathway_comparison_by_dataset.csv')

            print(f"\nPathway-wise Semantic Accuracy by Dataset (%):")
            print(pivot_semantic.to_string())


# pathway analyzer

class PathwayAnalyzer:
    def __init__(self, results_dir='thesis_results'):
        self.results_dir = results_dir
        self.supporting_data_dir = os.path.join(results_dir, 'supporting_data')
        self.metadata_dir = os.path.join(results_dir, 'metadata')

        os.makedirs(self.supporting_data_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def load_ground_truth_pathways(self, pathway_ground_truth_path):
        """Load ground truth pathways from separate human-annotated file"""
        print("\n" + "♡☆" * 40)
        print("Ground Truth Pathways loading...")
        print("♡☆" * 40)

        if not os.path.exists(pathway_ground_truth_path):
            raise FileNotFoundError(f"Pathway ground truth file not found: {pathway_ground_truth_path}")

        df_pathways = pd.read_csv(pathway_ground_truth_path)

        df_pathways.columns = df_pathways.columns.str.strip().str.lower()

        required_cols = ['word', 'pathway_type', 'video_id']
        missing_cols = [col for col in required_cols if col not in df_pathways.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in pathway ground truth: {missing_cols}")

        df_pathways['word'] = df_pathways['word'].astype(str).str.lower().str.strip()
        df_pathways['video_id'] = df_pathways['video_id'].astype(str)
        df_pathways['pathway_type'] = df_pathways['pathway_type'].astype(str).str.strip()

        print(f" Loaded {len(df_pathways)} pathway annotations")
        print(f"  Unique words: {df_pathways['word'].nunique()}")
        print(f"\nPathway distribution (ground truth):")
        print(df_pathways['pathway_type'].value_counts())

        df_pathways.to_csv(f'{self.supporting_data_dir}/ground_truth_pathways.csv', index=False)
        self._create_pathway_summary(df_pathways, suffix='_ground_truth')

        return df_pathways

    def discover_pathways_from_predictions(self, test_data, predictor, model_name):
        """Discover pathways based on MODEL's predictions"""
        print(f"\n" + "♡☆" * 40)
        print(f"DISCOVERING Pathways from {model_name.upper()} Predictions")
        print("♡☆" * 40)
        print("Model will independently classify pathways based on its own matches...")
        print(f"Using model's optimized threshold: {predictor.hyperparams['similarity_threshold']}")

        model_mappings = []

        for case in tqdm(test_data, desc=f"Model pathway discovery ({model_name})"):
            word = case['word']
            sentence = case['sentence']

            result = predictor.predict(word, sentence, method='semantic')

            if result['success']:
                predictions = result['predictions']
                max_confidence = predictions[0]['confidence']

                for pred in predictions:
                    if pred['confidence'] == max_confidence:
                        model_mappings.append({
                            'word': word,
                            'video_id': pred['video_id'],
                            'confidence': pred['confidence'],
                            'sentence': sentence
                        })
                    else:
                        break
            else:
                model_mappings.append({
                    'word': word,
                    'video_id': 'NO_MATCH',
                    'confidence': 0.0,
                    'sentence': sentence
                })

        df_model = pd.DataFrame(model_mappings)

        df_model['is_no_match'] = (df_model['video_id'] == 'NO_MATCH')

        pathway_analysis = []

        for word in df_model['word'].unique():
            word_data = df_model[df_model['word'] == word]

            valid_videos = word_data[~word_data['is_no_match']]['video_id'].unique()
            n_videos_for_word = len(valid_videos)

            if n_videos_for_word == 0:
                pathway_type = "No Match"
            elif n_videos_for_word > 1:
                pathway_type = "Pathway 1"
            else:
                video_id = valid_videos[0]

                video_meanings = predictor.df_meaning_video[
                    predictor.df_meaning_video['video_id'] == str(video_id)
                ]

                if len(video_meanings) > 0:
                    word_contexts = word_data[word_data['video_id'] == video_id]

                    avg_similarities = []
                    for _, word_row in word_contexts.iterrows():
                        query_text = f"{word_row['word']} {word_row['sentence']}"
                        query_emb = predictor.model.encode(query_text, convert_to_numpy=True)

                        for _, meaning_row in video_meanings.iterrows():
                            meaning_emb = meaning_row['embedding']
                            sim = cosine_similarity([query_emb], [meaning_emb])[0][0]
                            avg_similarities.append(sim)

                    if len(avg_similarities) > 0 and np.mean(avg_similarities) < predictor.hyperparams['similarity_threshold']:
                        pathway_type = "Pathway 2"
                    else:
                        pathway_type = "Pathway 3"
                else:
                    pathway_type = "Pathway 3"

            pathway_analysis.append({
                'word': word,
                'model_pathway_type': pathway_type,
                'model_video_count': n_videos_for_word,
                'model_name': model_name
            })

        df_model_pathways = pd.DataFrame(pathway_analysis)

        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        df_model_pathways.to_csv(
            f'{self.supporting_data_dir}/model_discovered_pathways_{safe_model_name}.csv',
            index=False
        )

        print(f" Model pathway discovery complete: {len(df_model_pathways)} unique words classified")
        print(f"  Pathway distribution from model's PoV:")
        print(df_model_pathways['model_pathway_type'].value_counts())

        return df_model_pathways

    def compare_pathway_classifications(self, ground_truth_df, model_pathway_df, model_name):
        """Compare ground truth pathways vs model-discovered pathways"""
        print(f"\n" + "♡☆" * 40)
        print(f"Pathway Classification Comparison: GROUND TRUTH vs {model_name.upper()}")
        print("♡☆" * 40)

        comparison = ground_truth_df[['word', 'pathway_type']].drop_duplicates('word').merge(
            model_pathway_df[['word', 'model_pathway_type']],
            on='word',
            how='inner'
        )

        comparison.columns = ['word', 'ground_truth_pathway', 'model_pathway']

        comparison['pathways_agree'] = (
            comparison['ground_truth_pathway'] == comparison['model_pathway']
        )

        agreement_rate = comparison['pathways_agree'].mean() * 100

        confusion = pd.crosstab(
            comparison['ground_truth_pathway'],
            comparison['model_pathway'],
            margins=True
        )

        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        comparison.to_csv(
            f'{self.metadata_dir}/pathway_comparison_{safe_model_name}.csv',
            index=False
        )
        confusion.to_csv(
            f'{self.metadata_dir}/pathway_confusion_matrix_{safe_model_name}.csv'
        )

        summary = {
            'model_name': model_name,
            'total_words_compared': len(comparison),
            'pathway_agreement_rate': float(agreement_rate),
            'agreements': int(comparison['pathways_agree'].sum()),
            'disagreements': int((~comparison['pathways_agree']).sum()),
            'pathway_wise_agreement': {}
        }

        for pathway_type in ['Pathway 1', 'Pathway 2', 'Pathway 3', 'No Match']:
            subset = comparison[comparison['ground_truth_pathway'] == pathway_type]
            if len(subset) > 0:
                pathway_agreement = (subset['pathways_agree'].sum() / len(subset)) * 100
                summary['pathway_wise_agreement'][pathway_type] = float(pathway_agreement)

        with open(f'{self.metadata_dir}/pathway_comparison_summary_{safe_model_name}.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n Pathway Classification Comparison:")
        print(f"  Overall Agreement: {agreement_rate:.1f}%")
        print(f"  Matches: {summary['agreements']}/{summary['total_words_compared']}")
        print(f"\n  Pathway-wise Agreement:")
        for pathway, agreement in summary['pathway_wise_agreement'].items():
            print(f"    {pathway}: {agreement:.1f}%")

        print(f"\n Pathway comparison saved")

        return comparison, confusion, summary

    def _create_pathway_summary(self, df_pathways, suffix=''):
        """Create summary statistics for pathway distribution"""
        summary = df_pathways.groupby('pathway_type').agg({
            'word': 'count'
        }).reset_index()

        summary.columns = ['Pathway Type', 'Count']
        summary['Percentage'] = (summary['Count'] / len(df_pathways) * 100).round(1)

        pathway_order = ['Pathway 1', 'Pathway 2', 'Pathway 3', 'No Match']
        summary['pathway_order'] = summary['Pathway Type'].map({p: i for i, p in enumerate(pathway_order)})
        summary = summary.sort_values('pathway_order').drop('pathway_order', axis=1)

        summary.to_csv(f'{self.metadata_dir}/pathway_distribution_summary{suffix}.csv', index=False)

        print(f"\nPathway Distribution{suffix}:")
        print(summary.to_string(index=False))


# error flow analyzer

class ErrorFlowAnalyzer:
    def __init__(self, results_dir='thesis_results'):
        self.results_dir = results_dir
        self.core_analysis_dir = os.path.join(results_dir, 'core_analysis')
        self.supporting_data_dir = os.path.join(results_dir, 'supporting_data')

        os.makedirs(self.core_analysis_dir, exist_ok=True)
        os.makedirs(self.supporting_data_dir, exist_ok=True)

    def analyze_error_flows(self, results_df):
        print(f"\nAnalyzing error flows...")

        pathway_errors = results_df.groupby('pathway_type').agg({
            'exact_match_correct': ['sum', 'count', 'mean'],
            'semantic_correct': ['sum', 'count', 'mean']
        }).round(3)

        pathway_errors.columns = ['_'.join(col) for col in pathway_errors.columns]
        pathway_errors['exact_error_rate'] = (1 - pathway_errors['exact_match_correct_mean']) * 100
        pathway_errors['semantic_error_rate'] = (1 - pathway_errors['semantic_correct_mean']) * 100
        pathway_errors['error_reduction'] = pathway_errors['exact_error_rate'] - pathway_errors['semantic_error_rate']

        pathway_errors = pathway_errors.reset_index()

        exact_only_fails = results_df[(~results_df['exact_match_correct']) & (results_df['semantic_correct'])]
        semantic_only_fails = results_df[(results_df['exact_match_correct']) & (~results_df['semantic_correct'])]
        both_fail = results_df[(~results_df['exact_match_correct']) & (~results_df['semantic_correct'])]
        both_succeed = results_df[(results_df['exact_match_correct']) & (results_df['semantic_correct'])]

        failure_modes = pd.DataFrame([{
            'failure_mode': 'Exact only fails',
            'count': len(exact_only_fails),
            'percentage': len(exact_only_fails) / len(results_df) * 100
        }, {
            'failure_mode': 'Semantic only fails',
            'count': len(semantic_only_fails),
            'percentage': len(semantic_only_fails) / len(results_df) * 100
        }, {
            'failure_mode': 'Both fail',
            'count': len(both_fail),
            'percentage': len(both_fail) / len(results_df) * 100
        }, {
            'failure_mode': 'Both succeed',
            'count': len(both_succeed),
            'percentage': len(both_succeed) / len(results_df) * 100
        }])

        pathway_errors.to_csv(f'{self.core_analysis_dir}/error_flow_by_pathway.csv', index=False)
        failure_modes.to_csv(f'{self.core_analysis_dir}/error_flow_failure_modes.csv', index=False)

        print(f" Error flow analysis saved")

        return {
            'pathway_errors': pathway_errors,
            'failure_modes': failure_modes
        }


# precision@k analyzer

class PrecisionAtKAnalyzer:
    def __init__(self, results_dir='thesis_results'):
        self.results_dir = results_dir
        self.core_analysis_dir = os.path.join(results_dir, 'core_analysis')
        self.supporting_data_dir = os.path.join(results_dir, 'supporting_data')

        os.makedirs(self.core_analysis_dir, exist_ok=True)
        os.makedirs(self.supporting_data_dir, exist_ok=True)

    def calculate_precision_at_k(self, test_data, predictor, pathway_df, k_values=[1, 3, 5, 10]):
        print(f"\nCalculating Precision@K...")

        results = []
        detailed_results = []

        for i, case in enumerate(tqdm(test_data, desc="Precision@K")):
            word = case['word']
            sentence = case['sentence']
            true_video = case['true_video_id']
            is_no_match = (true_video == 'NO_MATCH')

            pathway_info = pathway_df[pathway_df['word'] == word]
            if len(pathway_info) > 0:
                pathway_type = pathway_info['pathway_type'].iloc[0]
            else:
                pathway_type = 'unknown'

            exact_result = predictor.predict(word, sentence, method='exact_match')
            semantic_result = predictor.predict(word, sentence, method='semantic')

            exact_precision_at_k = {}
            semantic_precision_at_k = {}

            for k in k_values:
                if is_no_match:
                    exact_precision_at_k[k] = 1 if not exact_result['success'] else 0
                    semantic_precision_at_k[k] = 1 if not semantic_result['success'] else 0
                else:
                    exact_found = False
                    if exact_result['success']:
                        top_k_exact = exact_result['predictions'][:k]
                        exact_found = any(str(pred['video_id']) == str(true_video) for pred in top_k_exact)
                    exact_precision_at_k[k] = 1 if exact_found else 0

                    semantic_found = False
                    if semantic_result['success']:
                        top_k_semantic = semantic_result['predictions'][:k]
                        semantic_found = any(str(pred['video_id']) == str(true_video) for pred in top_k_semantic)
                    semantic_precision_at_k[k] = 1 if semantic_found else 0

            detailed_result = {
                'test_id': i,
                'word': word,
                'true_video': true_video,
                'is_no_match': is_no_match,
                'pathway_type': pathway_type
            }
            for k in k_values:
                detailed_result[f'exact_P@{k}'] = exact_precision_at_k[k]
                detailed_result[f'semantic_P@{k}'] = semantic_precision_at_k[k]
            detailed_results.append(detailed_result)

            results.append({
                'test_id': i,
                'is_no_match': is_no_match,
                **{f'exact_P@{k}': exact_precision_at_k[k] for k in k_values},
                **{f'semantic_P@{k}': semantic_precision_at_k[k] for k in k_values}
            })

        df_results = pd.DataFrame(results)
        df_detailed = pd.DataFrame(detailed_results)

        overall_summary = self._create_precision_summary(df_results, k_values, "Overall")
        match_results = df_results[~df_results['is_no_match']]
        match_summary = self._create_precision_summary(match_results, k_values, "Match Cases")
        nomatch_results = df_results[df_results['is_no_match']]
        nomatch_summary = self._create_precision_summary(nomatch_results, k_values, "No-Match Cases")

        combined_summary = pd.concat([overall_summary, match_summary, nomatch_summary], ignore_index=True)

        combined_summary.to_csv(f'{self.core_analysis_dir}/precision_at_k_summary.csv', index=False)
        df_detailed.to_csv(f'{self.supporting_data_dir}/precision_at_k_detailed.csv', index=False)

        return combined_summary, df_detailed

    def _create_precision_summary(self, df, k_values, category_name):
        summary = {'category': category_name, 'n_samples': len(df)}

        for k in k_values:
            summary[f'exact_P@{k}'] = (df[f'exact_P@{k}'].mean() * 100).round(1)
            summary[f'semantic_P@{k}'] = (df[f'semantic_P@{k}'].mean() * 100).round(1)

        return pd.DataFrame([summary])

    def analyze_ranking_quality(self, test_data, predictor):
        print(f"\nAnalyzing ranking quality...")

        ranking_results = []

        for i, case in enumerate(tqdm(test_data, desc="Ranking")):
            word = case['word']
            sentence = case['sentence']
            true_video = case['true_video_id']
            is_no_match = (true_video == 'NO_MATCH')

            if is_no_match:
                continue

            exact_result = predictor.predict(word, sentence, method='exact_match')
            semantic_result = predictor.predict(word, sentence, method='semantic')

            exact_position = None
            exact_confidence = None
            if exact_result['success']:
                for pos, pred in enumerate(exact_result['predictions'], 1):
                    if str(pred['video_id']) == str(true_video):
                        exact_position = pos
                        exact_confidence = pred['confidence']
                        break

            semantic_position = None
            semantic_confidence = None
            if semantic_result['success']:
                for pos, pred in enumerate(semantic_result['predictions'], 1):
                    if str(pred['video_id']) == str(true_video):
                        semantic_position = pos
                        semantic_confidence = pred['confidence']
                        break

            ranking_results.append({
                'test_id': i,
                'word': word,
                'true_video': true_video,
                'exact_position': exact_position,
                'exact_confidence': exact_confidence,
                'semantic_position': semantic_position,
                'semantic_confidence': semantic_confidence,
                'exact_found': exact_position is not None,
                'semantic_found': semantic_position is not None
            })

        df_ranking = pd.DataFrame(ranking_results)
        df_ranking.to_csv(f'{self.supporting_data_dir}/ranking_quality_detailed.csv', index=False)

        ranking_summary = {
            'total_match_cases': len(df_ranking),
            'exact_recall': float(df_ranking['exact_found'].sum() / len(df_ranking) * 100) if len(df_ranking) > 0 else 0,
            'semantic_recall': float(df_ranking['semantic_found'].sum() / len(df_ranking) * 100) if len(df_ranking) > 0 else 0
        }

        with open(f'{self.core_analysis_dir}/ranking_quality_summary.json', 'w') as f:
            json.dump(ranking_summary, f, indent=2)

        print(f" Ranking analysis complete")

        return df_ranking, ranking_summary


# method comparator

class MethodComparator:
    def __init__(self, results_dir='thesis_results'):
        self.results_dir = results_dir
        self.core_analysis_dir = os.path.join(results_dir, 'core_analysis')
        self.supporting_data_dir = os.path.join(results_dir, 'supporting_data')
        self.metadata_dir = os.path.join(results_dir, 'metadata')

        os.makedirs(self.core_analysis_dir, exist_ok=True)
        os.makedirs(self.supporting_data_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def conduct_deep_comparison(self, test_data, predictor, pathway_df):
        print(f"Conducting method comparison...")

        results = []
        error_cases = []

        for i, case in enumerate(tqdm(test_data, desc="Evaluating")):
            word = case['word']
            true_video = case['true_video_id']
            sentence = case['sentence']
            is_no_match = (true_video == 'NO_MATCH')

            pathway_info = pathway_df[pathway_df['word'] == word]
            if len(pathway_info) > 0:
                pathway_type = pathway_info['pathway_type'].iloc[0]
            else:
                pathway_type = 'unknown'

            exact_result = predictor.predict(word, sentence, method='exact_match')
            semantic_result = predictor.predict(word, sentence, method='semantic')

            if is_no_match:
                exact_correct = not exact_result['success']
                semantic_correct = not semantic_result['success']
                exact_confidence = 0.0
                semantic_confidence = 0.0
            else:
                exact_correct = False
                exact_confidence = 0.0
                if exact_result['success']:
                    best_exact = exact_result['predictions'][0]
                    exact_correct = (str(best_exact['video_id']) == str(true_video))
                    exact_confidence = best_exact['confidence']

                semantic_correct = False
                semantic_confidence = 0.0
                if semantic_result['success']:
                    predictions = semantic_result['predictions']
                    max_confidence = predictions[0]['confidence']
                    for pred in predictions:
                        if pred['confidence'] == max_confidence:
                            if str(pred['video_id']) == str(true_video):
                                semantic_correct = True
                                semantic_confidence = pred['confidence']
                                break
                        else:
                            break

                    if not semantic_correct and len(predictions) > 0:
                        semantic_confidence = predictions[0]['confidence']

            results.append({
                'test_id': i,
                'word': word,
                'true_video': true_video,
                'is_no_match': is_no_match,
                'pathway_type': pathway_type,
                'exact_match_correct': exact_correct,
                'exact_match_confidence': exact_confidence,
                'semantic_correct': semantic_correct,
                'semantic_confidence': semantic_confidence,
                'sentence_length': len(sentence),
                'improvement': semantic_correct and not exact_correct,
                'deterioration': exact_correct and not semantic_correct
            })

            if (exact_correct != semantic_correct) or (pathway_type not in ['Pathway 3']):
                error_cases.append({
                    'test_id': i,
                    'word': word,
                    'true_video': true_video,
                    'is_no_match': is_no_match,
                    'pathway_type': pathway_type,
                    'sentence': sentence[:150] + '...' if len(sentence) > 150 else sentence,
                    'exact_correct': exact_correct,
                    'semantic_correct': semantic_correct,
                    'exact_confidence': exact_confidence,
                    'semantic_confidence': semantic_confidence
                })

        df_results = pd.DataFrame(results)
        df_results.to_csv(f'{self.supporting_data_dir}/method_comparison_detailed.csv', index=False)

        df_errors = pd.DataFrame(error_cases)
        df_errors.to_csv(f'{self.core_analysis_dir}/error_analysis_qualitative.csv', index=False)

        self._generate_thesis_tables(df_results)

        print(f" Method comparison complete: {len(df_results)} test cases")

        return df_results, df_errors

    def _generate_thesis_tables(self, df_results):
        match_cases = df_results[~df_results['is_no_match']]
        no_match_cases = df_results[df_results['is_no_match']]

        overall_stats = {
            'total_test_cases': int(len(df_results)),
            'match_cases': int(len(match_cases)),
            'no_match_cases': int(len(no_match_cases)),
            'exact_match_accuracy': float(df_results['exact_match_correct'].mean() * 100),
            'semantic_accuracy': float(df_results['semantic_correct'].mean() * 100),
            'improvement_cases': int(df_results['improvement'].sum()),
            'deterioration_cases': int(df_results['deterioration'].sum())
        }

        pd.DataFrame([overall_stats]).to_csv(
            f'{self.core_analysis_dir}/overall_performance_comparison.csv', index=False
        )

        pathway_performance = df_results.groupby('pathway_type').agg({
            'exact_match_correct': 'mean',
            'semantic_correct': 'mean',
            'test_id': 'count'
        }).round(3).reset_index()

        pathway_performance['exact_accuracy'] = (pathway_performance['exact_match_correct'] * 100).round(1)
        pathway_performance['semantic_accuracy'] = (pathway_performance['semantic_correct'] * 100).round(1)
        pathway_performance['improvement'] = pathway_performance['semantic_accuracy'] - pathway_performance['exact_accuracy']

        pathway_performance.to_csv(
            f'{self.core_analysis_dir}/pathway_accuracy_comparison.csv', index=False
        )

        print(f" Performance tables generated")


# model comparison with individual optimization

class ModelComparison:
    def __init__(self, word_video_path, meaning_video_path, results_dir='thesis_results'):
        self.word_video_path = word_video_path
        self.meaning_video_path = meaning_video_path
        self.results_dir = results_dir
        self.model_comparison_dir = os.path.join(results_dir, 'model_comparison')

        os.makedirs(self.model_comparison_dir, exist_ok=True)

    def evaluate_single_model(self, model_name, test_data, hyperparams, pathway_df=None, exclude_indices=None):
        """Evaluate a single model on test data"""
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")

        try:
            print(f"  Loading model and generating embeddings...")
            predictor = CrossModalPredictor(
                self.word_video_path,
                self.meaning_video_path,
                embedding_model=model_name,
                hyperparams=hyperparams,
                exclude_indices=exclude_indices
            )

            results = []

            print(f"  Evaluating on {len(test_data)} test cases...")
            for case in tqdm(test_data, desc=f"  {model_name}", leave=False):
                word = case['word']
                sentence = case['sentence']
                true_video = case['true_video_id']
                is_no_match = (true_video == 'NO_MATCH')

                pathway_type = 'unknown'
                if pathway_df is not None:
                    pathway_info = pathway_df[pathway_df['word'] == word]
                    if len(pathway_info) > 0:
                        pathway_type = pathway_info['pathway_type'].iloc[0]

                exact_result = predictor.predict(word, sentence, method='exact_match')
                semantic_result = predictor.predict(word, sentence, method='semantic')

                if is_no_match:
                    exact_correct = not exact_result['success']
                    semantic_correct = not semantic_result['success']
                    exact_confidence = 0.0
                    semantic_confidence = 0.0
                else:
                    exact_correct = False
                    exact_confidence = 0.0
                    if exact_result['success']:
                        best_exact = exact_result['predictions'][0]
                        exact_correct = (str(best_exact['video_id']) == str(true_video))
                        exact_confidence = best_exact['confidence']

                    semantic_correct = False
                    semantic_confidence = 0.0
                    if semantic_result['success']:
                        predictions = semantic_result['predictions']
                        max_confidence = predictions[0]['confidence']
                        for pred in predictions:
                            if pred['confidence'] == max_confidence:
                                if str(pred['video_id']) == str(true_video):
                                    semantic_correct = True
                                    semantic_confidence = pred['confidence']
                                    break
                            else:
                                break
                        if not semantic_correct and len(predictions) > 0:
                            semantic_confidence = predictions[0]['confidence']

                results.append({
                    'word': word,
                    'pathway_type': pathway_type,
                    'exact_correct': exact_correct,
                    'semantic_correct': semantic_correct,
                    'exact_confidence': exact_confidence,
                    'semantic_confidence': semantic_confidence
                })

            df_results = pd.DataFrame(results)

            metrics = {
                'model_name': model_name,
                'exact_accuracy': float(df_results['exact_correct'].mean() * 100),
                'semantic_accuracy': float(df_results['semantic_correct'].mean() * 100),
                'improvement': float((df_results['semantic_correct'].mean() - df_results['exact_correct'].mean()) * 100),
                'avg_exact_confidence': float(df_results['exact_confidence'].mean()),
                'avg_semantic_confidence': float(df_results['semantic_confidence'].mean()),
                'total_cases': len(df_results)
            }

            if 'pathway_type' in df_results.columns and df_results['pathway_type'].notna().any():
                pathway_performance = df_results.groupby('pathway_type').agg({
                    'exact_correct': 'mean',
                    'semantic_correct': 'mean'
                }).round(4) * 100
                metrics['pathway_performance'] = pathway_performance.to_dict()

            print(f"   {model_name}: {metrics['semantic_accuracy']:.2f}% semantic accuracy")

            del predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return metrics, df_results

        except Exception as e:
            print(f"\n    Error evaluating {model_name}:")
            print(f"     {type(e).__name__}: {str(e)}")
            import traceback
            print(f"     Traceback:")
            traceback.print_exc()
            print(f"     Skipping this model and continuing with others...\n")
            return None, None

    def compare_models(self, test_data, model_configs, pathway_df=None, exclude_indices=None):
        """Compare multiple models, each with their own optimized hyperparameters"""
        print(f"\n{'='*80}")
        print(f"Comparing {len(model_configs)} Embedding Models on Validation Split")
        print(f"{'='*80}")

        if exclude_indices is not None:
            print(f"   Data leakage prevention: Excluding {len(exclude_indices)} examples from searchable data")

        all_metrics = []
        detailed_results = {}

        for config in model_configs:
            model_name = config['model_name']
            hyperparams = config['hyperparams']

            print(f"\nUsing optimized hyperparameters for {model_name}:")
            print(f"  threshold={hyperparams['similarity_threshold']}, top_k={hyperparams['top_k_semantic']}")

            metrics, df_results = self.evaluate_single_model(
                model_name, test_data, hyperparams, pathway_df, exclude_indices=exclude_indices
            )

            if metrics:
                all_metrics.append(metrics)
                detailed_results[model_name] = df_results

        df_comparison = pd.DataFrame(all_metrics)
        df_comparison = df_comparison.sort_values('semantic_accuracy', ascending=False)

        summary_cols = ['model_name', 'exact_accuracy', 'semantic_accuracy', 'improvement',
                       'avg_exact_confidence', 'avg_semantic_confidence', 'total_cases']
        df_comparison[summary_cols].to_csv(f'{self.model_comparison_dir}/model_comparison_summary.csv', index=False)

        for model_name, df_results in detailed_results.items():
            safe_name = model_name.replace('/', '_').replace('-', '_')
            df_results.to_csv(f'{self.model_comparison_dir}/detailed_{safe_name}.csv', index=False)

        self._create_pathway_comparison(detailed_results, df_comparison)
        self._create_comparison_report(df_comparison, detailed_results)

        print("\n" + "♡☆"*40)
        print("MODEL COMPARISON RESULTS")
        print("♡☆"*40)
        print(df_comparison[summary_cols[:4]].to_string(index=False))
        print(f"\n Best model: {df_comparison.iloc[0]['model_name']} ({df_comparison.iloc[0]['semantic_accuracy']:.2f}%)")
        print(f" Results saved to: {self.model_comparison_dir}/")

        return df_comparison

    def _create_pathway_comparison(self, detailed_results, df_comparison):
        """Create pathway-level performance comparison across models"""
        print("\nGenerating pathway-level comparison...")

        pathway_comparisons = []

        for model_name, df_results in detailed_results.items():
            if 'pathway_type' in df_results.columns and df_results['pathway_type'].notna().any():
                pathway_perf = df_results.groupby('pathway_type').agg({
                    'exact_correct': ['mean', 'count'],
                    'semantic_correct': 'mean'
                }).reset_index()

                pathway_perf.columns = ['pathway_type', 'exact_accuracy', 'count', 'semantic_accuracy']
                pathway_perf['exact_accuracy'] = (pathway_perf['exact_accuracy'] * 100).round(2)
                pathway_perf['semantic_accuracy'] = (pathway_perf['semantic_accuracy'] * 100).round(2)
                pathway_perf['improvement'] = (pathway_perf['semantic_accuracy'] - pathway_perf['exact_accuracy']).round(2)
                pathway_perf['model'] = model_name

                pathway_comparisons.append(pathway_perf)

        if pathway_comparisons:
            df_pathway_comparison = pd.concat(pathway_comparisons, ignore_index=True)
            df_pathway_comparison = df_pathway_comparison[['model', 'pathway_type', 'count',
                                                           'exact_accuracy', 'semantic_accuracy', 'improvement']]
            df_pathway_comparison.to_csv(f'{self.model_comparison_dir}/pathway_level_comparison.csv', index=False)
            print(f" Pathway-level comparison saved")

    def _create_comparison_report(self, df_comparison, detailed_results):
        """Create a comprehensive comparison report"""
        print("\nGenerating comprehensive comparison report...")

        report = {
            'comparison_timestamp': datetime.now().isoformat(),
            'models_evaluated': len(df_comparison),
            'total_test_cases': int(df_comparison.iloc[0]['total_cases']) if len(df_comparison) > 0 else 0,
            'best_model': {
                'name': df_comparison.iloc[0]['model_name'] if len(df_comparison) > 0 else None,
                'exact_accuracy': float(df_comparison.iloc[0]['exact_accuracy']) if len(df_comparison) > 0 else 0,
                'semantic_accuracy': float(df_comparison.iloc[0]['semantic_accuracy']) if len(df_comparison) > 0 else 0,
                'improvement': float(df_comparison.iloc[0]['improvement']) if len(df_comparison) > 0 else 0
            },
            'model_rankings': []
        }

        for idx, row in df_comparison.iterrows():
            report['model_rankings'].append({
                'rank': len(report['model_rankings']) + 1,
                'model': row['model_name'],
                'semantic_accuracy': float(row['semantic_accuracy']),
                'exact_accuracy': float(row['exact_accuracy']),
                'improvement': float(row['improvement'])
            })

        with open(f'{self.model_comparison_dir}/comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print(f" Comparison report saved")


# analysis generator

class AnalysisGenerator:
    def __init__(self, results_dir='thesis_results'):
        self.results_dir = results_dir
        self.error_analyzer = ErrorFlowAnalyzer(results_dir)
        self.precision_analyzer = PrecisionAtKAnalyzer(results_dir)
        self.method_comparator = MethodComparator(results_dir)

    def generate_complete_analysis(self, data, predictor, pathway_df, best_hyperparams,
                                   best_model_name, phase='validation'):
        """Generate complete analysis for the best model"""
        print(f"\n{'♡☆'*40}")
        print(f"{phase.upper()} Phase - Complete Analysis")
        print(f"Model: {best_model_name}")
        print(f"{'♡☆'*40}")

        if phase == 'test':
            phase_dir = os.path.join(self.results_dir, 'generalizability_evaluation')
        else:
            phase_dir = os.path.join(self.results_dir, 'overlap_evaluation')
        os.makedirs(phase_dir, exist_ok=True)

        # Update directories for this phase
        self.error_analyzer.core_analysis_dir = os.path.join(phase_dir, 'core_analysis')
        self.error_analyzer.supporting_data_dir = os.path.join(phase_dir, 'supporting_data')
        self.precision_analyzer.core_analysis_dir = os.path.join(phase_dir, 'core_analysis')
        self.precision_analyzer.supporting_data_dir = os.path.join(phase_dir, 'supporting_data')
        self.method_comparator.core_analysis_dir = os.path.join(phase_dir, 'core_analysis')
        self.method_comparator.supporting_data_dir = os.path.join(phase_dir, 'supporting_data')
        self.method_comparator.metadata_dir = os.path.join(phase_dir, 'metadata')

        os.makedirs(self.error_analyzer.core_analysis_dir, exist_ok=True)
        os.makedirs(self.error_analyzer.supporting_data_dir, exist_ok=True)
        os.makedirs(self.method_comparator.metadata_dir, exist_ok=True)

        # Method comparison
        print(f"\n1. Method comparison analysis...")
        results_df, errors_df = self.method_comparator.conduct_deep_comparison(
            data, predictor, pathway_df
        )

        # Error flow
        print(f"\n2. Error flow analysis...")
        error_flows = self.error_analyzer.analyze_error_flows(results_df)

        # Precision@K
        print(f"\n3. Precision@K analysis...")
        precision_summary, precision_detailed = self.precision_analyzer.calculate_precision_at_k(
            data, predictor, pathway_df
        )

        # Ranking quality
        print(f"\n4. Ranking quality analysis...")
        ranking_df, ranking_summary = self.precision_analyzer.analyze_ranking_quality(
            data, predictor
        )

        # Save phase summary
        if phase == 'test':
            phase_summary = {
                'phase': 'generalizability_test',
                'description': 'Evaluation on unseen vocabulary (no overlap with training)',
                'model_name': best_model_name,
                'hyperparameters': best_hyperparams,
                'performance': {
                    'semantic_accuracy': float(results_df['semantic_correct'].mean() * 100),
                    'note': 'Exact match N/A - no vocabulary overlap with training data'
                },
                'test_cases': len(results_df),
                'timestamp': datetime.now().isoformat()
            }
        else:
            phase_summary = {
                'phase': phase,
                'description': 'Evaluation on vocabulary with overlap (primary analysis)',
                'model_name': best_model_name,
                'hyperparameters': best_hyperparams,
                'performance': {
                    'exact_accuracy': float(results_df['exact_match_correct'].mean() * 100),
                    'semantic_accuracy': float(results_df['semantic_correct'].mean() * 100),
                    'improvement': float((results_df['semantic_correct'].mean() -
                                        results_df['exact_match_correct'].mean()) * 100)
                },
                'test_cases': len(results_df),
                'timestamp': datetime.now().isoformat()
            }
        return results_df

# helper functions

def create_validation_summary(results_dir, validation_results, best_model_name, hyperparams):
    """Create summary when only validation phase is complete"""

    summary = {
        'analysis_type': 'VALIDATION_ONLY',
        'status': 'development_phase',
        'test_set_evaluated': False,
        'best_model': best_model_name,
        'best_hyperparameters': hyperparams,
        'validation_performance': {
            'exact_accuracy': float(validation_results['exact_match_correct'].mean() * 100),
            'semantic_accuracy': float(validation_results['semantic_correct'].mean() * 100),
            'improvement': float((validation_results['semantic_correct'].mean() -
                                 validation_results['exact_match_correct'].mean()) * 100)
        },
        'available_results': {
            'validation_results': 'complete',
            'ablation_test_1': 'complete',
            'ablation_test_2': 'complete',
            'test_results': 'not_evaluated'
        },
        'recommendation': 'Use validation_results/ for development. Best model results in validation_results/core_analysis/',
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(results_dir, 'CURRENT_STATUS.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n Status saved: CURRENT_STATUS.json")


def create_final_summary(results_dir, validation_results, test_results,
                        best_model_name, hyperparams):
    """Create summary comparing overlap vs no-overlap evaluation"""

    summary = {
        'analysis_type': 'COMPLETE_EVALUATION',
        'best_model': best_model_name,
        'best_hyperparameters': hyperparams,

        'with_vocabulary_overlap': {
            'description': 'Validation set - words exist in training vocabulary',
            'exact_accuracy': float(validation_results['exact_match_correct'].mean() * 100),
            'semantic_accuracy': float(validation_results['semantic_correct'].mean() * 100),
            'improvement': float((validation_results['semantic_correct'].mean() -
                                 validation_results['exact_match_correct'].mean()) * 100),
            'test_cases': len(validation_results)
        },

        'without_vocabulary_overlap': {
            'description': 'Test set - completely unseen vocabulary',
            'semantic_accuracy': float(test_results['semantic_correct'].mean() * 100),
            'test_cases': len(test_results),
            'note': 'Exact match N/A - no vocabulary overlap'
        },

        'available_results': {
            'overlap_evaluation': 'with vocabulary overlap (primary)',
            'generalizability_evaluation': 'without vocabulary overlap',
            'ablation_test_1': 'complete',
            'ablation_test_2': 'complete'
        },

        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(results_dir, 'FINAL_SUMMARY.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n Summary saved: FINAL_SUMMARY.json")
    print(f"\nPERFORMANCE COMPARISON:")
    print(f"\n  WITH VOCABULARY OVERLAP (Validation):")
    print(f"    Exact Accuracy:    {summary['with_vocabulary_overlap']['exact_accuracy']:.2f}%")
    print(f"    Semantic Accuracy: {summary['with_vocabulary_overlap']['semantic_accuracy']:.2f}%")
    print(f"    Improvement:       +{summary['with_vocabulary_overlap']['improvement']:.2f}%")
    print(f"\n  WITHOUT VOCABULARY OVERLAP (Generalizability):")
    print(f"    Semantic Accuracy: {summary['without_vocabulary_overlap']['semantic_accuracy']:.2f}%")

# main execution
def main():
    print("♡☆" * 40)
    print("Ambiguity Resolution Across German Words and Their Sign Correspondents")
    print("Master Thesis Analysis Pipeline")
    print("♡☆" * 40)

    config = load_config()

    word_video_path = config['paths']['word_video']
    meaning_video_path = config['paths']['meaning_video']
    meaning_video_gt_path = config['paths']['meaning_video_gt']
    meaning_video_test_path = config['paths']['meaning_video_test']
    test_set_path = config['paths']['test_set']
    results_dir = config['paths']['results']
    pathway_ground_truth_path = config['paths']['pathway_ground_truth']
    embedding_models = config['models']['embedding_models']

    # THREE-WAY DATA SPLITTING
    print("\nTHREE-WAY DATA SPLITTING")
    print("♡☆" * 40)

    df_word_video = load_csv_with_encoding(word_video_path, 'word_video')
    optimizer = HyperparameterOptimizer(word_video_path, meaning_video_path, results_dir)

    data_splits = optimizer.create_three_way_split(
        df_word_video,
        optimization_size=config['optimization'].get('optimization_size', 100),
        validation_size=config['optimization'].get('validation_size', 150),
        random_state=config['optimization']['random_state']
    )

    optimization_data = data_splits['optimization']
    optimization_indices = data_splits['optimization_indices']
    validation_data = data_splits['validation']
    validation_indices = data_splits['validation_indices']
    train_indices = data_splits['train_indices']

    excluded_indices_for_optimization = optimization_indices + validation_indices

    print(f"\n Data split summary:")
    print(f"  Training pool: {len(train_indices)} examples")
    print(f"  Optimization set: {len(optimization_data)} examples")
    print(f"  Validation set: {len(validation_data)} examples")
    print(f"  Total excluded during optimization: {len(excluded_indices_for_optimization)} examples")

    # HYPERPARAMETER OPTIMIZATION FOR ALL MODELS
    print("\n" + "♡☆" * 40)
    print("Hyperparameter Optimization for All Models")
    print("Using dedicated Optimization Split")
    print("♡☆" * 40)

    model_configs = []

    param_grid = {
        'similarity_threshold': config['hyperparameters']['similarity_threshold_range'],
        'top_k_semantic': config['hyperparameters']['top_k_semantic_range'],
        'batch_size': [config['hyperparameters']['batch_size']]
    }

    for model_name in embedding_models:
        print(f"\nOptimizing: {model_name}")

        best_hyperparams, optimization_results = optimizer.optimize_for_model(
            optimization_data,
            model_name,
            exclude_indices=excluded_indices_for_optimization,
            param_grid=param_grid
        )

        model_configs.append({
            'model_name': model_name,
            'hyperparams': best_hyperparams
        })

    # Model Comparison on Validation Split
    print("\n" + "♡☆" * 40)
    print("Model Comparison on Validation Split")
    print("Using the separate Validation split")
    print("♡☆" * 40)
    print(f"\nComparing {len(embedding_models)} models with their optimized hyperparameters")

    pathway_analyzer = PathwayAnalyzer(results_dir)
    ground_truth_pathway_df = pathway_analyzer.load_ground_truth_pathways(pathway_ground_truth_path)

    model_comparator = ModelComparison(word_video_path, meaning_video_path, results_dir)

    model_comparison_results = model_comparator.compare_models(
        validation_data,
        model_configs,
        pathway_df=ground_truth_pathway_df,
        exclude_indices=excluded_indices_for_optimization
    )

    # SELECT BEST MODEL
    best_model_name = model_comparison_results.iloc[0]['model_name']
    best_model_accuracy = model_comparison_results.iloc[0]['semantic_accuracy']

    best_model_config = next(c for c in model_configs if c['model_name'] == best_model_name)
    best_hyperparams = best_model_config['hyperparams']

    print(f"\n" + "♡☆" * 40)
    print("BEST MODEL:")
    print("♡☆" * 40)
    print(f"  Model: {best_model_name}")
    print(f"  Validation Accuracy: {best_model_accuracy:.2f}%")
    print(f"  Hyperparameters: threshold={best_hyperparams['similarity_threshold']}, top_k={best_hyperparams['top_k_semantic']}")
    print(f"\n  This model will be used for all subsequent analyses")

    # Complete Analysis with only the Best Model 
    print("\n" + "♡☆" * 40)
    print("Complete Analysis")
    print("Only for BEST MODEL")
    print("♡☆" * 40)

    predictor_validation = CrossModalPredictor(
        word_video_path,
        meaning_video_path,
        embedding_model=best_model_name,
        hyperparams=best_hyperparams,
        exclude_indices=excluded_indices_for_optimization,
        ablation_mode='baseline'
    )

    analysis_generator = AnalysisGenerator(results_dir)

    validation_results = analysis_generator.generate_complete_analysis(
        data=validation_data,
        predictor=predictor_validation,
        pathway_df=ground_truth_pathway_df,
        best_hyperparams=best_hyperparams,
        best_model_name=best_model_name,
        phase='validation'
    )

    # Pathway discovery - validation phase
    print("\nModel Pathway Discovery - VALIDATION PHASE")
    print("♡☆" * 40)

    best_model_pathways_val = pathway_analyzer.discover_pathways_from_predictions(
        validation_data,
        predictor_validation,
        best_model_name
    )

    # Compare pathways
    print("\nComparing Pathway Classifications - VALIDATION PHASE")
    print("♡☆" * 40)

    pathway_analyzer.compare_pathway_classifications(
        ground_truth_pathway_df,
        best_model_pathways_val,
        best_model_name
    )

    print(f"\n Validation phase complete")
    print(f"  Results saved to: {results_dir}/validation_results/")

    # ABLATION TEST 1 - Column Impact (Word vs Sentence vs Combined)
 
    print("\n" + "♡☆" * 40)
    print("ABLATION TEST 1: Input Column Impact Analysis")
    print("Testing on Validation Split with BEST MODEL")
    print("♡☆" * 40)
    print("\nThis will test three configurations:")
    print("  1. BASELINE: word + sentence (our successful setup)")
    print("  2. NO_WORD: sentence only")
    print("  3. NO_SENTENCE: word only")
    print("\nThis analysis will show the impact of each column.")

    ablation_analyzer = AblationTestAnalyzer(
        word_video_path,
        meaning_video_path,
        results_dir
    )

    ablation_results_1 = ablation_analyzer.run_ablation_test(
        validation_data=validation_data,
        best_model_name=best_model_name,
        best_hyperparams=best_hyperparams,
        pathway_df=ground_truth_pathway_df,
        exclude_indices=excluded_indices_for_optimization
    )

    print(f"\n Ablation test 1 complete")
    print(f"  Results saved to: {results_dir}/ablation_test_1_columns/")

    # ABLATION TEST 2 - Meaning Dataset Comparison (BASE vs GT)

    print("\n" + "♡☆" * 40)
    print("ABLATION TEST 2: Meaning Dataset Comparison")
    print("Testing on Validation Split with BEST MODEL")
    print("♡☆" * 40)
    print("\nThis will compare two meaning datasets:")
    print(f"  1. BASE: {meaning_video_path}")
    print(f"  2. GT:   {meaning_video_gt_path}")
    print("\nThis tests the impact of different meaning descriptions.")

    meaning_ablation_analyzer = MeaningDatasetAblationAnalyzer(
        word_video_path,
        results_dir
    )

    ablation_results_2 = meaning_ablation_analyzer.run_meaning_dataset_ablation(
        validation_data=validation_data,
        best_model_name=best_model_name,
        best_hyperparams=best_hyperparams,
        pathway_df=ground_truth_pathway_df,
        exclude_indices=excluded_indices_for_optimization,
        meaning_video_base_path=meaning_video_path,
        meaning_video_gt_path=meaning_video_gt_path
    )

    print(f"\n Ablation test 2 complete")
    print(f"  Results saved to: {results_dir}/ablation_test_2_meaning_dataset/")

    # No-Overlap Test Phase (OPTIONAL)

    print("\n" + "♡☆" * 40)
    print("No-Overlap Test Evaluation (OPTIONAL)")
    print("♡☆" * 40)
    print("\nWARNING: This will evaluate on the held-out test set.")
    print("   This will load TestSet.csv, pathway_ground_truth_test.csv AND videoid_meaning_dataset_test.csv")
    print("   Only do this when you're ready for FINAL thesis results!")
    print("\nType 'YES' to proceed with test set evaluation:")

    confirmation = input("\nYour answer: ").strip().upper()

    if confirmation != 'YES':
        print("\nStopped before test evaluation (recommended during development)")
        print("\nAnalysis available in:")
        print(f"  - {results_dir}/validation_results/core_analysis/")
        print(f"  - {results_dir}/ablation_test_1_columns/ (word/sentence impact)")
        print(f"  - {results_dir}/ablation_test_2_meaning_dataset/ (base vs gt)")
        print(f"  - {results_dir}/supporting_data/ (pathway comparisons)")
        print(f"  - {results_dir}/model_comparison/ (all models)")

        create_validation_summary(results_dir, validation_results, best_model_name, best_hyperparams)

        try:
            zip_path = create_results_zip(results_dir)
            print(f"\nDownload your results: {zip_path}")
        except Exception as e:
            print(f"\nCould not create zip file: {str(e)}")

        return {
            'analysis_type': 'validation_only',
            'best_model': best_model_name,
            'best_hyperparams': best_hyperparams,
            'ground_truth_pathway_df': ground_truth_pathway_df,
            'validation_results': validation_results,
            'ablation_results_1': ablation_results_1,
            'ablation_results_2': ablation_results_2,
            'test_set_evaluated': False
        }

    # Load test set
    print("\n" + "♡☆" * 40)
    print("Held-out Test Data Loading...")
    print("♡☆" * 40)
    print(f"Loading test word-video pairs from: {test_set_path}")
    print(f"Loading test meanings from: {meaning_video_test_path}")

    test_data = optimizer.load_external_test_set(test_set_path)

    # Load test pathway ground truth
    print("\nLoading test pathway ground truth...")
    pathway_ground_truth_test_path = config['paths']['pathway_ground_truth_test']
    ground_truth_pathway_df_test = pathway_analyzer.load_ground_truth_pathways(pathway_ground_truth_test_path)

    # Create predictor for test phase
    print("\n" + "♡☆" * 40)
    print("Creating Predictor for Generalization Test")
    print("♡☆" * 40)
    print(f"Using best model: {best_model_name}")
    print("Using videoid_meaning_dataset_test.csv for final evaluation...")
    print("NOTE: Test set has no vocabulary overlap with training - semantic-only evaluation")

    predictor_test = CrossModalPredictor(
        word_video_path,
        meaning_video_test_path,
        embedding_model=best_model_name,
        hyperparams=best_hyperparams,
        exclude_indices=None,
        ablation_mode='baseline'
    )

    # Generate test phase analysis
    test_results = analysis_generator.generate_complete_analysis(
        data=test_data,
        predictor=predictor_test,
        pathway_df=ground_truth_pathway_df_test,
        best_hyperparams=best_hyperparams,
        best_model_name=best_model_name,
        phase='test'
    )

    # Pathway discovery - test phase
    print("\nModel Pathway Discovery - No Overlap")
    print("♡☆" * 40)

    best_model_pathways_test = pathway_analyzer.discover_pathways_from_predictions(
        test_data,
        predictor_test,
        f"{best_model_name}_test"
    )

    # Compare pathways - test phase
    print("\nPathway Classification Comparison - No Overlap")
    print("♡☆" * 40)

    pathway_analyzer.compare_pathway_classifications(
        ground_truth_pathway_df_test,
        best_model_pathways_test,
        f"{best_model_name}_test"
    )

    print(f"\nTest phase complete")
    print(f"  Results saved to: {results_dir}/test_results/")

    # Create final summary
    create_final_summary(results_dir,
                        validation_results,
                        test_results,
                        best_model_name,
                        best_hyperparams)

    print("\n" + "♡☆" * 40)
    print("ANALYSIS COMPLETE")
    print("♡☆" * 40)
    print(f"\nResults Structure:")
    print(f"  DATA SPLITS:")
    print(f"    - Training Pool: Used for predictions (searchable)")
    print(f"    - Optimization Set: Used for hyperparameter tuning")
    print(f"    - Validation Set: Used for model comparison")
    print(f"  OPTIMIZATION PHASE:")
    print(f"    - optimization/ (all models' hyperparameter search)")
    print(f"  MODEL COMPARISON:")
    print(f"    - model_comparison/ (all models on validation set)")
    print(f"  VALIDATION PHASE (BEST MODEL ONLY):")
    print(f"    - validation_results/core_analysis/")
    print(f"  ABLATION TEST 1 (BEST MODEL ONLY):")
    print(f"    - ablation_test_1_columns/ (word/sentence impact)")
    print(f"  ABLATION TEST 2 (BEST MODEL ONLY):")
    print(f"    - ablation_test_2_meaning_dataset/ (base vs gt comparison)")
    print(f"  TEST PHASE (BEST MODEL ONLY):")
    print(f"    - test_results/core_analysis/")
    print(f"  PATHWAY ANALYSIS:")
    print(f"    - supporting_data/ground_truth_pathways.csv")
    print(f"    - supporting_data/model_discovered_pathways_*.csv")
    print(f"    - metadata/pathway_comparison_*.csv")
    print(f"  SUMMARY:")
    print(f"    - FINAL_SUMMARY.json")

    try:
        zip_path = create_results_zip(results_dir)
        print(f"\nDownload your results: {zip_path}")
    except Exception as e:
        print(f"\nCould not create zip file: {str(e)}")

    return {
        'analysis_type': 'complete',
        'validation_results': validation_results,
        'test_results': test_results,
        'ablation_results_1': ablation_results_1,
        'ablation_results_2': ablation_results_2,
        'best_model': best_model_name,
        'best_hyperparams': best_hyperparams,
        'ground_truth_pathway_df': ground_truth_pathway_df,
        'model_comparison_results': model_comparison_results
    }


if __name__ == "__main__":
    results = main()
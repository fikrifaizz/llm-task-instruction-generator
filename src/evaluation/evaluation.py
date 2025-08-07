"""
Comprehensive Evaluation & Benchmarking Framework
For Task-Oriented Instruction Generation Models
Evaluates both quantitative metrics and qualitative aspects
"""

import torch
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import re
import time
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# NLP evaluation libraries
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize, sent_tokenize
    import nltk

    nltk.download('punkt', quiet=True)
    NLP_METRICS_AVAILABLE = True
except ImportError:
    print("Installing NLP evaluation libraries...")
    import os

    os.system("pip install rouge-score nltk")
    NLP_METRICS_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics"""

    # Quantitative metrics
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    bleu_score: float = 0.0

    # Task-specific metrics
    step_count_accuracy: float = 0.0
    structural_completeness: float = 0.0
    instruction_clarity: float = 0.0
    app_specificity: float = 0.0

    # Technical metrics
    inference_time: float = 0.0
    token_efficiency: float = 0.0
    repetition_rate: float = 0.0

    # Overall quality
    overall_score: float = 0.0


@dataclass
class BenchmarkResult:
    """Container for benchmark comparison results"""

    model_name: str
    metrics: EvaluationMetrics
    sample_outputs: List[str]
    timestamp: str


class TaskInstructionEvaluator:
    """
    Comprehensive evaluator for task-oriented instruction generation
    """

    def __init__(self, model_path: str, tokenizer_path: str = None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path

        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Evaluation configuration
        self.config = {
            "max_length": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3
        }

        # Ground truth patterns for validation
        self.validation_patterns = {
            "step_numbering": r'^\d+\.\s+',
            "action_verbs": ['open', 'tap', 'click', 'select', 'enter', 'go', 'navigate',
                             'buka', 'klik', 'pilih', 'masuk', 'isi'],
            "ui_elements": ['button', 'menu', 'tab', 'page', 'screen', 'app',
                            'tombol', 'menu', 'halaman', 'aplikasi'],
            "completion_indicators": ['confirm', 'complete', 'save', 'submit', 'done',
                                      'konfirmasi', 'selesai', 'simpan']
        }

        self.load_model()

    def load_model(self):
        """Load the fine-tuned model"""
        logger.info(f"Loading model from {self.model_path}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map=None
            ).to(self.device)

            logger.info("✅ Model loaded successfully for evaluation")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def generate_response(self, prompt: str) -> Tuple[str, float]:
        """Generate response and measure inference time"""

        # Format prompt
        if not prompt.strip().endswith('?'):
            prompt = prompt.strip() + '?'

        formatted_prompt = f"<s>User: {prompt}<sep>Assistant:"

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        # Measure inference time
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=120,
                temperature=self.config['temperature'],
                top_p=self.config['top_p'],
                do_sample=True,
                repetition_penalty=self.config['repetition_penalty'],
                no_repeat_ngram_size=self.config['no_repeat_ngram_size'],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True,
            )

        inference_time = time.time() - start_time

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        # Clean response
        response = response.replace("<sep>", "").strip()

        return response, inference_time

    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""

        if not NLP_METRICS_AVAILABLE:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, generated)

            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def calculate_bleu_score(self, generated: str, reference: str) -> float:
        """Calculate BLEU score"""

        if not NLP_METRICS_AVAILABLE:
            return 0.0

        try:
            # Tokenize
            gen_tokens = word_tokenize(generated.lower())
            ref_tokens = word_tokenize(reference.lower())

            # Calculate BLEU with smoothing
            smoothing = SmoothingFunction().method1
            bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing)

            return bleu
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            return 0.0

    def evaluate_step_structure(self, generated: str) -> Dict[str, float]:
        """Evaluate step structure and completeness"""

        lines = generated.strip().split('\n')

        # Count numbered steps
        numbered_steps = 0
        action_verbs = 0
        ui_elements = 0
        has_completion = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check numbering
            if re.match(self.validation_patterns['step_numbering'], line):
                numbered_steps += 1

            # Check action verbs
            line_lower = line.lower()
            if any(verb in line_lower for verb in self.validation_patterns['action_verbs']):
                action_verbs += 1

            # Check UI elements
            if any(element in line_lower for element in self.validation_patterns['ui_elements']):
                ui_elements += 1

        # Check completion indicators in last line
        if lines:
            last_line = lines[-1].lower()
            has_completion = any(indicator in last_line
                                 for indicator in self.validation_patterns['completion_indicators'])

        total_lines = len([l for l in lines if l.strip()])

        return {
            "step_count_accuracy": min(numbered_steps / max(total_lines, 1), 1.0),
            "action_verb_coverage": min(action_verbs / max(total_lines, 1), 1.0),
            "ui_element_coverage": min(ui_elements / max(total_lines, 1), 1.0),
            "has_completion": 1.0 if has_completion else 0.0,
            "structural_completeness": (numbered_steps > 3 and has_completion)
        }

    def evaluate_app_specificity(self, generated: str, app_name: str) -> float:
        """Evaluate app-specific terminology usage"""

        app_specific_terms = {
            "shopee": ["shopee", "me tab", "shopeepay", "shopee live"],
            "tokopedia": ["tokopedia", "daftar transaksi", "topup", "ovo"],
            "lazada": ["lazada", "my orders", "laz wallet", "lazada wallet"],
            "blibli": ["blibli", "akun saya", "pesanan saya", "bli points"]
        }

        app_lower = app_name.lower()
        if app_lower not in app_specific_terms:
            return 0.5  # Neutral score for unknown apps

        generated_lower = generated.lower()
        relevant_terms = app_specific_terms[app_lower]

        found_terms = sum(1 for term in relevant_terms if term in generated_lower)

        return min(found_terms / len(relevant_terms), 1.0)

    def calculate_repetition_rate(self, text: str) -> float:
        """Calculate repetition rate in the generated text"""

        words = text.lower().split()
        if len(words) < 4:
            return 0.0

        # Check for repeated n-grams
        repetitions = 0
        total_ngrams = 0

        for n in [2, 3, 4]:  # Check 2-4 gram repetitions
            ngrams = []
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i + n])
                ngrams.append(ngram)

            ngram_counts = Counter(ngrams)
            repeated = sum(count - 1 for count in ngram_counts.values() if count > 1)

            repetitions += repeated
            total_ngrams += len(ngrams)

        return repetitions / max(total_ngrams, 1)

    def evaluate_single_sample(self, prompt: str, reference: str, app_name: str = None) -> EvaluationMetrics:
        """Evaluate a single prompt-response pair"""

        # Generate response
        generated, inference_time = self.generate_response(prompt)

        # Calculate quantitative metrics
        rouge_scores = self.calculate_rouge_scores(generated, reference)
        bleu_score = self.calculate_bleu_score(generated, reference)

        # Calculate task-specific metrics
        structure_metrics = self.evaluate_step_structure(generated)

        # App specificity
        app_specificity = 0.5  # Default
        if app_name:
            app_specificity = self.evaluate_app_specificity(generated, app_name)

        # Technical metrics
        repetition_rate = self.calculate_repetition_rate(generated)
        token_count = len(self.tokenizer.encode(generated))
        token_efficiency = min(token_count / 150, 1.0)  # Optimal around 150 tokens

        # Calculate overall score (weighted combination)
        weights = {
            'rouge_l': 0.2,
            'bleu': 0.15,
            'structure': 0.25,
            'app_specificity': 0.15,
            'clarity': 0.15,
            'efficiency': 0.1
        }

        overall_score = (
                weights['rouge_l'] * rouge_scores['rougeL'] +
                weights['bleu'] * bleu_score +
                weights['structure'] * structure_metrics['structural_completeness'] +
                weights['app_specificity'] * app_specificity +
                weights['clarity'] * (1 - repetition_rate) +
                weights['efficiency'] * (1 - token_efficiency)
        )

        return EvaluationMetrics(
            rouge_1=rouge_scores['rouge1'],
            rouge_2=rouge_scores['rouge2'],
            rouge_l=rouge_scores['rougeL'],
            bleu_score=bleu_score,
            step_count_accuracy=structure_metrics['step_count_accuracy'],
            structural_completeness=structure_metrics['structural_completeness'],
            instruction_clarity=1 - repetition_rate,
            app_specificity=app_specificity,
            inference_time=inference_time,
            token_efficiency=1 - token_efficiency,
            repetition_rate=repetition_rate,
            overall_score=overall_score
        )

    def evaluate_test_set(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model on a complete test set"""

        logger.info(f"Evaluating model on {len(test_data)} test samples...")

        all_metrics = []
        detailed_results = []

        for i, sample in enumerate(test_data):
            prompt = sample['user_intent']
            reference = '\n'.join(sample['structured_instructions'])
            app_name = sample.get('app', None)

            # Evaluate sample
            metrics = self.evaluate_single_sample(prompt, reference, app_name)

            # Generate for detailed analysis
            generated, _ = self.generate_response(prompt)

            all_metrics.append(metrics)
            detailed_results.append({
                'prompt': prompt,
                'reference': reference,
                'generated': generated,
                'app': app_name,
                'metrics': metrics
            })

            if (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i + 1}/{len(test_data)} samples")

        # Aggregate metrics
        avg_metrics = self._aggregate_metrics(all_metrics)

        # Generate evaluation report
        report = self._generate_evaluation_report(avg_metrics, detailed_results)

        return {
            'aggregate_metrics': avg_metrics,
            'detailed_results': detailed_results,
            'evaluation_report': report
        }

    def _aggregate_metrics(self, metrics_list: List[EvaluationMetrics]) -> EvaluationMetrics:
        """Aggregate metrics across all samples"""

        if not metrics_list:
            return EvaluationMetrics()

        # Calculate averages
        avg_metrics = EvaluationMetrics()

        for attr in ['rouge_1', 'rouge_2', 'rouge_l', 'bleu_score',
                     'step_count_accuracy', 'structural_completeness',
                     'instruction_clarity', 'app_specificity', 'inference_time',
                     'token_efficiency', 'repetition_rate', 'overall_score']:
            values = [getattr(m, attr) for m in metrics_list]
            setattr(avg_metrics, attr, np.mean(values))

        return avg_metrics

    def _generate_evaluation_report(self, metrics: EvaluationMetrics,
                                    detailed_results: List[Dict]) -> str:
        """Generate comprehensive evaluation report"""

        report = f"""
=== TASK-ORIENTED INSTRUCTION GENERATION EVALUATION REPORT ===

📊 QUANTITATIVE METRICS:
- ROUGE-1 Score: {metrics.rouge_1:.3f}
- ROUGE-2 Score: {metrics.rouge_2:.3f}
- ROUGE-L Score: {metrics.rouge_l:.3f}
- BLEU Score: {metrics.bleu_score:.3f}

🎯 TASK-SPECIFIC METRICS:
- Step Count Accuracy: {metrics.step_count_accuracy:.3f}
- Structural Completeness: {metrics.structural_completeness:.3f}
- Instruction Clarity: {metrics.instruction_clarity:.3f}
- App Specificity: {metrics.app_specificity:.3f}

⚡ TECHNICAL METRICS:
- Average Inference Time: {metrics.inference_time:.3f}s
- Token Efficiency: {metrics.token_efficiency:.3f}
- Repetition Rate: {metrics.repetition_rate:.3f}

🏆 OVERALL SCORE: {metrics.overall_score:.3f}

📈 PERFORMANCE ANALYSIS:
"""

        # Performance categorization
        if metrics.overall_score >= 0.8:
            report += "✅ EXCELLENT: Model performs exceptionally well\n"
        elif metrics.overall_score >= 0.7:
            report += "✅ GOOD: Model performs well with minor improvements needed\n"
        elif metrics.overall_score >= 0.6:
            report += "⚠️  FAIR: Model shows promise but needs significant improvements\n"
        else:
            report += "❌ POOR: Model requires major improvements\n"

        # Detailed analysis
        report += f"""
🔍 DETAILED ANALYSIS:
- Structural Quality: {'High' if metrics.structural_completeness > 0.7 else 'Needs Improvement'}
- Language Clarity: {'High' if metrics.instruction_clarity > 0.8 else 'Needs Improvement'}
- App Specificity: {'High' if metrics.app_specificity > 0.6 else 'Needs Improvement'}
- Speed Performance: {'Fast' if metrics.inference_time < 1.0 else 'Slow'}

💡 RECOMMENDATIONS:
"""

        # Generate recommendations
        recommendations = []

        if metrics.structural_completeness < 0.7:
            recommendations.append("- Improve step numbering and logical flow")

        if metrics.app_specificity < 0.6:
            recommendations.append("- Add more app-specific terminology in training data")

        if metrics.repetition_rate > 0.2:
            recommendations.append("- Increase repetition penalty and improve diversity")

        if metrics.inference_time > 2.0:
            recommendations.append("- Optimize model size or generation parameters")

        if not recommendations:
            recommendations.append("- Model performs well, consider expanding to new domains")

        report += '\n'.join(recommendations)

        return report


class BenchmarkComparator:
    """Compare model performance against baselines"""

    def __init__(self):
        self.baseline_models = {
            "random": self._random_baseline,
            "template": self._template_baseline,
            "gpt2_base": self._gpt2_baseline
        }

    def _random_baseline(self, prompt: str) -> str:
        """Random baseline for comparison"""
        templates = [
            "1. Open the app\n2. Go to settings\n3. Find the option\n4. Make changes\n5. Save",
            "1. Launch application\n2. Navigate to menu\n3. Select required option\n4. Complete action",
            "1. Start the process\n2. Follow instructions\n3. Confirm changes"
        ]
        return np.random.choice(templates)

    def _template_baseline(self, prompt: str) -> str:
        """Template-based baseline"""
        if "password" in prompt.lower():
            return "1. Open app\n2. Go to settings\n3. Select forgot password\n4. Enter email\n5. Check email"
        elif "track" in prompt.lower() or "order" in prompt.lower():
            return "1. Open app\n2. Go to orders\n3. Find your order\n4. View tracking"
        else:
            return "1. Open the application\n2. Navigate to relevant section\n3. Complete required action"

    def _gpt2_baseline(self, prompt: str) -> str:
        """GPT-2 baseline (simplified)"""
        # This would load actual GPT-2 base model
        return "1. Follow the standard procedure\n2. Complete the required steps\n3. Verify the results"

    def run_benchmark(self, evaluator: TaskInstructionEvaluator,
                      test_data: List[Dict], baseline_name: str = "template") -> Dict:
        """Run benchmark comparison"""

        logger.info(f"Running benchmark against {baseline_name} baseline...")

        # Evaluate fine-tuned model
        ft_results = evaluator.evaluate_test_set(test_data)

        # Evaluate baseline
        baseline_metrics = []
        for sample in test_data:
            prompt = sample['user_intent']
            reference = '\n'.join(sample['structured_instructions'])

            # Generate baseline response
            baseline_response = self.baseline_models[baseline_name](prompt)

            # Calculate metrics for baseline
            rouge_scores = evaluator.calculate_rouge_scores(baseline_response, reference)
            bleu_score = evaluator.calculate_bleu_score(baseline_response, reference)

            baseline_metrics.append({
                'rouge_l': rouge_scores['rougeL'],
                'bleu': bleu_score
            })

        # Aggregate baseline metrics
        avg_baseline = {
            'rouge_l': np.mean([m['rouge_l'] for m in baseline_metrics]),
            'bleu': np.mean([m['bleu'] for m in baseline_metrics])
        }

        # Compare results
        ft_metrics = ft_results['aggregate_metrics']

        comparison = {
            'fine_tuned_model': {
                'rouge_l': ft_metrics.rouge_l,
                'bleu_score': ft_metrics.bleu_score,
                'overall_score': ft_metrics.overall_score
            },
            'baseline': avg_baseline,
            'improvement': {
                'rouge_l_improvement': ft_metrics.rouge_l - avg_baseline['rouge_l'],
                'bleu_improvement': ft_metrics.bleu_score - avg_baseline['bleu']
            }
        }

        return comparison


def create_test_dataset() -> List[Dict[str, Any]]:
    """Create comprehensive test dataset"""

    test_samples = [
        {
            "user_intent": "How do I reset my password in Shopee?",
            "structured_instructions": [
                "1. Open the Shopee mobile app",
                "2. Tap on 'Me' tab at the bottom",
                "3. Select 'Settings' from the menu",
                "4. Tap on 'Account & Security'",
                "5. Choose 'Password' option",
                "6. Tap 'Forgot Password'",
                "7. Enter your registered phone number or email",
                "8. Verify with OTP sent to your device",
                "9. Create and confirm your new password"
            ],
            "app": "Shopee",
            "category": "reset_password"
        },
        {
            "user_intent": "How to track my order in Tokopedia?",
            "structured_instructions": [
                "1. Open Tokopedia application",
                "2. Go to 'Daftar Transaksi' (Transaction List)",
                "3. Find the order you want to track",
                "4. Tap on the specific order",
                "5. View current order status",
                "6. Check shipping progress",
                "7. See estimated delivery time",
                "8. Copy tracking number if needed"
            ],
            "app": "Tokopedia",
            "category": "track_order"
        },
        {
            "user_intent": "Steps to add payment method in Lazada?",
            "structured_instructions": [
                "1. Open Lazada application",
                "2. Navigate to 'Account' section",
                "3. Select 'Payment Options'",
                "4. Tap 'Add Payment Method'",
                "5. Choose payment type (Credit/Debit/Wallet)",
                "6. Enter payment credentials",
                "7. Verify through bank authentication",
                "8. Confirm payment method addition",
                "9. Set as default if desired"
            ],
            "app": "Lazada",
            "category": "add_payment"
        }
    ]

    return test_samples


def main():
    """Main evaluation execution"""

    print("🎯 Comprehensive Evaluation & Benchmarking")

    # Configuration
    model_path = "./models/enhanced_m4"  # Adjust path

    try:
        # Initialize evaluator
        evaluator = TaskInstructionEvaluator(model_path)

        # Create test dataset
        test_data = create_test_dataset()

        # Run comprehensive evaluation
        print("\n📊 Running comprehensive evaluation...")
        results = evaluator.evaluate_test_set(test_data)

        # Print evaluation report
        print(results['evaluation_report'])

        # Run benchmark comparison
        print("\n🏆 Running benchmark comparison...")
        comparator = BenchmarkComparator()
        benchmark_results = comparator.run_benchmark(evaluator, test_data, "template")

        print(f"\n=== BENCHMARK COMPARISON ===")
        print(f"Fine-tuned Model ROUGE-L: {benchmark_results['fine_tuned_model']['rouge_l']:.3f}")
        print(f"Baseline ROUGE-L: {benchmark_results['baseline']['rouge_l']:.3f}")
        print(f"Improvement: {benchmark_results['improvement']['rouge_l_improvement']:.3f}")

        print(f"\nFine-tuned Model BLEU: {benchmark_results['fine_tuned_model']['bleu_score']:.3f}")
        print(f"Baseline BLEU: {benchmark_results['baseline']['bleu']:.3f}")
        print(f"Improvement: {benchmark_results['improvement']['bleu_improvement']:.3f}")

        # Save results
        output_dir = Path("../evaluation")
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "evaluation_results.json", 'w') as f:
            # Convert metrics to dict for JSON serialization
            serializable_results = {
                'aggregate_metrics': results['aggregate_metrics'].__dict__,
                'benchmark_comparison': benchmark_results
            }
            json.dump(serializable_results, f, indent=2)

        print(f"\n✅ Evaluation completed! Results saved to {output_dir}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
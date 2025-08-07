"""
Quality-Improved Training Pipeline
Fixes issues identified in initial training results
"""

import torch
import json
import os
import logging
from typing import Dict, List
import gc

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityImprovedTrainer:
    """Enhanced trainer addressing quality issues"""

    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

        logger.info(f"Using device: {self.device}")

        # Enhanced configuration
        self.config = {
            "model_name": "distilgpt2",
            "max_length": 256,  # Increased for complete responses
            "batch_size": 2,    # Reduced for better quality
            "epochs": 5,        # More epochs for better learning
            "learning_rate": 2e-5,  # Lower LR for stable learning
            "temperature": 0.7,
            "top_p": 0.9,
        }

    def load_model(self):
        """Load model with enhanced configuration"""
        logger.info(f"Loading {self.config['model_name']}...")

        # Enhanced tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])

        # Add special tokens for better structure
        special_tokens = {
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "sep_token": "<sep>"
        }

        self.tokenizer.add_special_tokens(special_tokens)

        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float32,
        )

        # Resize embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model = self.model.to(self.device)

        logger.info("✅ Enhanced model loaded!")

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters: {total_params:,}")

    def create_enhanced_dataset(self):
        """Create high-quality structured dataset"""

        # Enhanced examples with better structure
        enhanced_examples = [
            {
                "user_intent": "How do I reset my password in Shopee?",
                "app": "Shopee",
                "steps": [
                    "Open the Shopee mobile app",
                    "Tap on 'Me' tab at the bottom",
                    "Select 'Settings' from the menu",
                    "Tap on 'Account & Security'",
                    "Choose 'Password' option",
                    "Tap 'Forgot Password'",
                    "Enter your registered phone number or email",
                    "Verify with OTP sent to your device",
                    "Create and confirm your new password"
                ]
            },
            {
                "user_intent": "How to track my order in Tokopedia?",
                "app": "Tokopedia",
                "steps": [
                    "Open Tokopedia application",
                    "Go to 'Daftar Transaksi' (Transaction List)",
                    "Find the order you want to track",
                    "Tap on the specific order",
                    "View current order status",
                    "Check shipping progress",
                    "See estimated delivery time",
                    "Copy tracking number if needed"
                ]
            },
            {
                "user_intent": "Steps to add payment method in Lazada?",
                "app": "Lazada",
                "steps": [
                    "Open Lazada application",
                    "Navigate to 'Account' section",
                    "Select 'Payment Options'",
                    "Tap 'Add Payment Method'",
                    "Choose payment type (Credit/Debit/Wallet)",
                    "Enter payment credentials",
                    "Verify through bank authentication",
                    "Confirm payment method addition",
                    "Set as default if desired"
                ]
            },
            {
                "user_intent": "How to return item in Blibli?",
                "app": "Blibli",
                "steps": [
                    "Launch Blibli app",
                    "Access 'Pesanan Saya'",
                    "Locate relevant order",
                    "Select 'Return Barang'",
                    "Choose item to return",
                    "Select return category",
                    "Describe the issue",
                    "Attach supporting images",
                    "Submit return request",
                    "Monitor return progress"
                ]
            }
        ]

        # Create multiple variations for each example
        expanded_dataset = []
        variations = [
            ("How do I", "How can I", "What are the steps to", "How to"),
            ("Steps to", "Instructions to", "Process to", "How to"),
            ("in", "on", "using", "with"),
        ]

        for base_example in enhanced_examples:
            # Original version
            expanded_dataset.append(base_example)

            # Create variations
            for i in range(15):  # 15 variations per example
                varied_example = base_example.copy()

                # Vary the user intent
                original_intent = base_example["user_intent"]
                for old_phrase, *alternatives in variations:
                    if old_phrase in original_intent:
                        new_phrase = alternatives[i % len(alternatives)]
                        varied_example["user_intent"] = original_intent.replace(old_phrase, new_phrase)
                        break

                expanded_dataset.append(varied_example)

        # Format for training with better structure
        formatted_texts = []
        for example in expanded_dataset:
            user_intent = example["user_intent"]
            app = example["app"]
            steps = example["steps"]

            # Create numbered steps
            numbered_steps = [f"{i+1}. {step}" for i, step in enumerate(steps)]
            instructions = "\n".join(numbered_steps)

            # Enhanced format with clear delimiters
            text = f"<s>User: {user_intent}<sep>Assistant: {instructions}</s>"
            formatted_texts.append(text)

        # Tokenize with enhanced settings
        tokenized = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length=self.config['max_length'],
            return_tensors="pt"
        )

        # Create dataset
        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].clone()
        })

        # Enhanced train/eval split (80/20)
        train_size = int(0.8 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))

        logger.info(f"Enhanced dataset - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        return train_dataset, eval_dataset

    def create_enhanced_training_args(self, output_dir: str):
        """Enhanced training arguments for better quality"""

        return TrainingArguments(
            output_dir=output_dir,

            # Enhanced training schedule
            num_train_epochs=self.config['epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],

            # Better learning configuration
            learning_rate=self.config['learning_rate'],
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",

            # Enhanced evaluation
            eval_strategy="steps",
            eval_steps=25,
            save_strategy="steps",
            save_steps=50,
            logging_steps=10,

            # Quality improvements
            gradient_accumulation_steps=4,  # Effective batch size = 8
            max_grad_norm=1.0,
            dataloader_num_workers=0,
            remove_unused_columns=False,

            # Model saving
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # Logging
            report_to=[],
            logging_dir=f"{output_dir}/logs",
        )

    def enhanced_train(self, output_dir: str = "./enhanced_m4_model"):
        """Enhanced training pipeline"""

        try:
            # Step 1: Load enhanced model
            self.load_model()

            # Step 2: Create enhanced dataset
            train_dataset, eval_dataset = self.create_enhanced_dataset()

            # Step 3: Enhanced training args
            training_args = self.create_enhanced_training_args(output_dir)

            # Step 4: Enhanced data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8,  # Optimize for tensor cores
            )

            # Step 5: Create enhanced trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )

            # Step 6: Enhanced training
            logger.info("🚀 Starting enhanced training...")
            trainer.train()

            # Step 7: Save enhanced model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)

            logger.info(f"✅ Enhanced training completed! Saved to: {output_dir}")
            return trainer

        except Exception as e:
            logger.error(f"Enhanced training failed: {e}")
            if self.device == "mps":
                torch.mps.empty_cache()
            gc.collect()
            raise e

    def enhanced_inference(self, prompt: str, max_new_tokens: int = 100):
        """Enhanced inference with better generation parameters"""

        if self.model is None:
            logger.error("No model loaded!")
            return

        # Enhanced prompt formatting
        formatted_prompt = f"<s>User: {prompt}<sep>Assistant:"

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        # Enhanced generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,

                # Quality parameters
                temperature=self.config['temperature'],
                top_p=self.config['top_p'],
                do_sample=True,

                # Stop tokens
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,

                # Repetition control
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,

                # Early stopping
                early_stopping=True,
            )

        # Enhanced decoding
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        # Clean up response
        response = response.replace("<sep>", "").strip()

        return response

def main():
    """Enhanced main execution"""
    print("🎯 Enhanced Quality Training for M4")

    # Create enhanced trainer
    trainer = QualityImprovedTrainer()

    try:
        # Enhanced training
        trained_model = trainer.enhanced_train("./models/enhanced_m4")

        # Enhanced testing
        print("\n" + "="*60)
        print("🧪 TESTING ENHANCED MODEL")
        print("="*60)

        enhanced_test_questions = [
            "How do I reset my password in Shopee?",
            "How to track my order in Tokopedia?",
            "Steps to add payment method in Lazada?",
            "How to return an item in Blibli?",
            "How do I change my profile picture in Shopee?"
        ]

        for question in enhanced_test_questions:
            print(f"\n❓ Question: {question}")
            answer = trainer.enhanced_inference(question, max_new_tokens=120)
            print(f"🤖 Answer: {answer}")
            print("-" * 50)

        print("\n✅ Enhanced training and testing completed successfully!")

    except Exception as e:
        print(f"\n❌ Enhanced training error: {e}")

if __name__ == "__main__":
    main()
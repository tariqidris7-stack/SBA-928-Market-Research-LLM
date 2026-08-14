"""
SBA 928 - Market Research LLM Fine-Tuning

Fine-tunes Google FLAN-T5 Small on a structured
market research prompt-response dataset using LoRA/PEFT.
"""

import pandas as pd
import torch

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)


MODEL_NAME = "google/flan-t5-small"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 128


def main():

    print("Loading tokenizer and base model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    # Load train and validation data.
    dataset = load_dataset(
        "csv",
        data_files={
            "train": "data/train.csv",
            "validation": "data/validation.csv"
        }
    )

    def preprocess_function(examples):

        inputs = [
            "You are a market research analyst. "
            "Provide a clear, relevant, and actionable "
            "market research response.\n\n"
            f"Question: {prompt}\n"
            "Answer:"
            for prompt in examples["prompt"]
        ]

        targets = examples["response"]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True
        )

        labels = tokenizer(
            text_target=targets,
            max_length=MAX_TARGET_LENGTH,
            truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # LoRA configuration.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q", "v"]
    )

    model = get_peft_model(
        model,
        lora_config
    )

    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="./model/checkpoints",
        learning_rate=5e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator
    )

    print("Starting model fine-tuning...")

    trainer.train()

    print("Training complete.")

    final_eval = trainer.evaluate()

    print("Final evaluation:")
    print(final_eval)

    # Save trained LoRA adapter.
    adapter_path = "model/sba928_market_research_adapter"

    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Save training logs.
    logs = pd.DataFrame(
        trainer.state.log_history
    )

    logs.to_csv(
        "outputs/training_log.csv",
        index=False
    )

    print("Adapter and training logs saved.")


if __name__ == "__main__":
    main()

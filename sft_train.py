import typer
from datasets import Dataset
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np
import torch

app = typer.Typer()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


@app.command()
def train(
    model_path: str = "../model/Qwen3-0.6B",
    output_dir: str = "model/sft_output",
    data_file: str = "data/sft_data.json",
    gpu: int = 1,
    train_epochs: int = 1,
    batch_size: int = 5,
    learning_rate: float = 2e-5,
):
    print("Available GPUs:", torch.cuda.device_count())

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, device_map=f"cuda:{gpu}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = json.load(open(data_file, "r", encoding="utf-8"))
    split_data = {"train": dataset[:500], "test": dataset[500:]}

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    train_dataset = Dataset.from_list(split_data["train"])
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = Dataset.from_list(split_data["test"])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    id2label = {0: "Answer", 1: "Ask"}
    label2id = {"Answer": 0, "Ask": 1}

    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True, num_labels=2, id2label=id2label, label2id=label2id, device_map=f"cuda:{gpu}")
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy="no",
        eval_strategy="epoch",
        logging_steps=10,
        report_to="swanlab",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting Classification SFT training...")
    trainer.train()
    results = trainer.evaluate()
    print("Evaluation results:", results)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    app()

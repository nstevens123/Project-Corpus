from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
from tensor2tensor.data_generators import text_encoder
import torch
from utils.tokenizer import LatinHFTokenizer
from datetime import datetime
from pathlib import Path

# quick toggle
if True:
    MODEL_PATH = "bert-base-multilingual-cased"
    TOKENIZER_PATH = "bert-base-multilingual-cased"
    TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
else:
    MODEL_PATH = "models/latin_bert"
    TOKENIZER_PATH = "models/subword_tokenizer_latin/latin.subword.encoder"
    TOKENIZER = LatinHFTokenizer(TOKENIZER_PATH)

dataset = load_dataset("json", data_files="data/train.jsonl")
split_ds = dataset["train"].train_test_split(test_size=0.1, seed=42)


# 2. Tokenize both
def tokenize(example):
    return TOKENIZER(
        example["sentence"], truncation=True, padding="max_length", max_length=128
    )


train_set = split_ds["train"].map(tokenize, batched=True)
val_set = split_ds["test"].map(tokenize, batched=True)

# 3. Update TrainingArguments to actually perform evaluation
training_args = TrainingArguments(
    output_dir=f"./results/{Path(MODEL_PATH).name}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%s')}",
    per_device_train_batch_size=8,
    num_train_epochs=15,
    eval_strategy="epoch",  # Run evaluation after every epoch
    logging_strategy="epoch",  # Log stats after every epoch
    save_strategy="epoch",  # Save a checkpoint after every epoch
    load_best_model_at_end=True,  # Keeps the version that did best on the validation set
)

# 4. Pass the validation set to the Trainer
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,  # <--- This tells the model how to "check its homework"
)
trainer.train()
trainer.save_model(
    f"output_models/{Path(MODEL_PATH).name}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%s')}"
)

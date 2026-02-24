import torch
from datasets import load_dataset
from transformers import pipeline
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils.tokenizer import LatinHFTokenizer
from tensor2tensor.data_generators import text_encoder
from pathlib import Path

# quick toggle
if False:
    MODEL_PATH = (
        "output_models/bert-base-multilingual-cased_2026_02_18_23_38_1771475883"
    )
    TOKENIZER_PATH = "bert-base-multilingual-cased"
    TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
else:
    MODEL_PATH = "output_models/latin_bert_2026_02_18_23_31_1771475511"
    TOKENIZER_PATH = "models/subword_tokenizer_latin/latin.subword.encoder"
    TOKENIZER = LatinHFTokenizer(TOKENIZER_PATH)

# 1. Load your JSONL dataset
# Make sure your file is named 'test_data.jsonl' or change the path here
dataset = load_dataset("json", data_files="data/eval.jsonl", split="train")

# 2. LOAD YOUR LOCAL MODEL
model_path = MODEL_PATH
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
model.eval()

# 3. INITIALIZE PIPELINE
# We pass the actual objects (model and tokenizer) instead of strings
pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=TOKENIZER,
    device=0 if torch.cuda.is_available() else -1,
)
# 3. Define the evaluation metric (Accuracy/F1)
accuracy_metric = evaluate.load("accuracy")

# 4. Map the numeric labels (0, 1, 2) to your model's labels if necessary
# If your model was trained with 'LABEL_0', 'LABEL_1', 'LABEL_2'
label_map = {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2}


def evaluate_model(examples):
    # Run inference on the batch
    results = pipe(examples["sentence"])

    # Extract numeric predictions using the map
    predictions = [label_map[res["label"]] for res in results]
    return {"predictions": predictions}


# 5. Run inference on the entire dataset
results_ds = dataset.map(evaluate_model, batched=True, batch_size=8)

# 6. Compute final accuracy
final_score = accuracy_metric.compute(
    predictions=results_ds["predictions"], references=results_ds["label"]
)

# Mapping for readable output
# 0: Negative, 1: Positive, 2: Neutral (adjust based on your specific label logic)
human_labels = {0: "NEG", 1: "POS", 2: "NEU"}

print(f"\n{'#'*20} DATASET PREDICTIONS {'#'*20}")
print(f"{Path(MODEL_PATH).name}")
print(f"{'SENTENCE':<55} | {'ACTUAL':<8} | {'PREDICTED':<10}")
print("-" * 80)

for row in results_ds:
    sentence = row["sentence"]
    actual = row["label"]
    predicted = row["predictions"]

    # Truncate long sentences for display
    display_text = (sentence[:52] + "..") if len(sentence) > 52 else sentence

    # Convert numeric IDs to human-readable strings
    actual_str = human_labels.get(actual, str(actual))
    pred_str = human_labels.get(predicted, str(predicted))

    # Highlight mismatches
    status = "✓" if actual == predicted else "✗"

    print(f"{display_text:<55} | {actual_str:<8} | {pred_str:<10} {status}")

print("-" * 80)
print(f"Total processed: {len(results_ds)}")
print(f"Final Accuracy on Dataset: {final_score['accuracy'] * 100:.2f}%\n")

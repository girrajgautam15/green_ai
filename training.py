import pandas as pd
import torch
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import EarlyStoppingCallback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('task_training.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- 1. Load and Prepare the Data ---
df_sample = pd.read_csv('data_green_ai.csv')

# Encode labels
label_encoder = LabelEncoder()
df_sample['label'] = label_encoder.fit_transform(df_sample['complexity'])

# Save label encoder classes
np.save('label_encoder_classes.npy', label_encoder.classes_)
logger.info(f"Original class distribution:\n{df_sample['label'].value_counts()}")

# Compute class weights for imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(df_sample['label']), y=df_sample['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float)
logger.info(f"Class weights: {class_weights}")

# Split data (15% validation)
train_df, eval_df = train_test_split(df_sample, test_size=0.15, random_state=42, stratify=df_sample['label'])
logger.info(f"Validation class distribution:\n{eval_df['label'].value_counts()}")

# Convert to Dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# --- 2. Tokenization ---
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["user_input"], truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 3. Model Training ---
num_labels = len(label_encoder.classes_)  # 16
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Custom Trainer with class weights
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    per_class_f1 = precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)[2]
    acc = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Per-class F1: {dict(zip(label_encoder.classes_, per_class_f1))}")
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1
    }

# Training arguments
steps_per_epoch = len(train_dataset) // 32
training_args = TrainingArguments(
    output_dir="./task_results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.1,
    warmup_steps=int(0.1 * (len(train_dataset) // 32 * 8)),
    load_best_model_at_end=True,
    save_strategy="steps",
    eval_strategy="steps",
    save_steps=steps_per_epoch,
    eval_steps=steps_per_epoch,
    metric_for_best_model="macro_f1",
    logging_dir='./task_logs',
    logging_steps=steps_per_epoch,
)

# Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train
logger.info("Starting task classification fine-tuning...")
trainer.train()

# Evaluate
logger.info("Evaluating the fine-tuned model...")
evaluation_results = trainer.evaluate()
logger.info(f"Evaluation results: {evaluation_results}")

# Save best model
best_checkpoint = trainer.state.best_model_checkpoint
logger.info(f"Best checkpoint: {best_checkpoint}")
trainer.save_model('task_roberta')
tokenizer.save_pretrained('task_roberta')

# Test set evaluation
train_df, test_df = train_test_split(df_sample, test_size=0.1, random_state=42, stratify=df_sample['label'])
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)
test_results = trainer.evaluate(test_dataset)
logger.info(f"Test results: {test_results}")

# Example prediction
logger.info("--- Example Prediction ---")
new_requests = [
    "Write a SQL query to find all customers from Mumbai with a home loan.",
    "Draft a polite but firm email to a customer whose EMI payment has bounced.",
]
for request in new_requests:
    inputs = tokenizer(request, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
    logger.info(f"Input: '{request}'")
    logger.info(f"  --> Predicted Label: {predicted_label}")
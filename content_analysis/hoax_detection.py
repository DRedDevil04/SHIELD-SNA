import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import classification_report
import numpy as np
import transformers
print(transformers.__version__)
# Load dataset
df = pd.read_csv("../datasets/fetched_reddit_content.csv")
df['text'] = df['clean_title'].fillna('') + " " + df['content'].fillna('')
df['label'] = df['2_way_label']

# Hugging Face dataset
dataset = Dataset.from_pandas(df[['text', 'label']])

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize_function, batched=True)
tokenized = tokenized.train_test_split(test_size=0.2)

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


import inspect
# Training arguments
# Debugging: Print available arguments in TrainingArguments
print("ARGS:", TrainingArguments.__init__.__code__.co_varnames)

# Debugging: Print source file of TrainingArguments class
print("SOURCE:", inspect.getfile(TrainingArguments))

# Now your original code

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

# Train
trainer.train()

# Evaluation
predictions = trainer.predict(tokenized["test"])
preds = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids
print(classification_report(y_true, preds))

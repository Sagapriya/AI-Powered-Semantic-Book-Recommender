from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "facebook/bart-large-mnli"

# Download the model manually
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model downloaded successfully!")

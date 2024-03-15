from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


model_name_or_path = "outputs"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

text = "우울하다"

inputs = tokenizer(text, return_tensors="pt")
output = model.generate(**inputs, do_sample=True)

decoded_output = tokenizer.decode(output, skip_special_tokens=True)
print(decoded_output)

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from torch.utils.data import Dataset
import pandas as pd
import torch
import sys

from transformers import TrainingArguments, Trainer


class Poem_Dataet(Dataset):
    def __init__ (self, train_dataset, tokenizer):
        self.dataset = train_dataset
        self.tokenizer = tokenizer
        self.tokenized_dataset = []

        # tokenize dataset
        for data in self.dataset:
            tokenized_data = self.tokenizer(data, add_special_tokens=True, max_length=512, padding=False, truncation=True, return_tensors=None, return_token_type_ids=False)
            self.tokenized_dataset.append(tokenized_data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.tokenized_dataset[index]
        item["labels"] = item["input_ids"]
        return item


if __name__ == "__main__":
    # set device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model, tokenizer 
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/polyglot-ko-3.8b').to(device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-3.8b")
    tokenizer.pad_token = tokenizer.eos_token
    
    # add special token
    tokenizer.add_special_tokens({'additional_special_tokens': ["[YUN]"]})
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True)

    # freeze layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    for i, m in enumerate(model.gpt_neox.layers):
        # Only un-freeze the last n transformer blocks
        if i >= 24:
            for parameter in m.parameters():
                parameter.requires_grad = True

    for parameter in model.gpt_neox.final_layer_norm.parameters():
        parameter.requires_grad = True

    for parameter in model.embed_out.parameters():
        parameter.requires_grad = True

    # train dataset load & preprocessing
    file_path = "dataset/poem_dataset.csv"

    dataset = pd.read_csv(file_path)
    dataset = dataset[dataset["num_lines"] > 5]
    train_dataset = list(dataset["poem"])

    for i in range(len(train_dataset)):
        train_dataset[i] = train_dataset[i].replace("<yun>", "[YUN]")

    train_data = Poem_Dataet(train_dataset, tokenizer)

    # set TrainingArguments
    training_args=TrainingArguments(
        output_dir="output/",
        overwrite_output_dir=True,
        logging_steps=2000,
        save_steps=2000,
        save_total_limit=1,
        learning_rate= 2e-05,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=15,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        seed=42,
        fp16=True
    )

    # set Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator
    )

    # model train & save
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    with torch.autocast("cuda"):
        trainer.train()
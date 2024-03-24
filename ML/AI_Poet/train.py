from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, DataCollatorWithPadding
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
            data = "<s>" + data + "</s>"
            tokenized_data = self.tokenizer(data, add_special_tokens=True, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
            self.tokenized_dataset.append(tokenized_data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.tokenized_dataset[index]
        input_ids = item["input_ids"].squeeze(0)
        label_ids = item["input_ids"].squeeze(0)
        attention_mask = item["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids
        }


if __name__ == "__main__":
    # set device 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model, tokenizer 
    model = GPT2LMHeadModel.from_pretrained('skt/ko-gpt-trinity-1.2B-v0.5').to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/ko-gpt-trinity-1.2B-v0.5',
        bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

    # add special token
    tokenizer.add_special_tokens({'additional_special_tokens': ["<yun>"]})
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # train dataset load & preprocessing
    file_path = "dataset/poem_dataset.csv"

    dataset = pd.read_csv(file_path)
    dataset = dataset[dataset["num_lines"] > 5]
    train_dataset = list(dataset["poem"])
    train_data = Poem_Dataet(train_dataset, tokenizer)

    # set TrainingArguments
    training_args=TrainingArguments(
        output_dir="output/",
        overwrite_output_dir=True,
        logging_steps=1000,
        save_steps=1000,
        save_total_limit=1,
        learning_rate= 1e-05,
        per_device_train_batch_size=4,
        num_train_epochs=10,
        lr_scheduler_type="linear",
        warmup_steps=1000,
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
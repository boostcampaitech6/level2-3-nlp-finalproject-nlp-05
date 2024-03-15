from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, DataCollatorWithPadding
from torch.utils.data import Dataset
import pandas as pd
import torch

from transformers import TrainingArguments, Trainer


class Poem_Dataet(Dataset):
    def __init__ (self, train_dataset, tokenizer):
        self.dataset = train_dataset
        self.tokenizer = tokenizer
        self.tokenized_dataset = []

        # tokenizing 수행 후, input_ids 저장
        for data in self.dataset:
            tokenized_data = self.tokenizer(data, add_special_tokens=True, max_length=1024, padding="max_length", truncation=True, return_tensors='pt')
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
    # device 정의
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model, tokenizer, data_collator load
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2').to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # train dataset load & preprocessing
    file_path = "dataset/poem_dataset.csv"

    dataset = pd.read_csv(file_path)
    dataset = dataset[dataset["num_lines"] > 5]
    train_dataset = list(dataset["poem"])
    train_data = Poem_Dataet(train_dataset, tokenizer)

    # set TrainingArguments
    training_args=TrainingArguments(
    output_dir="output/version_4",
    overwrite_output_dir=True,
    logging_steps=2000,
    save_steps=2000,
    save_total_limit=1,
    learning_rate= 1e-05,
    per_device_train_batch_size=2,
    num_train_epochs=10,
    lr_scheduler_type="linear",
    warmup_steps=2000,
    seed=42
    )

    # set Trainer
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=data_collator
    )

    # run training
    trainer.train()
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import torch
import re

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
    
def data_preprocessing(df):

    # NaN 제거
    dataset = df.dropna(subset=["poem"]) # 현재 poem쪽에 NaN이 존재해서 처리해줌. TODO : 크롤링할 때, poem==NaN이면 skip하도록 구현

    # 중복 제거
    dataset = dataset.drop_duplicates(["poem"], keep="first")

    # 1000자 이상의 데이터 제거
    dataset = dataset[dataset["poem"].str.len() <= 1000]

    # index 재설정
    dataset = dataset.reset_index(drop=True) 

    # 아래의 패턴들을 전처리
    # 1. "\xa0"
    # 2. "\s" * m (m > 1)
    # 3. "\n\n "
    # 4. "\n" * m (m > 2)
    for i in range(len(dataset)):
        dataset.iloc[i, 2] = dataset.iloc[i, 2].replace(u"\xa0", u"")
        dataset.iloc[i, 2] = re.sub(r"( {2,})", "", dataset["poem"][i])
        dataset.iloc[i, 2] = re.sub(r"\n\n ", "", dataset["poem"][i])
        dataset.iloc[i, 2] = re.sub(r"(\n{3,})", "", dataset["poem"][i])

    return dataset

    

if __name__ == "__main__":
    # device 정의
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model, tokenizer, data_collator load
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2').to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # train dataset load & preprocessing
    file_path = "my_dataset/crawling_poem.csv"

    dataset = pd.read_csv(file_path)
    processed_dataset = data_preprocessing(dataset)
    train_dataset = list(processed_dataset["poem"])
    train_data = Poem_Dataet(train_dataset, tokenizer)

    # set TrainingArguments
    training_args=TrainingArguments(
    output_dir="output/",
    overwrite_output_dir=True,
    logging_steps=500,
    save_steps=500,
    save_total_limit=2,
    learning_rate= 1e-05,
    per_device_train_batch_size=2,
    num_train_epochs=10,
    lr_scheduler_type="linear",
    warmup_steps=500,
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
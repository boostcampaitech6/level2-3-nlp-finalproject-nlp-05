from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, BitsAndBytesConfig
from torch.utils.data import Dataset
import pandas as pd
import torch

from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training


class Poem_Dataet(Dataset):
    def __init__ (self, train_dataset, tokenizer):
        self.dataset = train_dataset
        self.tokenizer = tokenizer
        self.tokenized_dataset = []

        # tokenizing 수행 후, input_ids 저장
        for data in self.dataset:
            data = "[BOS]" + data + "[EOS]"
            tokenized_data = self.tokenizer(data, add_special_tokens=True, max_length=512, padding="max_length", truncation=True, return_tensors='pt')
            self.tokenized_dataset.append(tokenized_data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.tokenized_dataset[index]
        # input_ids = item["input_ids"].squeeze(0)
        # label_ids = item["input_ids"].squeeze(0)
        # attention_mask = item["attention_mask"].squeeze(0)
        item["labels"] = item["input_ids"]
        return item

if __name__ == "__main__":
    # model, tokenizer load
    tokenizer = AutoTokenizer.from_pretrained('kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b',  # or float32 version: revision=KoGPT6B-ryan1.5b
    bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained('kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b',  # or float32 version: revision=KoGPT6B-ryan1.5b
    pad_token_id=tokenizer.eos_token_id, quantization_config=bnb_config, device_map={"":0})
    
    tokenizer.add_special_tokens({'additional_special_tokens': ["[YUN]"]})
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['q_proj', 'fc_out', 'v_proj', 'out_proj', 'lm_head', 'fc_in', 'k_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CASUAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # train dataset load & preprocessing
    file_path = "dataset/poem_dataset.csv"

    dataset = pd.read_csv(file_path)
    dataset = dataset[dataset["num_lines"] > 5]
    train_dataset = list(dataset["poem"])

    for i in range(len(train_dataset)):
        train_dataset[i] = train_dataset[i].replace("<yun>", "[YUN]")

    train_data = Poem_Dataet(train_dataset, tokenizer)

    model.print_trainable_parameters()

    # set TrainingArguments
    training_args=TrainingArguments(
        output_dir="output/version_qlora",
        overwrite_output_dir=True,
        logging_steps=2000,
        save_steps=2000,
        save_total_limit=1,
        learning_rate= 1e-05,
        per_device_train_batch_size=4,
        num_train_epochs=10,
        lr_scheduler_type="linear",
        warmup_steps=2000,
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
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
        model, type(model)
    )

    # run training
    trainer.train()

    model.save_pretrained("output/version_qlora")
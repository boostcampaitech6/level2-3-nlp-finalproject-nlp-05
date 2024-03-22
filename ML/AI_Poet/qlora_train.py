from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from torch.utils.data import Dataset
import pandas as pd
import torch
import sys

from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training


class Poem_Dataet(Dataset):
    def __init__ (self, train_dataset, tokenizer):
        self.dataset = train_dataset
        self.tokenizer = tokenizer
        self.tokenized_dataset = []

        # tokenizing
        for data in self.dataset:
            # data = "[BOS]" + data + "[EOS]"
            tokenized_data = self.tokenizer(data, add_special_tokens=True, max_length=512, padding="max_length", truncation=True, return_tensors=None, return_token_type_ids=False)
            self.tokenized_dataset.append(tokenized_data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.tokenized_dataset[index]
        item["labels"] = item["input_ids"]
        return item

if __name__ == "__main__":
    # model, tokenizer load
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/polyglot-ko-1.3b', return_special_tokens_mask=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained('EleutherAI/polyglot-ko-1.3b', pad_token_id=tokenizer.eos_token_id, 
                                                 quantization_config=bnb_config, device_map={"":0})
    
    tokenizer.add_special_tokens({'additional_special_tokens': ["[YUN]"]})
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
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
        logging_steps=200,
        save_steps=200,
        save_total_limit=1,
        learning_rate= 3e-04,
        per_device_train_batch_size=32,
        num_train_epochs=1,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        seed=42,
        fp16=True,
        ddp_find_unused_parameters=None,
        group_by_length=True,
        optim="adamw_torch"
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

    # model train & save
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    with torch.autocast("cuda"):
        trainer.train()

    model.save_pretrained("output/version_qlora")
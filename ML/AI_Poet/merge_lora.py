from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


if __name__ == "__main__":
    
    base_model = AutoModelForCausalLM.from_pretrained("kakaobrain/kogpt", revision='KoGPT6B-ryan1.5b-float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    base_tokenizer = AutoTokenizer.from_pretrained("kakaobrain/kogpt", revision='KoGPT6B-ryan1.5b-float16', 
        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]')
    
    base_tokenizer.add_special_tokens({'additional_special_tokens': ["[YUN]"]})
    base_model.resize_token_embeddings(len(base_tokenizer))

    lora_model = PeftModel.from_pretrained(base_model, "output/version_qlora/", torch_dtype=torch.float16)

    model = lora_model.merge_and_unload()

    model.save_pretrained("output/merge_lora")
    base_tokenizer.save_pretrained("output/merge_lora")
    print("Complete!!")
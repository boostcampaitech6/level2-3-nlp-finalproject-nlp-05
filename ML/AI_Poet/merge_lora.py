from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


if __name__ == "__main__":
    
    base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-1.3b", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
    
    base_tokenizer.add_special_tokens({'additional_special_tokens': ["[YUN]"]})
    base_model.resize_token_embeddings(len(base_tokenizer))

    lora_model = PeftModel.from_pretrained(base_model, "output/version_qlora/checkpoint-200", torch_dtype=torch.float16)

    model = lora_model.merge_and_unload()

    model.save_pretrained("output/merge_lora")
    base_tokenizer.save_pretrained("output/merge_lora")
    print("Complete!!")
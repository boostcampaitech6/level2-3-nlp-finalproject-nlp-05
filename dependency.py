from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerFast, GPT2LMHeadModel

model = None
tokenizer = None

poem_model = None
poem_tokenizer = None

def load_model_tokenizer(model_path: str):
    
    global model
    global tokenizer
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
def get_model_tokenizer():
    
    global model
    global tokenizer
    
    return model,tokenizer

def load_poem_model_tokenizer(poem_model_path: str):
    
    global poem_model
    global poem_tokenizer
    
    poem_model = GPT2LMHeadModel.from_pretrained(poem_model_path)
    poem_tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>'
    )
    
def get_poem_model_tokenizer():
    global poem_model
    global poem_tokenizer
    
    return poem_model, poem_tokenizer
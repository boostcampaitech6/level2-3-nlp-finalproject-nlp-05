sentence_generator = None
poem_generator = None

def load_sentence_generator(model_path: str):
    import joblib

    global sentence_generator 
    sentence_generator = joblib.load(model_path)

def get_sentence_generator():
    global sentence_generator
    return sentence_generator


def load_poem_generator(model_path: str):
    import joblib

    global poem_generator 
    poem_generator = joblib.load(model_path)

def get_poem_generator():
    global poem_generator
    return poem_generator
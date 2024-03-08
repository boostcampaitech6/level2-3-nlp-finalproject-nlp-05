from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 학습된 모델과 토크나이저의 경로 또는 이름
model_name_or_path = "../output"

# 모델과 토크나이저 불러오기
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 모델을 사용할 준비
text = "우울하다"

# 입력을 모델이 이해할 수 있는 형식으로 변환
inputs = tokenizer(text, return_tensors="pt")

# 모델로 예측 실행
output = model.generate(**inputs)

# 결과 디코딩
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output)
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

speech, sr =librosa.load('test.wav')
speech = librosa.resample(speech, sr, 16000)
input_values = tokenizer(speech, return_tensors='pt').input_values
logits = model(input_values).logits
predict_id = torch.argmax(logits, dim = -1)
transcription = tokenizer.decode(predict_id[0])
print(transcription)
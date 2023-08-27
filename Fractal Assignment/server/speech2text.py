from IPython.display import Audio, display
import warnings
import speech_recognition as sr
import librosa
import torch
import whisper
from transformers import Wav2Vec2ForCTC, AutoProcessor
from time import process_time
from datasets import load_dataset
from evaluate import load

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     warnings.warn("deprecated", DeprecationWarning)
warnings.filterwarnings("ignore")

audio_files = ["crazy_ones.flac", "rocky_balbao.flac"]

def SpeechRecognition_s2t(audio_files):
  transcriptions = []

  for audio_file in audio_files:
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
      audio = r.record(source)
    transcription = r.recognize_google(audio)
    transcriptions.append(transcription)

  return transcriptions


def Wav2Vec2_s2t(audio_files, rate=16000):
  transcriptions = []
  processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
  model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

  for audio_file in audio_files:
    audio, rate = librosa.load(audio_file, sr=rate)
    inputs = processor([audio], sampling_rate=rate, return_tensors="pt")

    with torch.no_grad():
      logits = model(**inputs).logits
      predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(predicted_ids)[0]
    transcriptions.append(transcription)

  return transcriptions

def whisper_s2t(audio_files):
  transcriptions = []
  model = whisper.load_model("base")

  for audio_file in audio_files:
    result = model.transcribe(audio_file)
    transcriptions.append(result["text"])

  return transcriptions

def display_transcript(transcriptions):
  for i in range(len(audio_files)):
    display(Audio(audio_files[i]))
    display(transcriptions[i])



transcriptions = SpeechRecognition_s2t(audio_files)
display_transcript(transcriptions)

transcriptions = Wav2Vec2_s2t(audio_files)
display_transcript(transcriptions)

transcriptions = whisper_s2t(audio_files)
display_transcript(transcriptions)

def speech2text(audio_file):
  model = whisper.load_model("base")
  result = model.transcribe(audio_file)
  return result["text"]

audio_file = "rocky_balbao.mp3"
t = process_time()
transcript = speech2text(audio_file)
elapsed_time = process_time() - t
print(elapsed_time)
display(Audio(audio_file))
display(transcript)


audio_file = "MLKDreamSpeech.mp3"
t = process_time()
transcript = speech2text(audio_file)
elapsed_time = process_time() - t
print(elapsed_time)
display(Audio(audio_file))
display(transcript)

audio_file = "podcast.mp3"
t = process_time()
transcript = speech2text(audio_file)
elapsed_time = process_time() - t
print(elapsed_time)
display(Audio(audio_file))
display(transcript)


def check_asr_accuracy(audio_files, references, model_func):
  predictions = []
  wer_scores = []
  for audio_path, reference in zip(audio_files, references):
    transcript = model_func(audio_path)
    predictions.append(transcript)
    wer = load("wer")
    wer_score = wer.compute(predictions=[transcript], references=[reference])
    wer_scores.append(wer_score)
  final_score = wer.compute(predictions=predictions, references=references)
  return (predictions, wer_scores, final_score)


def whisper_check(audio_file):
  model = whisper.load_model("base")
  result = model.transcribe(audio_file)["text"]
  transcript = result.replace(".", "").replace(",", "").replace("?", "").replace("!", "").upper()[1:]
  return transcript


def Wav2Vec2_check(audio_file, rate=16000):
  processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
  model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

  audio, rate = librosa.load(audio_file, sr=rate)
  inputs = processor([audio], sampling_rate=rate, return_tensors="pt")

  with torch.no_grad():
    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)

  transcription = processor.batch_decode(predicted_ids)[0]

  return transcription


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")



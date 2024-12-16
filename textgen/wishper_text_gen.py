import os
import sys
import tempfile
from pydub import AudioSegment
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT


def convert_to_wav(input_file_path, temp_dir):
    try:
        file_extension = os.path.splitext(input_file_path)[1].lower()
        audio = AudioSegment.from_file(input_file_path, format=file_extension[1:])
        temp_wav_file_path = os.path.join(temp_dir, "temp_audio.wav")
        audio.export(temp_wav_file_path, format="wav")
        print(f"Converted {input_file_path} to {temp_wav_file_path}")
        return temp_wav_file_path
    except Exception as e:
        print(f"Failed to convert {input_file_path} to WAV: {e}")
        raise

def check_cuda_availability():
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        return 0
    else:
        print("CUDA is not available. Falling back to CPU.")
        return -1

def transcribe_audio_to_text(wav_file_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(wav_file_path)
        text = result["text"]
        print("Transcription: ", text)
        return text
    except Exception as e:
        print(f"Failed to transcribe {wav_file_path}: {e}")
        raise

def summarize_with_flan_t5(text, chunk_size=512, overlap=50):
    tokenizer = AutoTokenizer.from_pretrained("Alred/bart-base-finetuned-summarization-cnn-ver2")
    model = AutoModelForSeq2SeqLM.from_pretrained("Alred/bart-base-finetuned-summarization-cnn-ver2")
    
    text_chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        text_chunks.append(text[start:end])
        start += chunk_size - overlap

    summarized_text = ""
    for i, chunk in enumerate(text_chunks):
        print(f"Summarizing chunk {i + 1}/{len(text_chunks)}")
        inputs = tokenizer("summarize: " + chunk, return_tensors="pt", max_length=chunk_size, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=chunk_size, min_length=chunk_size // 2, 
                                    length_penalty=1.5, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summarized_text += summary + " "

    print("Summary: ", summarized_text.strip())
    return summarized_text.strip()

def generate_bullet_points(summary_text):
    sentences = summary_text.split('. ')
    bullet_points = [f"• {sentence.strip()}" for sentence in sentences if sentence.strip()]
    return "\n".join(bullet_points)

def extract_key_phrases(summary_text):
    kw_model = KeyBERT()
    key_phrases = kw_model.extract_keywords(summary_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
    formatted_key_phrases = "\n".join([f"• {phrase[0]}" for phrase in key_phrases])
    print("Key Phrases: ", formatted_key_phrases)
    return formatted_key_phrases

def get_unique_filename(base_name):
    counter = 1
    unique_name = base_name
    while os.path.exists(unique_name):
        name, ext = os.path.splitext(base_name)
        unique_name = f"{name}_{counter}{ext}"
        counter += 1
    return unique_name

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_file> <output_text_file>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_text_file = sys.argv[2]

    if not os.path.exists(input_file_path):
        print(f"The file {input_file_path} does not exist.")
        sys.exit(1)

    if os.path.exists(output_text_file):
        response = input(f"The file {output_text_file} already exists. Do you want to replace it? (y/n): ").strip().lower()
        if response != 'y':
            output_text_file = get_unique_filename(output_text_file)
            print(f"Saving transcription to a new file: {output_text_file}")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            wav_file_path = convert_to_wav(input_file_path, temp_dir)
            transcription_text = transcribe_audio_to_text(wav_file_path)

            summary_text = summarize_with_flan_t5(transcription_text)
            bullet_points = generate_bullet_points(summary_text)
            key_phrases = extract_key_phrases(summary_text)

            with open(output_text_file, "w") as file:
                file.write("Transcription:\n")
                file.write(transcription_text)
                file.write("\n\nSummary:\n")
                file.write(summary_text)
                file.write("\n\nBullet Points:\n")
                file.write(bullet_points)
                file.write("\n\nKey Phrases:\n")
                file.write(key_phrases)

            print(f"Transcription, summary, bullet points, and key phrases saved to {output_text_file}")
        except Exception as e:
            print(f"An error occurred: {e}")

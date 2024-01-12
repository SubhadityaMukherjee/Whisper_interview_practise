from base import *
# from ui import *
from pathlib import Path
import torch
from whisper import whisper

data_path = Path("./data/")

def transcribe_and_statistics(data_path):
    whisper_model = whisper.load_model("small")
    whisper_model = torch.compile(whisper_model)

    transcription_df = files_to_dataframe(data_path)
    transcription_df = prediction_for_all_files(whisper_model, data_path, transcription_df)
    print(transcription_df.head())
    print(text_statistics(transcription_df))

def find_relevant_documents_rag(data_path, question, k=3):
    return rag_retrieval(Path(data_path), question, int(k))

transcribe_and_statistics(data_path)
print(find_relevant_documents_rag(data_path, "Were there any cars?", k=3))
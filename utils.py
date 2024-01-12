import os
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from collections import Counter
from spellchecker import SpellChecker

def files_to_dataframe(data_path):
    # Check if transcription.csv exists in data_path and if not create it and add all files in test_audio folder (ignore hidden files)
    # list_audio_files = [str(data_path/"test_audio"/x) for x in os.listdir(data_path/"test_audio")]
    list_audio_files = [str(data_path/"test_audio"/x) for x in os.listdir(data_path/"test_audio") if x != ".gitkeep"]
    if os.path.isfile(data_path /"transcription.csv"):
        transcription_df = pd.read_csv(data_path / "transcription.csv")
        transcription_df = check_for_changes_in_files(list_audio_files, transcription_df)
    else:
        transcription_df = pd.DataFrame(list_audio_files, columns=["file_name"])
    
    transcription_df.to_csv(data_path / "transcription.csv", index=False)

    return transcription_df       

def check_for_changes_in_files(list_audio_files, transcription_df):
    # Check if there are any new files in test_audio folder
    existing_files = transcription_df["file_name"].tolist()
    new_files = list(set(list_audio_files) - set(existing_files))
    print(f"There are {len(new_files)} new files in test_audio folder")
    temp_df = pd.DataFrame(new_files, columns=["file_name"])
    # Add new files to transcription_df
    transcription_df = pd.concat([transcription_df, temp_df], ignore_index=True)

    # Check if there are any files that have been deleted from test_audio folder
    deleted_files = list(set(existing_files) - set(list_audio_files))
    print(f"There are {len(deleted_files)} files deleted from test_audio folder")
    # Remove deleted files from transcription_df
    transcription_df = transcription_df[~transcription_df["file_name"].isin(deleted_files)]

    return transcription_df

def create_columns_if_not_exists(transcription_df, column_names):
    # Check if columns exist in transcription_df and if not create them
    for column_name in column_names:
        if column_name not in transcription_df.columns:
            transcription_df[column_name] = 0

    return transcription_df

def text_statistics(transcription_df):
    all_text = "\n".join(transcription_df["transcribed"].tolist())
    set_of_words = set(all_text.split(" "))
    max_word_length = max([len(x) for x in set_of_words])

    # counter set 
    counter = Counter(all_text.split(" "))
    most_common_words = counter.most_common(10)

    return f"""Statistics:
        Number of words: {len(all_text.split(" "))}\n
        Number of unique words: {len(set_of_words)}\n
        Max word length: {max_word_length}\n
        Most common words: {most_common_words}\n
        """
def df_to_chroma(data_path):
    # Check if embedding.csv exists in data_path
    if os.path.isfile(data_path /"transcription.csv"):
        loader = CSVLoader(data_path / "transcription.csv", source_column="transcribed", metadata_columns=["file_name"])
        docs = loader.load()
    else:
        return "embedding.csv does not exist in data_path"
    
    text_splitter = RecursiveCharacterTextSplitter()
    all_splits = text_splitter.split_documents(docs)

    return Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def spellcheck_df(df, language="en"):
    spell = SpellChecker()
    df["transcribed"] = df["transcribed"].apply(lambda x: " ".join([spell.correction(word) for word in x.split(" ")]))
    return df
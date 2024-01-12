from utils import *
from whisper import whisper
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from collections import Counter
from langchain_community.llms import GPT4All
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma

def prediction_for_all_files(model, data_path, transcription_df): # uses whisper
    # Read transcription.csv and predict for all rows in the dataframe (if not done already) and save the predictions in a new column called "transcribed"
    column_names = ["transcribed", "language"]
    transcription_df = create_columns_if_not_exists(transcription_df, column_names)
    for i, row in tqdm(transcription_df.iterrows(), total=transcription_df.shape[0]):
            # check if the file has already been transcribed
            if transcription_df.loc[i,"transcribed"] in [0, np.nan, ""]:
                result = model.transcribe(row["file_name"], verbose=False)
                transcription_df.loc[i, "transcribed"] = result["text"]
                transcription_df.loc[i, "language"] = result["language"]
     # spellcheck
    # transcription_df = spellcheck_df(transcription_df)

    transcription_df.to_csv(data_path / "transcription.csv", index=False)
    return transcription_df

def rag_retrieval(data_path, question, k=3): #uses a rag pipelines
    vectorstore = df_to_chroma(data_path)
    docs = vectorstore.similarity_search(question, k = k)
    # vectorstore.persist()
    return f"Length of docs: {len(docs)}",f"Files: {[doc.metadata['file_name'] for doc in docs]}"

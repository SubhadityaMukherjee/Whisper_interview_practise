# Interview Practise - Whisper and RAG
- This code was written as practise for an interview with a company (skipping the name for privacy reasons)

## Features
- Identify changes in audio file directory: see if there are new files, delete old files
- Cache results in a dataframe to avoid reprocessing files. Only run whisper on new files. Delete old files from the dataframe
- Use Whisper to transcribe multiple audio files
- Use RAG to analyse the transcriptions and find relevant audio giles given a search term
- Basic spellcheck
- Text statistics (word count, unique word count, average word length, etc)
- Basic UI to display RAG results

## Relevant files
- `main.py`: main file to run. Calls all other functions
- `whisper` folder: contains all the whisper code (directly copied from the whisper repo)
- `base.py`: contains all the ML code
- `utils.py`: contains all the utility functions that are used throughout the code
- `ui.py`: contains the UI code (uses gradio. But this is not tested yet)

## Note
- This was not meant to be a production ready application, but rather a proof of concept. I spent around a day on it, so there is a lot of room for improvement. 

## How to use
- Clone the repo
- Install the requirements
- Get audio files and save them to the `audio` folder
- run main.py
- If there are random files as output, delete data/transcription.csv and run again
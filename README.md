# Memoize Labs Speech Transcription and Tokenization for LLM Tuning

### Library Requirements

Library requirements are contained within requirements.txt. They can be installed by running:
```bash
pip install -r requirements.txt
```

### Prerequisites to Run Script
An audio file of the target speaker is required to compare against speakers detected within the selected audio file. The longer this file is, the more accurate the speaker recognition will be. Using excessively long files will lead to slightly longer processing times (processing time for this is ~1.5 min/1 hour of recording). 

### Technical Process

All necessary methods are contained within the ```memoizeProcessAudio``` class. 
The selected audio file for tokenization is passed into the ```process``` method. It is sequentially split up and saved into audio subfiles of all segments containing different detected speakers. Each of those audio files are then compared against the reference audio file of the target speaker using the ```speechbrain``` library. OpenAI's ```whisper``` is then used to transcribe the audio files and save them into ```transcription.txt``` into a script-like format, split up by speaker with the target speaker's speech labeled as the target speaker. The script is then tokenized using the ```transformers``` library, but nothing is currently done with these tokens. 

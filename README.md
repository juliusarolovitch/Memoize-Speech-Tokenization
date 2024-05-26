# Memoize Labs Speech Transcription and Tokenization for LLM Tuning

### Library Requirements

Library requirements are contained within requirements.txt. They can be installed by running:
```bash
pip install -r requirements.txt
```

### Prerequisites to Run Script
An audio file of the target speaker is required to compare against speakers detected within the selected audio file. The longer this file is, the more accurate the speaker recognition will be. Using excessively long files will lead to slightly longer processing times (processing time for this is ~1.5 min/1 hour of recording). 

### Technical Process

All necessary methods are contained within the ```python memoizeProcessAudio``` class. 
The selected audio file for tokenization is passed into the ```python process``` method. It is split up and saved into audio subfiles containing all of the different speakers detec

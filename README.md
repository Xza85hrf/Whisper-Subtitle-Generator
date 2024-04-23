# Whisper-Subtitle-Generator
The Whisper Subtitle Generator leverages OpenAI's Whisper model to generate subtitles from audio and video files. This Python-based tool supports multiple languages and employs advanced audio processing techniques to ensure high accuracy in transcription.

## Features

- Support for multiple video and audio formats.
- Multilingual support with language auto-detection.
- Noise reduction and audio normalization.
- Utilizes GPU for accelerated processing (if available).
- Outputs in SRT and VTT subtitle formats.

## Prerequisites

- Python 3.8 or newer.
- ffmpeg for handling video files.
- Whisper, librosa, PyDub, and other Python libraries.

## Installation

Clone this repository:

```bash
git clone https://github.com/Xza85hrf/Whisper-Subtitle-Generator.git
cd whisper-subtitle-generator

Install the required Python packages:
pip install -r requirements.txt

Usage
Run the GUI application:
python gui.py

For command-line usage, you can use the following command:
python main.py --input_file "path/to/your/video.mp4" --output "path/to/output.srt"

Additional flags and options are described in the help:
python main.py --help

Contributing
Contributions are welcome! Please fork the repository and submit pull requests with your improvements.

License
Distributed under the MIT License. See LICENSE for more information.

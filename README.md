# Whisper-Subtitle-Generator

The Whisper Subtitle Generator leverages OpenAI's Whisper model to generate
subtitles from audio and video files. This Python-based tool supports
multiple languages and employs advanced audio processing techniques to
ensure high accuracy in transcription.

## Features

- Support for multiple video and audio formats.
- Multilingual support with language auto-detection.
- Noise reduction and audio normalization.
- Utilizes GPU for accelerated processing (if available).
- Outputs in SRT and VTT subtitle formats.

## Quick Start

```bash
git clone https://github.com/Xza85hrf/Whisper-Subtitle-Generator.git
cd Whisper-Subtitle-Generator
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt

# GUI
python GUI.py

# CLI
python Generate_TL_Sub_Frm_Video.py --input_file "path/to/video.mp4" --output "path/to/output.srt"
```

## Prerequisites

- Python 3.10 or newer.
- `ffmpeg` on `PATH` for video decoding. Install via
  [gyan.dev builds](https://www.gyan.dev/ffmpeg/builds/) on Windows,
  `brew install ffmpeg` on macOS, or `sudo apt install ffmpeg` on Debian-family
  Linux.
- **GPU (optional):** A CUDA-capable NVIDIA GPU accelerates Whisper
  substantially. The project falls back to CPU automatically if CUDA is not
  available — expect inference to be 5-10× slower without a GPU. See the
  [PyTorch install guide](https://pytorch.org/get-started/locally/) to
  replace the default CPU wheel with a CUDA wheel matching your driver.

## Usage

- **GUI:** `python GUI.py` — opens a Tkinter window for picking the input
  media and output directory.
- **CLI:**
  ```bash
  python Generate_TL_Sub_Frm_Video.py --input_file "path/to/video.mp4" \
      --output "path/to/output.srt"
  ```
  Run `python Generate_TL_Sub_Frm_Video.py --help` for all options.

## Contributing

Contributions are welcome — please fork the repository and submit pull
requests with your improvements.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more
information.

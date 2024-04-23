import os
import re
import hashlib
import logging
import pysrt
from datetime import datetime
import whisper
import threading
import librosa
import numpy as np
import torch
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import argparse
from multiprocessing import Pool
import pkg_resources
import psutil
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
from tqdm import tqdm
import soundfile as sf
import concurrent.futures
import inspect
import noisereduce as nr
from pyAudioAnalysis import audioSegmentation as aS
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Constants
REQUIRED_PACKAGES = ['moviepy', 'pysrt', 'whisper', 'librosa', 'numpy', 'torch', 'langdetect', 'psutil', 'argparse',
                     'tqdm', 'PySoundFile']
SUPPORTED_LANGUAGES = ['en', 'fr', 'de', 'nl', 'it', 'es', 'ru', 'zh', 'ar', 'fa', 'ja', 'ko']
DETECTOR_FACTORY_SEED = 0
DEFAULT_CHUNK_SIZE = 10
AUDIO_FILE_NAME = "my_audio.wav"
DEFAULT_NUM_PROCESSES = 6  # Default number of processes for multiprocessing
TEXT_FORMATTING_OPTIONS = {'clean_utterances': True, 'remove_fillers': True}
TEMPERATURE_SETTINGS = {'temperature': 0.0}
SUPPORTED_VIDEO_TYPES = ['mp4', 'avi', 'mkv', 'flv', 'mov']  # Add more formats here
transcription_cache = {}


# Check if all necessary packages are installed
def check_packages():
    for package in REQUIRED_PACKAGES:
        try:
            dist = pkg_resources.get_distribution(package)
            print('{} ({}) is installed'.format(dist.key, dist.version))
        except pkg_resources.DistributionNotFound:
            print('{} is NOT installed'.format(package))
            return False  # return false if any package is not installed
    return True


def check_video_type(video_file):
    """
    Check if video is of supported type
    """
    file_extension = os.path.splitext(video_file)[1][1:]
    if file_extension in SUPPORTED_VIDEO_TYPES:
        return True
    else:
        return False


# Set up logging
def set_up_logging():
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)  # Change this to DEBUG
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)
    file_handler = logging.FileHandler("subtitle_gen.log")
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    return log


logger = set_up_logging()


# Load the Whisper ASR model
def load_whisper_model(model_name, cuda_usage):
    try:
        whisper_model = whisper.load_model(model_name)
        if cuda_usage:
            whisper_model = whisper_model.to("cuda")
        logger.info("Whisper ASR model loaded successfully.")
        return whisper_model
    except Exception as load_model_error:
        logger.error(f"Failed to load Whisper ASR model: {load_model_error}")
        return None


# Read the audio data
def read_audio_data(audio_file, start, stop):
    try:
        audio_data, _ = librosa.load(audio_file, offset=start, duration=stop - start, sr=None)
        audio_data = nr.reduce_noise(y=audio_data, sr=None,
                                     verbose=False)  # Apply noise reduction
        if not isinstance(audio_data, np.ndarray):
            logger.error("Audio data is not a numpy array.")
            return None
        if len(audio_data.shape) > 1:
            logger.error("Audio data is not one-dimensional.")
            return None
        return audio_data.astype(np.float32)
    except Exception as read_audio_data_error:
        logger.error(f"Failed to read audio data: {read_audio_data_error}")
        return None


# Extract audio from the video
def extract_audio(video_file, audio_file):
    if not os.path.exists(audio_file):
        # Check input types
        if not isinstance(video_file, str) or not isinstance(audio_file, str):
            raise TypeError("Both video_file and audio_file should be strings")
        try:
            logger.info("Extracting audio from video...")
            start_time = datetime.now()
            ffmpeg_extract_audio(video_file, audio_file)
            logger.info(f"Audio extraction completed in {datetime.now() - start_time}")

            if not os.path.exists(audio_file):
                logger.error(f"Failed to extract audio from video: the audio file {audio_file} was not created.")
                return False

            # Load the audio data
            audio_data, samplerate = librosa.load(audio_file, sr=None)

            # Check if audio is stereo and convert to mono if necessary
            audio_data = librosa.to_mono(audio_data)

            # Convert the audio data to 32-bit floating point numbers
            audio_data = audio_data.astype(np.float32)

            # Save the converted audio data
            sf.write(audio_file, audio_data, int(samplerate))  # convert samplerate to int
            logger.info("Audio saved successfully.")
        except Exception as extract_audio_error:
            logger.error(f"Failed to extract audio from video: {extract_audio_error}")
            return False  # return false if audio extraction failed
    return True  # return true if audio extraction succeeded


def read_audio_data(audio_file, start, stop):
    try:
        audio_data, _ = librosa.load(audio_file, offset=start, duration=stop - start, sr=None)
        audio_data = nr.reduce_noise(y=audio_data, verbose=False)  # Apply noise reduction

        # Convert the numpy array back to an audio segment
        audio_segment = AudioSegment(audio_data.tobytes(), frame_rate=22050, sample_width=2, channels=1)

        # Split the audio segment into chunks at points of silence
        chunks = split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-40)

        # Do something with the chunks (e.g., transcribe each one separately)

        if not isinstance(audio_data, np.ndarray):
            logger.error("Audio data is not a numpy array.")
            return None
        if len(audio_data.shape) > 1:
            logger.error("Audio data is not one-dimensional.")
            return None
        return audio_data.astype(np.float32)
    except Exception as read_audio_data_error:
        logger.error(f"Failed to read audio data: {read_audio_data_error}")
        return None


def speaker_diarization(audio_file):
    try:
        # The function returns a numpy array where each element is the speaker label for the corresponding short-term frame
        flags = aS.speakerDiarization(audio_file, n_speakers, mid_window=2.0, mid_step=0.2, short_window=0.05,
                                      lda_dim=35)

        # Do something with the flags (e.g., split the audio or the transcription at points where the speaker changes)

    except Exception as speaker_diarization_error:
        logger.error(f"Failed to perform speaker diarization: {speaker_diarization_error}")
        return None


# Transcribe a chunk of audio
def transcribe_chunk(chunk_args):
    start, end, audio_file, language, model_name = chunk_args
    if not isinstance(start, int) or not isinstance(end, int) or not isinstance(audio_file, str):
        raise TypeError("Incorrect types for chunk_args")
    logger.debug(f"Transcribing chunk from {start} to {end}")

    # Ensure start and end indices are integers
    start, end = int(start), int(end)

    # Load the Whisper ASR model
    model = load_whisper_model(model_name, torch.cuda.is_available())
    if model is None:
        logger.error("Failed to load Whisper ASR model. Exiting.")
        return 'Failed to load Whisper ASR model.'

    # Load the audio data for this chunk
    audio_data = read_audio_data(audio_file, start, end)
    if audio_data is None:
        return 'Failed to load audio data.'
    try:
        logger.debug(f"Loading audio data from {start} to {end}")
        audio_data, _ = librosa.load(audio_file, offset=start, duration=end - start, sr=None)
    except Exception as load_audio_error:
        logger.error(f"Failed to load audio data: {load_audio_error}")
        return f'Failed to load audio data: {load_audio_error}'

    # Check if the chunk is already transcribed
    chunk_hash = hashlib.md5(audio_data).hexdigest()
    if chunk_hash in transcription_cache:
        return transcription_cache[chunk_hash]

    # Convert the audio data to 32-bit floating point numbers
    audio_data = audio_data.astype(np.float32)

    # Transcribe the audio data
    decode_options = {"language": language, "word_timestamps": True}
    decode_options.update(TEMPERATURE_SETTINGS)
    try:
        logger.debug("Transcribing audio data")
        chunk_transcript = model.transcribe(audio_data, decode_options)
        logger.debug(f"Chunk transcription completed: {chunk_transcript}")
        if TEXT_FORMATTING_OPTIONS['clean_utterances']:
            chunk_transcript = re.sub(r'\s+', ' ', chunk_transcript).strip()  # Remove excess whitespace
        if TEXT_FORMATTING_OPTIONS['remove_fillers']:
            chunk_transcript = re.sub(r'\s*(uh|um)\s*', ' ', chunk_transcript)  # Remove fillers like 'uh' and 'um'
    except Exception as transcribe_chunk_error:
        logger.error(f"Failed to transcribe chunk: {transcribe_chunk_error}")
        return f'Failed to transcribe chunk: {transcribe_chunk_error}'

    # Cache the transcription
    transcription_cache[chunk_hash] = chunk_transcript

    return chunk_transcript


# Transcribe the audio using the Whisper ASR model
def transcribe_audio_whisper(audio_file, language, model_name, chunk_size=DEFAULT_CHUNK_SIZE, single_chunk_test=False):
    transcript = ''
    total_duration = None
    error_count = 0
    try:
        logger.info("Transcribing audio...")
        start_time = datetime.now()

        # Calculate the total duration of the audio file
        total_duration = librosa.get_duration(path=audio_file)
        logger.debug(f"Total duration of the audio file: {total_duration}")

        # Calculate chunk size based on available memory
        available_memory = psutil.virtual_memory().available * 0.8  # 80% of available memory
        audio_file_size = os.path.getsize(audio_file)
        logger.debug(f"Audio file size: {audio_file_size}")
        logger.debug(f"Available memory: {available_memory}")

        if audio_file_size > available_memory:
            chunk_size = int(total_duration * (available_memory / audio_file_size))
        else:
            chunk_size = int(chunk_size)  # use the provided chunk size if file fits into memory

        logger.debug(f"Chunk size: {chunk_size}")

        # Create a list of chunks
        chunks = [(start, min(start + chunk_size, int(total_duration)), audio_file, language, model_name)
                  for start in range(0, int(total_duration), chunk_size)]
        logger.debug(f"Number of chunks: {len(chunks)}")
        for chunk in chunks:
            print(chunk)

        if single_chunk_test:
            # Transcribe a single chunk synchronously for testing
            chunk = (0, min(chunk_size, int(total_duration)), audio_file, language, model_name)
            transcript = transcribe_chunk(chunk)
            if transcript.startswith('Failed'):
                logger.error(f"Single chunk test failed: {transcript}")
            else:
                logger.info("Single chunk test passed.")
            return transcript, total_duration, 0 if transcript else 1
        # Process the audio file in chunks
        with Pool(processes=num_processes, initializer=init_worker, initargs=(model_name,)) as pool:
            logger.debug("Starting multiprocessing pool for transcription")
            results = pool.map(transcribe_chunk, chunks)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            transcripts = list(executor.map(transcribe_chunk, chunks))
        transcript = ' '.join(transcripts)

        # Filter results and count errors
        error_messages = []
        for result in results:
            if result.startswith("Failed"):
                error_count += 1
                error_messages.append(result)
            else:
                transcript += result

        if error_count > 0:
            logger.error(f"Transcription failed for {error_count} chunks. Error messages:")
            for msg in error_messages:
                logger.error(msg)
        else:
            logger.info(f"Transcription completed in {datetime.now() - start_time}")
    except Exception as transcribe_audio_whisper_error:
        logger.error(f"Failed to transcribe audio: {transcribe_audio_whisper_error}")
        error_count = 1

    return transcript, total_duration, error_count


# Define initializer function for multiprocessing
def init_worker(model_name):
    global model
    model = load_whisper_model(model_name, torch.cuda.is_available())
    if model is None:
        logger.error("Failed to load model. Exiting.")
        exit(1)


# Get the list of supported languages
def get_supported_languages():
    supported_languages = ['en', 'fr', 'de', 'nl', 'it', 'es', 'ru', 'zh', 'ar', 'fa', 'ja', 'ko']
    return supported_languages


# Detect the language of the audio
def detect_language(audio_file):
    detected_languages = []
    for duration in [60, 300, 900]:  # 1 minute, 5 minutes, 15 minutes
        try:
            audio_data, _ = librosa.load(audio_file, duration=duration, sr=None)
            transcript = model.transcribe(audio_data)
            detected_languages.append(detect(transcript))
        except Exception as e:
            logger.error(f"Failed to detect language for duration {duration}: {e}")
            return f"Error: Failed to detect language for duration {duration}"

    # Process the first 30 seconds of the audio file
    try:
        audio_data, _ = librosa.load(audio_file, offset=0, duration=30, sr=None)
        logger.info("Audio data read successfully for language detection.")
    except Exception as e1:
        logger.error(f"Failed to read audio data for language detection: {e1}")
        return "Error: Could not read audio data"

    # Convert the audio data to 32-bit floating point numbers
    audio_data = audio_data.astype(np.float32)

    # Transcribe the audio data
    try:
        transcript = model.transcribe(audio_data)
        logger.info("Audio data transcribed successfully for language detection.")
    except Exception as e2:
        logger.error(f"Failed to transcribe audio data for language detection: {e2}")
        return "Error: Could not transcribe audio data"

    try:
        language = detect(transcript)
        logger.info(f"Language detected successfully: {language}")
    except LangDetectException as e3:
        logger.error(f"Failed to detect language: {e3}")
        language = "Error: Could not detect language"

    most_detected_language = max(detected_languages, key=detected_languages.count)

    if language.startswith("Error"):
        return language
    else:
        return most_detected_language


# Split the transcript into subtitles
def split_transcript(transcript):
    subtitles = transcript.split('\n')
    srt_subs = [pysrt.SubRipItem(index=i, text=sub) for i, sub in enumerate(subtitles, start=1)]
    return srt_subs


# Save the subtitles to a file
def save_subtitles(srt_subs, output_file):
    try:
        with open(output_file, 'w') as f:
            for sub in srt_subs:
                f.write(str(sub))
                f.write('\n')
        logger.info("Subtitles saved successfully.")
    except Exception as save_subtitles_error:
        logger.error(f"Failed to save subtitles: {save_subtitles_error}")


# Main function
def main(input_file, language=None, output_file='subs.srt', languages=None, chunk_size=DEFAULT_CHUNK_SIZE,
         multi_processing=False, single_chunk_test=False, model_name="medium"):
    # Check if all necessary packages are installed
    if not check_packages():
        logger.error("Not all necessary packages are installed. Exiting.")
        exit(1)

    # Get the list of supported languages
    supported_languages = languages if languages else get_supported_languages()
    logger.info(f"Supported languages: {supported_languages}")

    # Initialize progress bar
    progress_bar = tqdm(total=5, desc="Progress", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

    audio_file = "my_audio.wav"
    if input_file.endswith('.mp4') or input_file.endswith('.avi'):  # check if the input file is a video file
        extract_audio(input_file, audio_file)  # extract audio from video file
        if not os.path.exists(audio_file):
            logger.error("Failed to extract audio from video. Stopping script.")
            return
    else:
        audio_file = input_file  # use the input file directly if it is not a video file
    progress_bar.update()

    # Check if video is of supported type
    if check_video_type(input_file):
        print(f"Video of type {os.path.splitext(input_file)[1][1:]} is supported.")
        extract_audio(input_file, audio_file)  # extract audio from video file
        if not os.path.exists(audio_file):
            logger.error("Failed to extract audio from video. Stopping script.")
            return
    else:
        print(f"Video of type {os.path.splitext(input_file)[1][1:]} is not supported.")
        logger.error("Video type not supported. Stopping script.")
        return

    # Load the Whisper ASR model
    model = load_whisper_model(model_name, cuda_usage)
    if model is None:
        logger.error("Failed to load model. Exiting.")
        exit(1)
    print(inspect.signature(model.transcribe))

    # Test the Whisper ASR model with the extracted audio sample
    audio_info = librosa.get_duration(path=audio_file)
    num_frames = int(min(5000, int(round(audio_info * librosa.get_samplerate(audio_file)))))

    # Use the first 5000 frames or the total number of frames if less than 5000
    duration = int(round(num_frames / librosa.get_samplerate(audio_file)))
    test_audio_data, _ = librosa.load(audio_file, duration=duration, sr=None)

    # read the test frames for testing
    # Ensure the input data type and shape match what the Whisper ASR model expects
    if not isinstance(test_audio_data, np.ndarray):
        logger.error("Test audio data is not a numpy array.")
        return
    if len(test_audio_data.shape) != 1:
        logger.error("Test audio data is not one-dimensional.")
        return
    # Transcribe the test audio data
    try:
        test_audio_data = test_audio_data.astype(np.float32)  # Add this line
        test_transcript = model.transcribe(test_audio_data)
        logger.info(f"Test transcript: {test_transcript}")
    except Exception as main_error:
        logger.error(f"Failed to transcribe test audio: {main_error}")
        return

    # Stop the script if an error occurred
    if stop_flag.is_set():
        logger.error("An error occurred. Stopping script.")
        return

    # Detect the language of the audio
    detected_language = language if language else detect_language(audio_file)  # Detect language from audio
    logger.info(f"Detected language: {detected_language}")
    if detected_language not in supported_languages:
        logger.info(f"Detected language {detected_language} is not supported. Stopping transcription.")
        return

    # Transcribe the audio
    transcript, total_duration, error_count = transcribe_audio_whisper(audio_file, detected_language, model_name,
                                                                       int(chunk_size), single_chunk_test)
    # Use Whisper ASR for transcription
    if error_count > 0:
        logger.error(f"Subtitle generation failed with {error_count} errors.")

    if multi_processing:
        pass
    else:
        # Threading transcription
        chunks = [(start, min(start + chunk_size, int(total_duration)), audio_file, detected_language, model_name)
                  for start in range(0, int(total_duration), chunk_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_chunk = {executor.submit(transcribe_chunk, chunk): chunk for chunk in chunks}
            results = []
            for future in concurrent.futures.as_completed(future_to_chunk):
                try:
                    data = future.result()
                    results.append(data)
                except Exception as exc:
                    logger.error(f'Chunk generated an exception: {exc}')
            transcript = "".join(results)

    # Stop the script if an error occurred
    if stop_flag.is_set():
        logger.error("An error occurred. Stopping script.")
        return

    # Split the transcript into subtitles
    srt_subs = split_transcript(transcript)

    # Stop the script if an error occurred
    if stop_flag.is_set():
        logger.error("An error occurred. Stopping script.")
        return

    # Save the subtitles to a file
    save_subtitles(srt_subs, output_file)
    logger.info("Subtitle generation completed successfully!")


def f(x):
    return x * x


# Entry point
if __name__ == '__main__':
    with Pool(processes=4) as pool:  # start 4 worker processes
        result = pool.apply_async(f, (10,))  # evaluate "f(10)" asynchronously in a single process
        print(result.get(timeout=1))  # prints "100" unless your computer is very slow
    DetectorFactory.seed = DETECTOR_FACTORY_SEED
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logger = set_up_logging()
    cuda_usage = torch.cuda.is_available()
    logger.info(
        f"CUDA {'is' if cuda_usage else 'is not'} available. Will {'use' if cuda_usage else 'not use'} GPU for "
        f"transcription.")
    stop_flag = threading.Event()
    model_name = "medium"  # specify your model name here
    model = load_whisper_model(model_name, cuda_usage)
    if model is None:
        logger.error("Failed to load model. Exiting.")
        exit(1)
    print("Available Whisper ASR models: ", whisper.available_models())
    print("Current Whisper ASR model: ", model)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate subtitles from a video or audio file.')
    parser.add_argument('input_file', help='The video or audio file to generate subtitles from.')
    parser.add_argument('-l', '--language', default=None, help='The language of the audio (default: auto-detect).')
    parser.add_argument('-o', '--output', default='output.srt', help='The output file (default: output.srt).')
    parser.add_argument('-c', '--chunk_size', default=DEFAULT_CHUNK_SIZE,
                        help='The chunk size in seconds (default: 10).')
    parser.add_argument('-n', '--num_processes', type=int, default=4, help='Number of processes for multiprocessing.')
    parser.add_argument('-m', '--multi_processing', default=False, action=argparse.BooleanOptionalAction,
                        help='Use multi-processing (default: False).')
    parser.add_argument('-s', '--single_chunk_test', default=False, action=argparse.BooleanOptionalAction,
                        help='Only transcribe a single chunk for testing (default: False).')

    # parser.add_argument('--model_name', type=str, default='medium', help='The name of the model to use.')
    # parser.add_argument('--languages', nargs='+', default=SUPPORTED_LANGUAGES, help='The list of supported languages.')
    args = parser.parse_args()

    # Run the main function
    main(args.input_file, args.language, args.output, args.chunk_size, args.multi_processing, args.single_chunk_test)
    parser.add_argument('--num_processes', type=int, default=4)  # Set a default value
    args = parser.parse_args()

    try:
        main(args.input_file, args.language, args.output if args.output else 'output.srt', args.languages,
             args.chunk_size, args.multi_processing, args.single_chunk_test)
        logger.info("Subtitle generation process completed successfully.")
    except Exception as e:
        logger.error(f"Subtitle generation process failed: {e}")

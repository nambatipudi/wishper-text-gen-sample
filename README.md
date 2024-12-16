# Wisheper Text Gen

This project focuses on generating text using various models and techniques. The main script, `wishper_text_gen.py`, handles the transcription of audio files to text, summarization, bullet point generation, and key phrase extraction.

## Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/nambatipudi/wishper-text-gen.git
    cd wishper-text-gen
    ```

2. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Install FFmpeg:**
    - **Windows:** Download and install from [FFmpeg website](https://ffmpeg.org/download.html).
    - **macOS:** Use Homebrew:
        ```sh
        brew install ffmpeg
        ```
    - **Linux:** Use your package manager, for example:
        ```sh
        sudo apt-get install ffmpeg
        ```

4. **Set Hugging Face Token:**
    Ensure your Hugging Face token is set in the environment path:
    ```sh
    export HUGGINGFACE_TOKEN=<your_huggingface_token>
    ```

## Usage

### Transcribing Audio to Text

To transcribe an audio file to text, use the `wishper_text_gen.py` script:

```sh
python textgen/wishper_text_gen.py <input_audio_file> <output_text_file>
```

If the output file already exists, you will be prompted to either replace it or save the transcription to a new file.

### Example

To run the script with the provided launch configurations in Visual Studio Code:

1. Open the `Run and Debug` pane.
2. Select either "I have a dream" or "We chose the moon" configuration.
3. Click the green play button to start debugging.

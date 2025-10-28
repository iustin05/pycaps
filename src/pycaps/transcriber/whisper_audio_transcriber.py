from .base_transcriber import AudioTranscriber
from typing import Optional, Any
from pycaps.common import Document, Segment, Line, Word, TimeFragment
from pycaps.logger import logger

class WhisperAudioTranscriber(AudioTranscriber):
    def __init__(self, model_size: str = "base", language: Optional[str] = None, model: Optional[Any] = None):
        """
        Transcribes audio using OpenAI's Whisper model.

        Args:
            model_size: Size of the Whisper model to use (e.g., "tiny", "base").
            language: Language of the audio (e.g., "en", "es").
            model: (Optional) A pre-loaded Whisper model instance. If provided, model_size is ignored.
        """
        self._model_size = model_size
        self._language = language
        self._model = model

    def transcribe(self, audio_path: str) -> Document:
        """
        Transcribes the audio file and returns segments with timestamps.
        """
        result = self._get_model().transcribe(
            audio_path,
            word_timestamps=True,
            language=self._language,
            verbose=False # TODO: we should pass our --verbose param here
        )

        if "segments" not in result or not result["segments"]:
            logger().warning("Whisper returned no segments in the transcription.")
            return Document()

        logger().debug(f"Whisper result: {result}")
        document = Document()
        for segment_info in result["segments"]:
            segment_start = float(segment_info["start"])
            segment_end = float(segment_info["end"])
            if segment_start == segment_end:
                segment_end = segment_start + 0.01
            segment_time = TimeFragment(start=segment_start, end=segment_end)
            segment = Segment(time=segment_time)
            line = Line(time=segment_time)
            segment.lines.add(line)

            if not "words" in segment_info or not isinstance(segment_info["words"], list):
                logger().debug(f"Segment '{segment_info['text']}' has no detailed word data.")
                continue

            for word_entry in segment_info["words"]:
                # Ensure 'word' is a string, sometimes Whisper might return non-string for certain symbols.
                word_text = str(word_entry["word"]).strip()
                if not word_text:
                    continue

                word_start = float(word_entry["start"])
                word_end = float(word_entry["end"])
                if word_start == word_end:
                    word_end = word_start + 0.01
                word_time = TimeFragment(start=word_start, end=word_end)
                word = Word(text=word_text, time=word_time)
                line.words.add(word) # so far is everything in one single line (we split it in next steps of the pipeline)

            document.segments.add(segment)
        
        if not document.segments:
            logger().warning("No valid segments were processed from Whisper's transcription.")

        return document 

    def _get_model(self):
        if self._model:
            return self._model
        
        import whisper

        try:
            self._model = whisper.load_model("large")
            return self._model
        except Exception as e:
            raise RuntimeError(
                f"Error loading Whisper model (size: {self._model_size}): {e}\n" 
                f"Ensure Whisper is installed and models are available (or can be downloaded)."
            )

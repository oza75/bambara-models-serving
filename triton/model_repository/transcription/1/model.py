from threading import Thread

import torch
from faster_whisper import WhisperModel
from faster_whisper import tokenizer
import triton_python_backend_utils as pb_utils
import numpy as np
import io
import logging

# Setup logging to output debug information for tracking
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Adding Bambara ('bm') as a language code
tokenizer._LANGUAGE_CODES = tokenizer._LANGUAGE_CODES + ('bm',)


class TritonPythonModel:
    def initialize(self, _):
        """
        Initialize the model. This is called once when the model is loaded.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.language = 'bm'  # Specify Bambara as the language
        # Load the Faster Whisper model for Bambara
        self.model = WhisperModel(
            "/repository/whisper-bambara-ct2",
            compute_type="float16" if torch.cuda.is_available() else "default",
            device=device,
            local_files_only=True,
        )

    def execute(self, requests):
        """
        This method is called for each inference request. It processes input audio and streams transcription.
        """
        for request in requests:
            # Extract input tensor named "INPUTS", which is expected as binary audio data (BYTES)
            inputs_tensor = pb_utils.get_input_tensor_by_name(request, "INPUTS").as_numpy()

            # The binary audio data is located in the first element of the tensor
            audio_bytes = inputs_tensor[0]
            audio_stream = io.BytesIO(audio_bytes)

            # Create response sender for streaming output
            response_sender = request.get_response_sender()

            # Stream transcription segments
            def stream_transcription():
                try:
                    segments_generator, _ = self.model.transcribe(
                        audio_stream,
                        language=self.language,
                        without_timestamps=True,
                        beam_size=5,
                        vad_filter=True,
                    )

                    # Send each segment as it is transcribed
                    for segment in segments_generator:
                        transcribed_segment = segment.text
                        logger.info(f"Transcribed segment: {transcribed_segment}")

                        # Convert the segment text to a UTF-8 encoded byte string
                        out_output = pb_utils.Tensor(
                            "OUTPUT_TEXT", np.array([transcribed_segment.encode('utf-8')], dtype=np.object_)
                        )

                        # Send the partial response back
                        response_sender.send(pb_utils.InferenceResponse(output_tensors=[out_output]))

                except Exception as e:
                    logger.error(f"Error during transcription: {str(e)}")
                    response_sender.send(
                        pb_utils.InferenceResponse(
                            error=pb_utils.TritonError(f"Failed to process the request: {str(e)}")
                        )
                    )

                # After all segments are sent, send the final response
                response_sender.send(
                    None, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )

            # Run transcription in a separate thread to allow streaming
            transcription_thread = Thread(target=stream_transcription)
            transcription_thread.start()
            transcription_thread.join()

        return None

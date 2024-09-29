import logging

import numpy as np
import triton_python_backend_utils as pb_utils
import io
import soundfile as sf
import torch
from tts import BambaraTTS

# Setup logging to output debug information for tracking
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """
    TritonPythonModel serves as an entry point for Triton Inference Server.
    It integrates with the BambaraTTS service to perform text-to-speech processing.
    """

    def initialize(self, args):
        """
        This method is called once when the model is loaded in Triton.
        It initializes the BambaraTTS service.
        """
        try:
            # Initialize the BambaraTTS text-to-speech model
            logger.info("Initializing BambaraTTS model...")
            self.tts = BambaraTTS("/repository/bambara-tts")
            logger.info("BambaraTTS model successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize BambaraTTS model: {str(e)}")
            raise RuntimeError(f"Initialization failed: {str(e)}")

    def execute(self, requests):
        """
        This method is called for each batch of inference requests.
        It processes each text input and returns the generated audio.
        """
        # Initialize a list to hold the responses
        responses = []

        # Iterate through each request in the batch
        for request in requests:
            try:
                # Extract the input text tensor from the request
                input_text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()

                # Convert the byte-encoded input text to a UTF-8 string (batch processing)
                input_text = input_text_tensor[0].decode('utf-8')

                logger.debug(f"Received text for TTS: {input_text}")

                # Use the BambaraTTS model to generate the audio and sampling rate
                sampling_rate, audio_tensor = self.tts.text_to_speech(input_text)

                # Convert the torch tensor (audio) to a NumPy array (float32)
                audio_np = audio_tensor.squeeze().cpu().numpy()

                # Convert the audio tensor to WAV format using an in-memory buffer
                with io.BytesIO() as wav_buffer:
                    sf.write(wav_buffer, audio_np, sampling_rate, format='WAV')
                    wav_buffer.seek(0)
                    audio_wav_bytes = wav_buffer.read()

                logger.debug(f"Generated WAV audio size in bytes: {len(audio_wav_bytes)}")

                audio_output_tensor = pb_utils.Tensor("AUDIO", np.array([audio_wav_bytes], dtype=np.object_))

                # Create the inference response with the audio and sampling rate
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[audio_output_tensor]
                )

                # Add the inference response to the list of responses
                responses.append(inference_response)

            except Exception as e:
                # Log any error that occurs and return an error response
                logger.error(f"Error processing request: {str(e)}")
                inference_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"Failed to process the request: {str(e)}")
                )
                responses.append(inference_response)

        # Return the list of responses
        return responses

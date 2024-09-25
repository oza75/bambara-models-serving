import tritonclient.grpc as grpcclient
import numpy as np
import io
import sys


def transcribe_audio(audio_path: str, server_url: str = "localhost:8001"):
    """
    Transcribe an audio file by sending it to the Triton Inference Server using gRPC streaming.

    Args:
        audio_path (str): Path to the audio file (e.g., .wav) to be transcribed.
        server_url (str): The URL of the Triton server (default: 'localhost:8001').

    Returns:
        None: The transcription results are printed in real-time as they are streamed back.
    """

    # Callback function to handle streaming responses
    def callback(result, error):
        if error is not None:
            print(f"An error occurred: {error}")
        else:
            # Extract the transcribed segment and print it
            transcribed_text = result.as_numpy("OUTPUT_TEXT")[0].decode("utf-8")
            print(f"Transcribed segment: {transcribed_text}")

    # Create a gRPC client to connect to Triton Inference Server
    client = grpcclient.InferenceServerClient(url=server_url)

    # Read the audio file as binary
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    # Prepare input tensor with binary audio data (wrapped in a list for batching support)
    inputs = [
        grpcclient.InferInput("INPUTS", [1], "BYTES")  # [1] indicates batch size of 1
    ]
    # Set audio binary data as input
    inputs[0].set_data_from_numpy(np.array([audio_data], dtype=np.object_))

    # Specify output tensor to retrieve transcription
    outputs = [
        grpcclient.InferRequestedOutput("OUTPUT_TEXT")
    ]

    # Start the bi-directional stream
    client.start_stream(callback=callback)

    # Send the request over the stream
    client.async_stream_infer(model_name="transcription", inputs=inputs, outputs=outputs)

    # Close the stream (important to ensure it's closed after processing)
    client.stop_stream()


if __name__ == "__main__":
    print(transcribe_audio(sys.argv[1], "localhost:8001"))

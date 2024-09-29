import argparse
import numpy as np
import tritonclient.grpc as grpcclient
import wave
import os


def save_wav_file(filename, audio_data, sampling_rate):
    """
    Save the audio data (binary) to a WAV file.

    Args:
        filename (str): Path to save the WAV file.
        audio_data (bytes): Audio data in binary (WAV format).
        sampling_rate (int): Sampling rate of the audio.
    """
    with wave.open(filename, 'wb') as wav_file:
        # Set the parameters for WAV file: nchannels, sampwidth, framerate, nframes, comptype, compname
        wav_file.setnchannels(1)  # Mono channel
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(audio_data)


def text_to_speech_request(text, model_name, output_filename, server_url="localhost:8001"):
    """
    Call Triton Inference Server for text-to-speech inference and save the output.

    Args:
        text (str): Text to be converted to speech.
        model_name (str): The name of the TTS model deployed on Triton.
        output_filename (str): The filename to save the audio output.
        server_url (str): The URL of the Triton server (default: localhost:8001).
    """
    try:
        # Create a Triton gRPC client
        triton_client = grpcclient.InferenceServerClient(url=server_url)

        # Prepare the input tensor
        inputs = [grpcclient.InferInput("TEXT", [1], "BYTES")]
        inputs[0].set_data_from_numpy(np.array([text.encode('utf-8')], dtype=np.object_))

        # Specify the output tensor (expecting audio in bytes)
        outputs = [grpcclient.InferRequestedOutput("AUDIO")]

        # Perform inference
        response = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

        # Extract the audio data and save it as a WAV file
        audio_bytes = response.as_numpy("AUDIO")[0]

        # Assuming sampling rate is returned with audio (can be modified if not the case)
        # If your Triton model returns sampling_rate, add it to the outputs above and extract here
        sampling_rate = 24000  # Set to a default value if not returned by the model

        # Save the audio file
        save_wav_file(output_filename, audio_bytes, sampling_rate)

        print(f"Audio saved to {output_filename}")

    except Exception as e:
        print(f"Failed to process TTS request: {str(e)}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Text-to-Speech CLI tool")
    parser.add_argument("text", type=str, help="Text to convert to speech")
    parser.add_argument("output", type=str, help="Path to save the output WAV file")
    parser.add_argument("--model", type=str, default="tts",
                        help="TTS model name on Triton (default: tts)")
    parser.add_argument("--server_url", type=str, default="localhost:8001",
                        help="Triton server URL (default: localhost:8001)")

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Call Triton endpoint and save output to WAV file
    text_to_speech_request(args.text, args.model, args.output, args.server_url)

import logging
import os
import time

import torch
from typing import Optional, Tuple

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from coqpit import Coqpit
from huggingface_hub import hf_hub_download, hf_hub_url
from preprocessing import BambaraTokenizer

# Setup logging to output debug information for tracking
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BambaraXtts(Xtts):
    """
    A class for the Bambara language that extends the Xtts class.
    Attributes:
        tokenizer: An instance of BambaraTokenizer.
    """

    def __init__(self, config: Coqpit):
        """
        Initializes the BambaraXtts with the provided configuration.
        Args:
            config: An instance of Coqpit containing configuration settings.
        """
        super().__init__(config)
        self.tokenizer = BambaraTokenizer()  # Initialize tokenizer for Bambara
        self.init_models()

    @classmethod
    def init_from_config(cls, config: "XttsConfig", **kwargs) -> "BambaraXtts":
        """
        Class method to create an instance of BambaraXtts from a configuration object.
        Args:
            config: An instance of XttsConfig containing configuration settings.
            **kwargs: Additional keyword arguments.
        Returns:
            An instance of BambaraXtts.
        """
        return cls(config)


class BambaraTTS:
    """
    Bambara Text-to-Speech (TTS) class that initializes and uses a TTS model for the Bambara language.
    Attributes:
        language_code (str): The ISO language code for Bambara.
        checkpoint_repo_or_dir (str): URL or local path to the model checkpoint directory.
        local_dir (str): The directory to store downloaded checkpoints.
        paths (dict): A dictionary of paths to model components.
        config (XttsConfig): Configuration object for the TTS model.
        model (BambaraXtts): The TTS model instance.
    """

    def __init__(self, checkpoint_repo_or_dir: str, local_dir: Optional[str] = None):
        """
        Initialize the BambaraTTS instance.
        Args:
            checkpoint_repo_or_dir: A string that represents either a Hugging Face hub repository
                                    or a local directory where the TTS model checkpoint is located.
            local_dir: An optional string representing a local directory path where model checkpoints
                       will be downloaded. If not specified, a default local directory is used based
                       on `checkpoint_repo_or_dir`.
        The initialization process involves setting up local directories for model components,
        ensuring the model checkpoint is available, and loading the model configuration and tokenizer.
        """

        # Set the language code for Bambara
        self.language_code = 'bm'

        # Store the checkpoint location and local directory path
        self.checkpoint_repo_or_dir = checkpoint_repo_or_dir
        # If no local directory is provided, use the default based on the checkpoint
        self.local_dir = local_dir if local_dir else self.default_local_dir(checkpoint_repo_or_dir)

        # Initialize the paths for model components
        self.paths = self.init_paths(self.local_dir)

        # Load the model configuration from a JSON file
        self.config = XttsConfig()
        self.config.load_json(self.paths['config.json'])

        # Initialize the TTS model with the loaded configuration
        self.model = BambaraXtts(self.config)

        # Set up the tokenizer for the model, using the vocabulary file path
        self.model.tokenizer = BambaraTokenizer(vocab_file=self.paths['vocab.json'])

        # Load the model checkpoint into the initialized model
        self.model.load_checkpoint(
            self.config,
            vocab_path="fake_vocab.json",
            # The 'fake_vocab.json' is specified because the base model class might
            # attempt to override our tokenizer if a vocab file is present
            checkpoint_dir=self.local_dir,
            # use_deepspeed=torch.cuda.is_available()  # Utilize DeepSpeed if CUDA is available
            use_deepspeed=False  # disable because make it fails on huggingface space
        )

        # Move the model to GPU if CUDA is available
        if torch.cuda.is_available():
            self.model.cuda()

        self.log_tokenizer()

    def default_local_dir(self, checkpoint_repo_or_dir: str) -> str:
        """
        Generates a default local directory path for storing the model checkpoint.
        Args:
            checkpoint_repo_or_dir: The original checkpoint repository or directory path.
        Returns:
            The default local directory path.
        """
        if os.path.exists(checkpoint_repo_or_dir):
            return checkpoint_repo_or_dir

        model_path = f"models--{checkpoint_repo_or_dir.replace('/', '--')}"
        local_dir = os.path.join(os.path.expanduser('~'), "bambara_tts", model_path)
        return local_dir.lower()

    @staticmethod
    def init_paths(local_dir: str) -> dict:
        """
        Initializes paths to various model components based on the local directory.
        Args:
            local_dir: The local directory where model components are stored.
        Returns:
            A dictionary with keys as component names and values as file paths.
        """
        components = ['model.pth', 'config.json', 'vocab.json', 'dvae.pth', 'mel_stats.pth']
        return {name: os.path.join(local_dir, name) for name in components}

    def text_to_speech(
            self,
            text: str,
            speaker_reference_wav_path: Optional[str] = None,
            temperature: Optional[float] = 0.1,
            enable_text_splitting: bool = False
    ) -> Tuple[int, torch.Tensor]:
        """
        Converts text into speech audio.
        Args:
            text: The input text to be converted into speech.
            speaker_reference_wav_path: A path to a reference WAV file for the speaker.
            temperature: The temperature parameter for sampling.
            enable_text_splitting: Flag to enable or disable text splitting.
        Returns:
            A tuple containing the sampling rate and the generated audio tensor.
        """
        if speaker_reference_wav_path is None:
            speaker_reference_wav_path = f"{self.checkpoint_repo_or_dir}/references/male_3.wav"
            self.log(f"Using default speaker reference {self.checkpoint_repo_or_dir}/references/male_3.wav")

        self.log("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[speaker_reference_wav_path]
        )

        self.log("Starting inference...")
        start_time = time.time()
        out = self.model.inference(
            text,
            self.language_code,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature,
            enable_text_splitting=enable_text_splitting
        )
        end_time = time.time()

        audio = torch.tensor(out["wav"]).unsqueeze(0).cpu()
        sampling_rate = torch.tensor(self.config.model_args.output_sample_rate).cpu().item()

        self.log(f"Speech generated in {end_time - start_time:.2f} seconds.")

        return sampling_rate, audio

    def log(self, message: str):
        """
        Logs a message to the console with a uniform format.
        Args:
            message: The message to be logged.
        """
        logger.info(f"[BambaraTTS] {message}")

    def log_tokenizer(self):
        """
        Logs the tokenizer information.
        """
        self.log(f"Tokenizer: {self.model.tokenizer}")

from collections import defaultdict

import numpy as np
import onnxruntime
import triton_python_backend_utils as pb_utils
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.pipelines import pipeline
import logging
import torch
from transformers import AutoTokenizer

# Setup logging to output debug information for tracking
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TritonPythonModel:
    """
    TritonPythonModel is the entry point for the translation service within the Triton Inference Server.
    It uses the Hugging Face Optimum pipeline to perform translation using an ONNX model.
    """

    def initialize(self, args):
        """
        This method is called once when the model is loaded in Triton.
        Initializes the translation pipeline with the ONNX model.

        Args:
            args (dict): Triton model arguments, unused here.
        """
        try:
            # Initialize the Hugging Face translation pipeline using the ONNX model
            logger.info("Initializing translation pipeline with ONNX model...")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            session_options = onnxruntime.SessionOptions()
            session_options.log_severity_level = 0
            session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

            checkpoint = "/repository/nllb-600M-mt-french-bambara-v2.onnx"
            model = ORTModelForSeq2SeqLM.from_pretrained(
                checkpoint,
                use_io_binding=True,
                use_cache=True,
                provider="CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider",
                session_options=session_options,
            )
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)

            self.pipeline = pipeline(
                "translation",
                model=model,
                tokenizer=tokenizer,
                accelerator="ort",
                device=device,
                max_length=512
            )
            logger.info("Translation pipeline successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize translation pipeline: {str(e)}")
            raise RuntimeError(f"Initialization failed: {str(e)}")

    def execute(self, requests):
        """
        This method is called for each batch of inference requests.
        All requests in the batch should be processed together to take advantage of dynamic batching.
        """
        # Initialize a list to hold the responses
        responses = []

        # Collect all inputs from the batch of requests
        all_inputs = []

        # Collect inputs from all the requests and store the references to request objects
        for request in requests:
            inputs_tensor = pb_utils.get_input_tensor_by_name(request, "INPUTS").as_numpy()
            all_inputs.append(inputs_tensor)

        # Stack all the inputs together into a 3D NumPy array (batch_size, 3)
        all_inputs = np.vstack(all_inputs)

        # Vectorized decoding of byte strings to utf-8 strings (instead of looping)
        all_inputs = np.char.decode(all_inputs.astype(np.bytes_), encoding='utf-8')

        # Now we have all batched inputs together in `all_inputs`
        logger.debug(f"Received batched inputs for translation: {all_inputs}")

        # Group texts by (src_lang, tgt_lang) for efficient translation
        grouped_inputs = defaultdict(list)
        index_map = defaultdict(list)

        # Using NumPy for vectorized string matching to group inputs by language pairs
        src_langs = all_inputs[:, 1]
        tgt_langs = all_inputs[:, 2]
        texts = all_inputs[:, 0]

        # Combine src_lang and tgt_lang into a single string for efficient grouping, separated by ::
        lang_pairs = np.char.add(src_langs, np.char.add(["::"] * len(tgt_langs), tgt_langs))

        # Populate the grouped_inputs dictionary and index_map
        unique_lang_pairs, group_indices = np.unique(lang_pairs, return_inverse=True)

        for i, unique_pair in enumerate(unique_lang_pairs):
            # Group texts by their (src_lang, tgt_lang) pairs
            grouped_inputs[unique_pair].extend(texts[group_indices == i])
            index_map[unique_pair].extend(np.where(group_indices == i)[0])

        translated_texts = np.empty(len(all_inputs), dtype=object)  # Prepare empty array to store translations

        # Process each group of texts sharing the same src_lang and tgt_lang
        for unique_pair, texts in grouped_inputs.items():
            lang_pair = unique_pair.split("::")
            src_lang, tgt_lang = lang_pair[0], lang_pair[1]
            logger.debug(f"Translating {len(texts)} texts from {src_lang} to {tgt_lang}")

            # Perform batch translation using the pipeline
            results = self.pipeline(texts, src_lang=src_lang, tgt_lang=tgt_lang, max_length=512)

            # Extract the translated text and store it in the right order using index_map
            # Vectorized assignment of translated results into the final output array using index_map
            translated_texts[index_map[unique_pair]] = [res['translation_text'] for res in results]

        # Log the translated output
        logger.debug(f"Translated texts (ordered): {translated_texts}")

        # Now we need to generate one response for each request in the batch
        for i, request in enumerate(requests):
            # Extract the translation corresponding to this request
            translated_text = translated_texts[i]

            # Create a response for this specific request
            output_text_tensor = pb_utils.Tensor("OUTPUT_TEXT", np.array([translated_text], dtype=np.object_))
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_text_tensor])

            # Append the response for this request
            responses.append(inference_response)

        return responses

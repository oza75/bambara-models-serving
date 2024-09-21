import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained("/nllb-600M-tokenizer")

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get the input tensor by name (TEXT) and convert it into a string
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()

            # Convert byte array to string (handle multiple batches if present)
            input_texts = [input_bytes[0].decode('utf-8') for input_bytes in input_tensor]

            tokenized = self.tokenizer._build_translation_inputs(
                input_texts,
                return_tensors="np",
                padding="max_length",
                max_length=512,
                truncation=True,
                src_lang="fr",
                tgt_lang="en",
            )
            input_ids = tokenized['input_ids'].astype("int64")
            # attention_mask = tokenized['attention_mask'].astype("int64")

            input_ids_tensor = pb_utils.Tensor("input_ids", input_ids)
            # attention_mask_tensor = pb_utils.Tensor("attention_mask", attention_mask)

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[input_ids_tensor])
            )
        return responses

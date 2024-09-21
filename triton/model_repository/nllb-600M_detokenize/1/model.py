import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained("/nllb-600M-tokenizer")

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get the logits tensor from the previous model
            output_ids = pb_utils.get_input_tensor_by_name(request, "output_ids").as_numpy()

            # Decode the tokens back to text using batch_decode
            decoded_texts = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)

            # Prepare the output tensor
            decoded_texts_tensor = pb_utils.Tensor("TEXT", np.array(decoded_texts, dtype=object))

            # Send the decoded texts as response
            responses.append(pb_utils.InferenceResponse(output_tensors=[decoded_texts_tensor]))

        return responses

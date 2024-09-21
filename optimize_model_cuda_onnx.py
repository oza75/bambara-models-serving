import onnxruntime
from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.pipelines import pipeline
from transformers import AutoTokenizer

TRANSLATION_MODEL_CHECKPOINT = "oza75/nllb-600M-mt-french-bambara"
SAVE_DIR = "./onnx/nllb-600M-mt-french-bambara"

session_options = onnxruntime.SessionOptions()
session_options.log_severity_level = 0

ort_model = ORTModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL_CHECKPOINT,
    export=True,
    use_io_binding=True,
    provider="CUDAExecutionProvider",
    session_options=session_options
)
tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_CHECKPOINT)

ort_model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("Building engine for a short sequence...")
translator = pipeline("translation", model=ort_model, tokenizer=tokenizer, accelerator="ort", device="cuda")
translated = translator("Hi how are you ?", src_lang="eng_Latn", tgt_lang="bam_Latn")
print(f"Translated short sentence 'Hi how are you ?' give {translated}")

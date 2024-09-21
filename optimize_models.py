from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

TRANSLATION_MODEL_CHECKPOINT = "oza75/nllb-600M-mt-french-bambara"
SAVE_DIR = "./onnx"

ort_model = ORTModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_CHECKPOINT, export=True)
tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_CHECKPOINT)

ort_model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
quantizer = ORTQuantizer.from_pretrained(ort_model)

quantizer.quantize(save_dir=SAVE_DIR, quantization_config=qconfig)

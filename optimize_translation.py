import onnxruntime
from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.pipelines import pipeline
from transformers.pipelines import TranslationPipeline
from transformers import NllbTokenizer, M2M100ForConditionalGeneration
from transformers import AutoTokenizer

TRANSLATION_MODEL_CHECKPOINT = "oza75/nllb-600M-mt-french-bambara"
SAVE_DIR = "./onnx/nllb-600M-mt-french-bambara"

provider_options = {
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "{SAVE_DIR}/trt_caches/nllb-600M-mt-french-bambara"
}

session_options = onnxruntime.SessionOptions()
session_options.log_severity_level = 0

ort_model = ORTModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL_CHECKPOINT,
    export=True,
    use_cache=False,
    provider="TensorrtExecutionProvider",
    provider_options=provider_options,
    session_options=session_options
)

tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_CHECKPOINT)

ort_model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("Building engine for a short sequence...")
translator = pipeline("translation", model=ort_model, tokenizer=tokenizer, accelerator="ort", device="cuda")
translated = translator("Hi", src_lang="eng_Latn", tgt_lang="bam_Latn")
print(f"Translated short sentence 'Hi' give {translated}")

print("Building engine for a long sequence...")
long_sentence = """
In the ever-evolving landscape of modern technology, artificial intelligence has emerged as one of the most transformative and disruptive forces in a wide variety of industries, ranging from healthcare and finance to entertainment and education, fundamentally changing how humans interact with machines and how machines, in turn, respond to the vast and complex datasets provided by the human experience, and while the development of machine learning models, such as BERT, has facilitated more efficient processing of natural language, these advancements also come with their own set of challenges, particularly when it comes to ensuring that these models are not only accurate and efficient but also ethical and aligned with societal values, raising important questions about the implications of AI on privacy, security, bias, and fairness, all of which must be carefully considered as we move forward into a future where AI systems are likely to become even more integrated into daily life, potentially affecting everything from the way people work, communicate, and consume information to the way governments and organizations make decisions that impact millions of individuals around the world, and as the complexity of these systems grows, so too does the need for robust frameworks for accountability, transparency, and oversight, which will require collaboration not only between AI researchers and developers but also between policymakers, ethicists, and the broader public, all working together to ensure that the benefits of artificial intelligence are shared equitably and that the risks are managed in ways that protect human rights and uphold democratic values, because without such coordination, the rapid pace of AI innovation could lead to unintended consequences, exacerbating existing inequalities or creating new forms of social, economic, and political instability that could be difficult, if not impossible, to reverse once they have been set in motion.
"""
translated = translator(long_sentence, src_lang="eng_Latn", tgt_lang="bam_Latn")
print(f"Translated long sentence give {translated}")

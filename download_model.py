import bentoml

translation_model = "oza75/nllb-600M-mt-french-bambara-onnx-cuda"
bentoml.transformers.import_model("nllb-600M-mt-bam", translation_model, clone_repository=True)

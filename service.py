from __future__ import annotations

import os
from collections import defaultdict
from typing import List, Dict
import logging
import bentoml
from pydantic import BaseModel

TRANSLATION_MODEL_NAME = "nllb-600m-mt-bam:latest"
logger = logging.getLogger("bentoml")


class BatchTranslateItem(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str


@bentoml.service(resources={"gpu": 1, "cpu": 4}, traffic={"timeout": 120}, workers=2)
class BatchTranslationService:
    model_ref = bentoml.transformers.get(TRANSLATION_MODEL_NAME)

    def __init__(self):
        import torch
        from optimum.pipelines import pipeline
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer
        logger.info(f"LD_LIBRARY_PATH is {os.environ.get('LD_LIBRARY_PATH')}")
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            self.model_ref.path,
            use_io_binding=True,
            provider="CUDAExecutionProvider",
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_ref.path)

        self.translator = pipeline(
            "translation",
            model=ort_model,
            tokenizer=tokenizer,
            max_length=512,
            device=device,
            accelerator="ort"
        )

    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=32, max_latency_ms=5000)
    def predict(self, items: List[BatchTranslateItem]) -> List[str]:
        """
        Translates a batch of text items by grouping them based on the source and target languages,
        while preserving the input order.

        Args:
        - items (List[BatchTranslateItem]): List of texts along with source and target language information.

        Returns:
        - List[str]: A list of translated texts in the same order as the input.
        """
        logger.info(f"Batch translate received: {items}")
        # Initialize a result list of None to store translations in the correct order
        translations = [''] * len(items)

        # Group texts by (src_lang, tgt_lang) while keeping track of their original indices
        grouped_items = defaultdict(list)

        for idx, item in enumerate(items):
            grouped_items[(item.src_lang, item.tgt_lang)].append((idx, item.text))

        logger.info(f"Batch translate group: {grouped_items}")

        # Translate each group and store the results in the correct positions
        for (src_lang, tgt_lang), texts_with_indices in grouped_items.items():
            indices, texts = zip(*texts_with_indices)  # Unpack indices and texts in one go
            results = self.translator(list(texts), src_lang=src_lang, tgt_lang=tgt_lang)

            # Directly place the translated results back into the correct positions
            for i, result in zip(indices, results):
                translations[i] = result["translation_text"]

        logger.info(f"Batch translations: {translations}")

        return translations


@bentoml.service(resources={"cpu": 2}, traffic={"timeout": 120}, workers=10)
class TranslationService:
    batch = bentoml.depends(BatchTranslationService)

    @bentoml.api()
    async def translate(self, text: str, src_lang: str, tgt_lang: str) -> Dict[str, str]:
        result = await self.batch.to_async.predict(
            [BatchTranslateItem(text=text, src_lang=src_lang, tgt_lang=tgt_lang)]
        )
        return {"translated_text": result[0]}

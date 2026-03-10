"""Translation service for English to Hindi. Supports NLLB and IndicTrans2."""

import os
from pathlib import Path
from typing import Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Apply IndicTrans2 compat patches for transformers 5.x
def _patch_indic_trans():
    """Patch cached IndicTrans2 modules for transformers 5.x compatibility."""
    cache_root = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    if not cache_root.exists():
        return
    # Patch tokenizer: reorder __init__ to call super().__init__ before setting *_id attrs
    for p in cache_root.rglob("tokenization_indictrans.py"):
        try:
            text = p.read_text(encoding="utf-8")
            if "object.__setattr__" in text and "Call super().__init__ early" in text:
                continue
            old = """        # Store token content directly instead of accessing .content
        self.unk_token = (
            hasattr(unk_token, "content") and unk_token.content or unk_token
        )
        self.pad_token = (
            hasattr(pad_token, "content") and pad_token.content or pad_token
        )
        self.eos_token = (
            hasattr(eos_token, "content") and eos_token.content or eos_token
        )
        self.bos_token = (
            hasattr(bos_token, "content") and bos_token.content or bos_token
        )"""
            new = """        # Store token content (object.__setattr__ avoids parent __setattr__ before super().__init__)
        _unk = hasattr(unk_token, "content") and unk_token.content or unk_token
        _pad = hasattr(pad_token, "content") and pad_token.content or pad_token
        _eos = hasattr(eos_token, "content") and eos_token.content or eos_token
        _bos = hasattr(bos_token, "content") and bos_token.content or bos_token
        object.__setattr__(self, "unk_token", _unk)
        object.__setattr__(self, "pad_token", _pad)
        object.__setattr__(self, "eos_token", _eos)
        object.__setattr__(self, "bos_token", _bos)"""
            if old in text:
                text = text.replace(old, new)
                # Reorder: super().__init__ before SPM load and token IDs
                old2 = """        # Load SPM models
        self.src_spm = self._load_spm(self.src_spm_fp)
        self.tgt_spm = self._load_spm(self.tgt_spm_fp)

        # Initialize current settings
        self._switch_to_input_mode()

        # Cache token IDs
        self.unk_token_id = self.src_encoder[self.unk_token]
        self.pad_token_id = self.src_encoder[self.pad_token]
        self.eos_token_id = self.src_encoder[self.eos_token]
        self.bos_token_id = self.src_encoder[self.bos_token]

        super().__init__("""
                new2 = """        # Call super().__init__ early so _special_tokens_map exists
        super().__init__("""
                if old2 in text:
                    text = text.replace(old2, new2)
                    text = text.replace(
                        "            **kwargs,\n        )\n\n    def add_new_language_tags",
                        """            **kwargs,
        )

        # Load SPM models and finish setup (after super().__init__)
        self.src_spm = self._load_spm(self.src_spm_fp)
        self.tgt_spm = self._load_spm(self.tgt_spm_fp)
        self._switch_to_input_mode()
        self.unk_token_id = self.src_encoder[self.unk_token]
        self.pad_token_id = self.src_encoder[self.pad_token]
        self.eos_token_id = self.src_encoder[self.eos_token]
        self.bos_token_id = self.src_encoder[self.bos_token]

    def add_new_language_tags"""
                    )
                    p.write_text(text, encoding="utf-8")
                    logger.info("patched_indic_trans_tokenizer", path=str(p))
        except Exception as e:
            logger.debug("patch_indic_trans_skip", path=str(p), error=str(e))
    # Patch config: optional ONNX import (transformers 5.x removed transformers.onnx)
    for p in cache_root.rglob("configuration_indictrans.py"):
        try:
            text = p.read_text(encoding="utf-8")
            if "_ONNX_AVAILABLE" in text:
                continue
            old = """from transformers import PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast
from transformers.onnx.utils import compute_effective_axis_dimension
from transformers.utils import TensorType, is_torch_available"""
            new = """from transformers import PreTrainedTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import TensorType, is_torch_available

try:
    from transformers.onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast
    from transformers.onnx.utils import compute_effective_axis_dimension
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False
    OnnxConfig = object
    OnnxSeq2SeqConfigWithPast = object"""
            if old in text:
                text = text.replace(old, new)
                text = text.replace(
                    "class IndicTransOnnxConfig(OnnxSeq2SeqConfigWithPast):",
                    "class IndicTransOnnxConfig(OnnxSeq2SeqConfigWithPast if _ONNX_AVAILABLE else object):",
                )
                p.write_text(text, encoding="utf-8")
                logger.info("patched_indic_trans_config", path=str(p))
        except Exception as e:
            logger.debug("patch_indic_trans_config_skip", path=str(p), error=str(e))
    # Patch model: tie_weights must accept recompute_mapping (transformers 5.x)
    for p in cache_root.rglob("modeling_indictrans.py"):
        try:
            text = p.read_text(encoding="utf-8")
            if "recompute_mapping" in text and "def tie_weights" in text:
                continue
            old = "    def tie_weights(self):"
            new = "    def tie_weights(self, missing_keys=None, recompute_mapping=True):"
            if old in text and new not in text:
                p.write_text(text.replace(old, new), encoding="utf-8")
                logger.info("patched_indic_trans_model", path=str(p))
        except Exception as e:
            logger.debug("patch_indic_trans_model_skip", path=str(p), error=str(e))

_patch_indic_trans()

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)

_translation_model = None

# Language codes (eng_Latn / hin_Deva work for both NLLB and IndicTrans2)
SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"

# NLLB models: ungated, no trust_remote_code, transformers-native
NLLB_MODEL_IDS = ("facebook/nllb-200-distilled-600M", "facebook/nllb-200-1.3B")


def _get_device() -> str:
    """Detect available compute device."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _is_nllb_model(model_id: str) -> bool:
    """Check if model is NLLB (no trust_remote_code, native transformers)."""
    return model_id.startswith("facebook/nllb") or "nllb" in model_id.lower()


def _load_translation_model():
    """Lazy load translation model. Prefers NLLB (reliable); falls back to IndicTrans2."""
    global _translation_model
    if _translation_model is not None:
        return _translation_model

    device = _get_device()
    model_id = settings.translation_model
    token = settings.hf_token or os.environ.get("HF_TOKEN")

    try:
        # Use NLLB when specified: ungated, no IndicTransTokenizer compatibility issues
        if _is_nllb_model(model_id):
            return _load_nllb_model(model_id, device, token)
        return _load_indic_trans_model(model_id, device, token)
    except Exception as e:
        logger.error("translation_model_load_failed", error=str(e))
        raise


def _load_nllb_model(model_id: str, device: str, token: Optional[str]):
    """Load NLLB model (transformers-native, no trust_remote_code)."""
    global _translation_model
    cache_dir = str(settings.models_path / "nllb")
    logger.info("loading_translation_model", model=model_id, device=device, backend="nllb")

    import torch
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        token=token,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        token=token,
        torch_dtype=dtype,
    ).to(device)

    _translation_model = {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "backend": "nllb",
    }
    return _translation_model


def _load_indic_trans_model(model_id: str, device: str, token: Optional[str]):
    """Load IndicTrans2 model (trust_remote_code; may have compat issues with transformers 5.x)."""
    global _translation_model
    cache_dir = str(settings.models_path / "indic_trans")
    logger.info("loading_translation_model", model=model_id, device=device, backend="indic_trans")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        token=token,
    )
    if not hasattr(tokenizer, "_special_tokens_map") or tokenizer._special_tokens_map is None:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        tokenizer._special_tokens_map = dict.fromkeys(
            PreTrainedTokenizerBase.SPECIAL_TOKENS_ATTRIBUTES
        )
    if not hasattr(tokenizer, "_extra_special_tokens"):
        tokenizer._extra_special_tokens = []

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=True,
        token=token,
    ).to(device)

    processor = None
    try:
        from IndicTransToolkit.processor import IndicProcessor

        processor = IndicProcessor(inference=True)
    except ImportError:
        logger.warning("IndicProcessor not found, using raw tokenizer")

    _translation_model = {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "processor": processor,
        "backend": "indic_trans",
    }
    return _translation_model


class TranslationService:
    """Translate English text to Hindi."""

    def __init__(self, model_id: Optional[str] = None):
        """Initialize with optional model override."""
        self.model_id = model_id or settings.translation_model

    def translate_batch(
        self, texts: list[str], src_lang: str = SRC_LANG, tgt_lang: str = TGT_LANG
    ) -> list[str]:
        """
        Translate a batch of texts from English to Hindi.

        Args:
            texts: List of English texts.
            src_lang: Source language code (eng_Latn for English).
            tgt_lang: Target language code (hin_Deva for Hindi).

        Returns:
            List of Hindi translations.
        """
        if not texts:
            return []
        mdl = _load_translation_model()
        backend = mdl.get("backend", "indic_trans")

        if backend == "nllb":
            return self._translate_batch_nllb(mdl, texts, src_lang, tgt_lang)
        return self._translate_batch_indic_trans(mdl, texts, src_lang, tgt_lang)

    def _translate_batch_nllb(
        self, mdl: dict, texts: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        """Translate using NLLB (forced_bos_token_id for target language)."""
        tokenizer = mdl["tokenizer"]
        model = mdl["model"]
        device = mdl["device"]
        batch_size = 4 if device == "cuda" else 2
        results = []

        tokenizer.src_lang = src_lang
        tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                inputs = tokenizer(
                    batch,
                    padding="longest",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_lang_id,
                    max_length=256,
                    num_beams=5,
                )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                # Ensure the language tag isn't accidentally included in the returned text natively by NLLB 
                # (since transformers skip_special_tokens=True natively misses Language tags that aren't marked as special).
                decoded = [d.replace(tgt_lang, "").strip() for d in decoded]
                results.extend(decoded)
            except Exception as e:
                logger.error("translation_batch_failed", batch_idx=i, error=str(e))
                results.extend(batch)
        return results

    def _translate_batch_indic_trans(
        self, mdl: dict, texts: list[str], src_lang: str, tgt_lang: str
    ) -> list[str]:
        """Translate using IndicTrans2 with optional IndicProcessor."""
        tokenizer = mdl["tokenizer"]
        model = mdl["model"]
        device = mdl["device"]
        processor = mdl.get("processor")
        batch_size = 4 if device == "cuda" else 2
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                if processor is not None:
                    batch_processed = processor.preprocess_batch(
                        batch, src_lang=src_lang, tgt_lang=tgt_lang
                    )
                else:
                    # Manually form the IndicTrans2 prompt: "<2tgt_lang> source_text"
                    # The actual format depends on the specific IndicTrans2 version,
                    # but typically appending the target lang token is enough, or
                    # simply relying on the tokenizer if it supports it directly.
                    # Since IndicProcessor isn't found, we just pass the batch directly,
                    # BUT we must set the tokenizer's src_lang and pass forced_bos.
                    tokenizer.src_lang = src_lang
                    batch_processed = batch

                if processor is None:
                    # Instruct the tokenizer to treat the inputs purely as text and force the tags
                    tokenizer.src_lang = src_lang
                    
                    # For IndicTrans2 without IndicProcessor, the tokenizer requires the first token 
                    # of the source sentence to be the TARGET language tag!
                    # E.g., "hin_Deva in the house..."
                    batch_processed = [
                        t if t.startswith(tgt_lang) else f"{tgt_lang} {t}"
                        for t in batch_processed
                    ]

                inputs = tokenizer(
                    batch_processed,
                    padding="longest",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                
                # If processor is missing, we must ensure tgt_lang is forced
                gen_kwargs = {"max_length": 256, "num_beams": 5}
                if processor is None:
                    gen_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(tgt_lang)

                outputs = model.generate(**inputs, **gen_kwargs)
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                if processor is not None:
                    decoded = processor.postprocess_batch(decoded, lang=tgt_lang)
                results.extend(decoded)
            except Exception as e:
                logger.error("translation_batch_failed", batch_idx=i, error=str(e))
                results.extend(batch)
        return results

    def translate_segments(
        self, segments: list[dict], src_lang: str = SRC_LANG, tgt_lang: str = TGT_LANG
    ) -> list[dict]:
        """
        Translate transcription segments to Hindi.

        Returns:
            Segments with added 'hindi_text' key.
        """
        texts = [s["text"] for s in segments]
        hindi_texts = self.translate_batch(texts, src_lang, tgt_lang)
        for seg, hindi in zip(segments, hindi_texts):
            seg["hindi_text"] = hindi
        return segments

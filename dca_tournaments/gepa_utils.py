"""GEPA utilities — multimodal reflection support for the tournament pipeline."""

from __future__ import annotations

import logging
import re
import threading

import dspy

logger = logging.getLogger("tournament.gepa_utils")

_DATA_IMAGE_URL_RE = re.compile(r"(data:image/[^'\"}\s]+)")


# ═══════════════════════════════════════════════════════════════════
# Thread-local image store
# ═══════════════════════════════════════════════════════════════════

class _GepaImageStore(threading.local):
    """Thread-local storage for passing extracted image URLs into reflection calls."""

    def __init__(self):
        super().__init__()
        self.urls: list[str] = []

    def store(self, urls: list[str]) -> None:
        clean_urls = [u for u in urls if isinstance(u, str) and u]
        if not clean_urls:
            return
        self.urls.extend(clean_urls)
        if len(self.urls) > 100:
            self.urls = self.urls[-100:]

    def clear(self) -> None:
        self.urls = []

    def get_recent_images(self, max_images: int = 6) -> list[str]:
        recent = list(self.urls[-max_images:])
        seen = set()
        deduped_reversed = []
        for url in reversed(recent):
            if url not in seen:
                seen.add(url)
                deduped_reversed.append(url)
        return list(reversed(deduped_reversed))


# Global singleton — each thread gets its own copy of the data
gepa_image_store = _GepaImageStore()


# ═══════════════════════════════════════════════════════════════════
# Multimodal Reflection LM wrapper
# ═══════════════════════════════════════════════════════════════════

class MultimodalReflectionLM(dspy.BaseLM):
    """BaseLM-compatible wrapper that injects image URLs into reflection prompts.

    DSPy teleprompters validate that active LMs are instances of ``dspy.BaseLM``.
    Wrapping a plain callable causes runtime failures in trajectory capture.
    """

    def __init__(self, base_lm: dspy.BaseLM, detail: str = "low", max_images: int = 6):
        if not isinstance(base_lm, dspy.BaseLM):
            raise TypeError(f"base_lm must be dspy.BaseLM, got {type(base_lm)!r}")

        self.base_lm = base_lm
        self.detail = detail
        self.max_images = max_images

        base_kwargs = dict(getattr(base_lm, "kwargs", {}) or {})
        temperature = base_kwargs.pop("temperature", 0.0)
        max_tokens = base_kwargs.pop("max_tokens", 1000)

        super().__init__(
            model=getattr(base_lm, "model", "multimodal-reflection-wrapper"),
            model_type=getattr(base_lm, "model_type", "chat"),
            temperature=temperature,
            max_tokens=max_tokens,
            cache=getattr(base_lm, "cache", True),
            **base_kwargs,
        )

    def forward(self, prompt=None, messages=None, **kwargs):
        image_urls = gepa_image_store.get_recent_images(self.max_images)
        if not image_urls:
            return self._forward_base(prompt=prompt, messages=messages, **kwargs)

        if messages is not None:
            return self._forward_base(
                messages=self._inject_images(messages, image_urls),
                **kwargs,
            )

        prompt_text = "" if prompt is None else str(prompt)
        content = [{"type": "text", "text": prompt_text}] + [
            {"type": "image_url", "image_url": {"url": url, "detail": self.detail}}
            for url in image_urls[:self.max_images]
        ]
        return self._forward_base(messages=[{"role": "user", "content": content}], **kwargs)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        image_urls = gepa_image_store.get_recent_images(self.max_images)
        if not image_urls:
            return await self._aforward_base(prompt=prompt, messages=messages, **kwargs)

        if messages is not None:
            return await self._aforward_base(
                messages=self._inject_images(messages, image_urls),
                **kwargs,
            )

        prompt_text = "" if prompt is None else str(prompt)
        content = [{"type": "text", "text": prompt_text}] + [
            {"type": "image_url", "image_url": {"url": url, "detail": self.detail}}
            for url in image_urls[:self.max_images]
        ]
        return await self._aforward_base(messages=[{"role": "user", "content": content}], **kwargs)

    def copy(self, **kwargs):
        copied_base = self.base_lm.copy(**kwargs)
        copied = MultimodalReflectionLM(
            base_lm=copied_base,
            detail=kwargs.get("detail", self.detail),
            max_images=kwargs.get("max_images", self.max_images),
        )
        for key, value in kwargs.items():
            if hasattr(copied, key):
                setattr(copied, key, value)
            elif value is None:
                copied.kwargs.pop(key, None)
            else:
                copied.kwargs[key] = value
        return copied

    def _forward_base(self, prompt=None, messages=None, **kwargs):
        if messages is not None:
            return self.base_lm.forward(messages=messages, **kwargs)
        return self.base_lm.forward(prompt=prompt, **kwargs)

    async def _aforward_base(self, prompt=None, messages=None, **kwargs):
        if hasattr(self.base_lm, "aforward"):
            if messages is not None:
                return await self.base_lm.aforward(messages=messages, **kwargs)
            return await self.base_lm.aforward(prompt=prompt, **kwargs)
        # Fallback: if async path is unavailable, use sync forward.
        return self._forward_base(prompt=prompt, messages=messages, **kwargs)

    def _inject_images(self, messages: list[dict], image_urls: list[str]) -> list[dict]:
        image_blocks = [
            {"type": "image_url", "image_url": {"url": url, "detail": self.detail}}
            for url in image_urls[:self.max_images]
        ]
        if not image_blocks:
            return messages

        new_messages = []
        images_added = False
        for msg in messages:
            if msg.get("role") == "user" and not images_added:
                content = msg.get("content", "")
                if isinstance(content, str):
                    new_content = [{"type": "text", "text": content}] + image_blocks
                elif isinstance(content, list):
                    new_content = list(content) + image_blocks
                else:
                    new_content = [{"type": "text", "text": str(content)}] + image_blocks
                new_messages.append({**msg, "content": new_content})
                images_added = True
            else:
                new_messages.append(msg)
        return new_messages


# ═══════════════════════════════════════════════════════════════════
# Metric wrapper that stores images
# ═══════════════════════════════════════════════════════════════════

def _extract_image_urls(value) -> list[str]:
    urls: list[str] = []
    if isinstance(value, str):
        urls.extend(_DATA_IMAGE_URL_RE.findall(value))
        if value.startswith(("http://", "https://")):
            urls.append(value)
    return urls


def make_image_storing_metric(base_metric):
    """Wrap a feedback metric so recent image URLs remain available for reflection."""

    def _wrapped_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        image_urls = []
        image_urls.extend(_extract_image_urls(getattr(example, "image_original", None)))
        image_urls.extend(_extract_image_urls(getattr(example, "image_reimage", None)))
        gepa_image_store.store(image_urls)
        return base_metric(
            example,
            prediction,
            trace=trace,
            pred_name=pred_name,
            pred_trace=pred_trace,
        )

    return _wrapped_metric


# ═══════════════════════════════════════════════════════════════════
# GEPA reflective dataset patch (future use)
# ═══════════════════════════════════════════════════════════════════

_PATCH_APPLIED = False


def _extract_and_replace_image_urls(data, extracted_urls: list[str]):
    if isinstance(data, dict):
        for key, value in list(data.items()):
            data[key] = _extract_and_replace_image_urls(value, extracted_urls)
        return data
    if isinstance(data, list):
        for idx, item in enumerate(list(data)):
            data[idx] = _extract_and_replace_image_urls(item, extracted_urls)
        return data
    if isinstance(data, str):
        matches = _DATA_IMAGE_URL_RE.findall(data)
        if matches:
            extracted_urls.extend(matches)
            return f"[Image content omitted: {len(matches)} image(s)]"
    return data


def patch_gepa_reflective_dataset(send_images: bool = False):
    """Monkey-patch DSPy's GEPA adapter to scrub inline image URLs and store them."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    try:
        from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
    except ImportError:
        logger.warning("Cannot import DspyAdapter — GEPA multimodal patch skipped")
        return

    original_method = getattr(DspyAdapter, "make_reflective_dataset", None)
    if original_method is None:
        logger.warning("DspyAdapter.make_reflective_dataset not found — patch skipped")
        return

    def _patched_make_reflective_dataset(self, *args, **kwargs):
        result = original_method(self, *args, **kwargs)
        try:
            extracted_urls: list[str] = []
            _extract_and_replace_image_urls(result, extracted_urls)
            if extracted_urls:
                gepa_image_store.store(extracted_urls)
                logger.info(
                    "GEPA reflective dataset: extracted %d image URL(s)%s",
                    len(extracted_urls),
                    " for multimodal reflection" if send_images else " while using placeholders",
                )
        except Exception as exc:
            logger.warning("Image extraction from reflective dataset failed: %s", exc)

        return result

    DspyAdapter.make_reflective_dataset = _patched_make_reflective_dataset
    _PATCH_APPLIED = True
    logger.info("GEPA multimodal reflection patch applied")

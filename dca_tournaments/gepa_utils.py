"""GEPA utilities — multimodal reflection support for the tournament pipeline."""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

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

@dataclass
class MultimodalReflectionLM:
    """Wrap a base LM to inject image URLs into a GEPA reflection prompt."""

    base_lm: object
    detail: str = "low"
    max_images: int = 6

    def __call__(self, prompt=None, messages=None, **kwargs):
        image_urls = gepa_image_store.get_recent_images(self.max_images)
        if not image_urls:
            return self._call_base_lm(prompt=prompt, messages=messages, **kwargs)

        if messages is not None:
            return self._call_base_lm(
                messages=self._inject_images(messages, image_urls),
                **kwargs,
            )

        prompt_text = "" if prompt is None else str(prompt)
        content = [{"type": "text", "text": prompt_text}] + [
            {"type": "image_url", "image_url": {"url": url, "detail": self.detail}}
            for url in image_urls[:self.max_images]
        ]
        return self._call_base_lm(messages=[{"role": "user", "content": content}], **kwargs)

    def __getattr__(self, name):
        return getattr(self.base_lm, name)

    def _call_base_lm(self, prompt=None, messages=None, **kwargs):
        if messages is not None:
            return self.base_lm(messages=messages, **kwargs)
        if prompt is not None:
            try:
                return self.base_lm(prompt, **kwargs)
            except TypeError:
                return self.base_lm(prompt=prompt, **kwargs)
        return self.base_lm(**kwargs)

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

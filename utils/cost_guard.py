"""[wrapper-only] Live token/cost tracking + a HARD cost cap for LM runs (GEPA / eval). No run.py analog.

Registers a litellm success callback that fires after EVERY real LLM call — the student (gpt-4.1-mini),
the M4 coverage judge (gpt-5.2) and the GEPA reflection LM (gpt-5.2) all go through litellm, so one hook
catches them all. It accumulates tokens per model, prices them with a LOCAL table (litellm's cost map may
not know gpt-4.1-mini / gpt-5.2), logs a running total, and FORCE-STOPS the process (os._exit) once the
cumulative cost exceeds the cap. Cache hits don't reach litellm (dspy/`_judge_result_cache` short-circuit
first), so they aren't counted — cost reflects only real API spend.
"""
import os
import sys
import threading
import logging

logger = logging.getLogger("mbd.cost")  # [wrapper-only] cost-tracking logger (surfaces in the tee'd run log)

# USD per 1M tokens (input, output). Checked MOST-SPECIFIC-FIRST. Estimates for Azure OpenAI as of 2026-07;
# override via set_pricing() if your contract differs. gpt-5.2 rate is an estimate — confirm on your portal.
_PRICING = [
    ("gpt-4.1-nano", (0.10, 0.40)),
    ("gpt-4.1-mini", (0.40, 1.60)),
    ("gpt-4.1",      (2.00, 8.00)),
    ("gpt-5-mini",   (0.25, 2.00)),
    ("gpt-5-nano",   (0.05, 0.40)),
    ("gpt-5.2",      (1.25, 10.00)),
    ("gpt-5",        (1.25, 10.00)),
    ("gpt-4o-mini",  (0.15, 0.60)),
    ("gpt-4o",       (2.50, 10.00)),
]
_DEFAULT_PRICE = (2.00, 8.00)  # [wrapper-only] unknown model -> conservative gpt-4.1-class (over-estimate => caps earlier, safe)


def set_pricing(model_substr: str, price_in_per_m: float, price_out_per_m: float):  # [wrapper-only] optional override
    _PRICING.insert(0, (model_substr.lower(), (price_in_per_m, price_out_per_m)))


def _price_for(model: str):  # [wrapper-only] (in,out) USD/1M by first substring match
    m = (model or "").lower()
    for key, price in _PRICING:
        if key in m:
            return price
    return _DEFAULT_PRICE


class _CostState:  # [wrapper-only] thread-safe running totals
    def __init__(self, cap_usd):
        self.cap = cap_usd
        self.lock = threading.Lock()
        self.cost = 0.0
        self.calls = 0
        self.in_tok = 0
        self.out_tok = 0
        self.per_model = {}          # model -> [in_tok, out_tok, cost]
        self.next_report = 0.5       # log a running line every $0.50 crossed
        self.stopped = False


_state = None  # [wrapper-only] module-global state (one guard per process)


def _extract_usage(response_obj, kwargs):  # [wrapper-only] (model, prompt_tokens, completion_tokens) from a litellm response
    model = ""
    try:
        model = getattr(response_obj, "model", None) or (kwargs.get("model") if kwargs else "") or ""
    except Exception:
        pass
    pt = ct = 0
    try:
        u = getattr(response_obj, "usage", None)
        if u is None and isinstance(response_obj, dict):
            u = response_obj.get("usage")
        if u is not None:
            _get = (lambda k: getattr(u, k, None)) if not isinstance(u, dict) else (lambda k: u.get(k))
            pt = int(_get("prompt_tokens") or 0)
            ct = int(_get("completion_tokens") or 0)
    except Exception:
        pass
    return model, pt, ct


def _record(model, pt, ct):  # [wrapper-only] add one call's tokens, log, and force-stop if over cap
    if _state is None:
        return
    pin, pout = _price_for(model)
    call_cost = pt / 1e6 * pin + ct / 1e6 * pout
    with _state.lock:
        _state.calls += 1
        _state.in_tok += pt
        _state.out_tok += ct
        _state.cost += call_cost
        pm = _state.per_model.setdefault(model or "?", [0, 0, 0.0])
        pm[0] += pt; pm[1] += ct; pm[2] += call_cost
        cost, calls, cap = _state.cost, _state.calls, _state.cap
        report = cost >= _state.next_report
        if report:
            _state.next_report = (int(cost / 0.5) + 1) * 0.5
        over = (cap is not None and cost >= cap and not _state.stopped)
        if over:
            _state.stopped = True
    if report:
        _emit(f"[cost] ${cost:.2f}"
              + (f" / ${cap:.2f} cap" if cap else " (no cap)")
              + f"  |  {calls} LM calls, {_state.in_tok:,} in / {_state.out_tok:,} out tokens")
    if over:
        _emit(f"[cost] *** CAP EXCEEDED: ${cost:.2f} >= ${cap:.2f} after {calls} LM calls — FORCE-STOPPING (os._exit 2) ***")
        _emit("[cost] per-model spend: " + " | ".join(
            f"{k}: ${v[2]:.2f} ({v[0]:,} in / {v[1]:,} out)" for k, v in _state.per_model.items()))
        try:
            for h in logging.getLogger().handlers:
                h.flush()
        except Exception:
            pass
        os._exit(2)  # [wrapper-only] HARD stop from whatever thread the callback runs on (GEPA's gepa_state.bin preserves progress)


def _emit(msg):  # [wrapper-only] log + print to the real stderr (survives stdout tee / buffering)
    try:
        logger.warning(msg)
    except Exception:
        pass
    try:
        print(msg, file=sys.__stderr__, flush=True)
    except Exception:
        pass


def install_cost_guard(cap_usd):  # [wrapper-only] register the litellm callback; call ONCE, early (after the LM is built)
    """Track token cost across all models and force-stop when it exceeds cap_usd (None = track only, no cap)."""
    global _state
    if _state is not None:  # [wrapper-only] idempotent — don't double-register
        _state.cap = cap_usd
        return
    _state = _CostState(cap_usd)
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class _Tracker(CustomLogger):  # [wrapper-only] fires after every successful litellm completion (sync + async paths)
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            try:
                if kwargs and (kwargs.get("cache_hit") or (kwargs.get("litellm_params") or {}).get("cache_hit")):
                    return  # [wrapper-only] cache hit = $0, don't count (usually never fires, but be safe)
                _record(*_extract_usage(response_obj, kwargs))
            except Exception:
                pass

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            try:
                if kwargs and (kwargs.get("cache_hit") or (kwargs.get("litellm_params") or {}).get("cache_hit")):
                    return
                _record(*_extract_usage(response_obj, kwargs))
            except Exception:
                pass

    litellm.callbacks = list(getattr(litellm, "callbacks", []) or []) + [_Tracker()]
    _emit(f"[cost] guard installed — hard cap ${cap_usd:.2f} (force-stop on exceed)" if cap_usd
          else "[cost] tracking installed — no cap")


def current_cost():  # [wrapper-only] running total (for summaries)
    return _state.cost if _state else 0.0


def cost_summary():  # [wrapper-only] dict for the run-summary JSON
    if _state is None:
        return {}
    with _state.lock:
        return {
            "total_usd": round(_state.cost, 4),
            "lm_calls": _state.calls,
            "input_tokens": _state.in_tok,
            "output_tokens": _state.out_tok,
            "cap_usd": _state.cap,
            "per_model": {k: {"input_tokens": v[0], "output_tokens": v[1], "usd": round(v[2], 4)}
                          for k, v in _state.per_model.items()},
        }

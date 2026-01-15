import re, os
from typing import List, Tuple, Optional, Union
import logging

import torch
from comfy import model_management

# ---------------------------
# prompt attention parser
# ---------------------------

logger = logging.getLogger(__name__)

re_attention = re.compile(r"""
\\\(|\\\)|\\\[|\\]|\\\\|\\|
\(|\[|:([+-]?[.\d]+)\)|\)|]|
[^\\()\[\]:]+|:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

def parse_prompt_attention(text: str):
    """
    Returns list of [segment_text, weight].
    (abc) -> *1.1
    [abc] -> /1.1
    (abc:3.0) -> *3.0
    """
    res = []
    round_brackets = []
    square_brackets = []
    round_mult = 1.1
    square_mult = 1 / 1.1

    def multiply_range(start_pos, mult):
        for p in range(start_pos, len(res)):
            res[p][1] *= mult

    for m in re_attention.finditer(text or ""):
        tok = m.group(0)
        w = m.group(1)

        if tok.startswith("\\"):
            res.append([tok[1:], 1.0])
        elif tok == "(":
            round_brackets.append(len(res))
        elif tok == "[":
            square_brackets.append(len(res))
        elif w is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(w))
        elif tok == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_mult)
        elif tok == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_mult)
        else:
            parts = re.split(re_break, tok)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1.0])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_mult)
    for pos in square_brackets:
        multiply_range(pos, square_mult)

    if not res:
        res = [["", 1.0]]

    # merge adjacent with same weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res

# ---------------------------
# scheduled prompt parser
# ---------------------------

try:
    import lark

    schedule_parser = lark.Lark(r"""
    !start: (prompt | /[][():]/+)*

    prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*

    !emphasized: "(" prompt ")"
            | "(" prompt ":" prompt ")"
            | "[" prompt "]"

    scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER "]"
    alternate: "[" prompt ("|" prompt)+ "]"

    WHITESPACE: /\s+/
    plain: /([^\\\[\]():|]|\\.)+/

    %import common.SIGNED_NUMBER -> NUMBER
    """)

    def get_learned_conditioning_prompt_schedules(prompts, steps):
        def collect_steps(steps_, tree):
            l = [steps_]
            class CollectSteps(lark.Visitor):
                def scheduled(self, tree_):
                    tree_.children[-1] = float(tree_.children[-1])
                    if tree_.children[-1] < 1:
                        tree_.children[-1] *= steps_
                    tree_.children[-1] = min(steps_, int(tree_.children[-1]))
                    l.append(tree_.children[-1])

                def alternate(self, tree_):
                    l.extend(range(1, steps_ + 1))

            CollectSteps().visit(tree)
            return sorted(set(l))

        def at_step(step, tree):
            class AtStep(lark.Transformer):
                def scheduled(self, args):
                    before, after, _, when = args
                    yield before or () if step <= when else after

                def alternate(self, args):
                    yield next(args[(step - 1) % len(args)])

                def start(self, args):
                    def flatten(x):
                        if isinstance(x, str):
                            yield x
                        else:
                            for gen in x:
                                yield from flatten(gen)
                    return "".join(flatten(args))

                def plain(self, args):
                    yield args[0].value

                def __default__(self, data, children, meta):
                    for child in children:
                        yield child
            return AtStep().transform(tree)

        def get_schedule(prompt):
            try:
                tree = schedule_parser.parse(prompt)
            except lark.exceptions.LarkError:
                return [[steps, prompt]]
            return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

        promptdict = {p: get_schedule(p) for p in set(prompts)}
        return [promptdict[p] for p in prompts]

except Exception:
    lark = None
    schedule_parser = None

    def get_learned_conditioning_prompt_schedules(prompts, steps):
        return [[[steps, p]] for p in prompts]

# ---------------------------
# Z-Image helpers
# ---------------------------

def _encode_ids(tokenizer, text: str) -> List[int]:
    if callable(tokenizer):
        enc = tokenizer(text, add_special_tokens=False, truncation=False)
        ids = getattr(enc, "input_ids", None)
        if ids is None and isinstance(enc, dict):
            ids = enc.get("input_ids", None)
        if ids is None:
            raise RuntimeError("Tokenizer callable but no input_ids")
    elif hasattr(tokenizer, "encode"):
        try:
            enc = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            enc = tokenizer.encode(text)
        ids = getattr(enc, "ids", enc)
    else:
        raise RuntimeError(f"Tokenizer is not callable and has no encode(): {type(tokenizer)}")

    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    return list(map(int, ids))

def _find_sublist(haystack: List[int], needle: List[int], start: int = 0) -> int:
    if not needle:
        return -1
    n = len(needle)
    end = len(haystack) - n + 1
    for i in range(start, end):
        if haystack[i:i+n] == needle:
            return i
    return -1

def _decode_piece(tokenizer, tid: int) -> str:
    try:
        return tokenizer.decode([tid], clean_up_tokenization_spaces=False, skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([tid])

def _strip_attention_syntax(prompt: str):
    segs = parse_prompt_attention(prompt or "")
    clean = "".join((" " if t == "BREAK" else t) for t, _ in segs)
    segs2 = []
    for t, w in segs:
        if t == "BREAK":
            segs2.append((" ", 1.0))
        else:
            segs2.append((t, float(w)))
    return clean, segs2

def _token_weights_from_segments(tokenizer, prompt: str) -> Tuple[List[int], List[float], str]:
    clean_text, segs = _strip_attention_syntax(prompt)
    if not clean_text:
        return [], [], clean_text

    ids = _encode_ids(tokenizer, clean_text)
    pieces = [_decode_piece(tokenizer, tid) for tid in ids]
    recon = "".join(pieces)

    if recon != clean_text:
        token_ids: List[int] = []
        weights: List[float] = []
        for t, w in segs:
            if not t:
                continue
            tid = _encode_ids(tokenizer, t)
            token_ids.extend(tid)
            weights.extend([float(w)] * len(tid))
        return token_ids, weights, clean_text
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]

    pieces = [_decode_piece(tokenizer, tid) for tid in ids]
    recon = "".join(pieces)

    if recon != clean_text:
        token_ids: List[int] = []
        weights: List[float] = []
        for t, w in segs:
            if not t:
                continue
            e = tokenizer(t, add_special_tokens=False, truncation=False)
            tid = e.input_ids
            if isinstance(tid, torch.Tensor):
                tid = tid.tolist()
            if isinstance(tid, list) and len(tid) == 1 and isinstance(tid[0], list):
                tid = tid[0]
            token_ids.extend(tid)
            weights.extend([float(w)] * len(tid))
        return token_ids, weights, clean_text

    spans: List[Tuple[int, int, float]] = []
    pos = 0
    for t, w in segs:
        ln = len(t)
        if ln > 0:
            spans.append((pos, pos + ln, float(w)))
        pos += ln

    weights: List[float] = []
    tpos = 0
    span_i = 0
    for piece in pieces:
        ts, te = tpos, tpos + len(piece)
        tpos = te

        while span_i < len(spans) and spans[span_i][1] <= ts:
            span_i += 1

        wsum = 0.0
        osum = 0
        j = span_i
        while j < len(spans) and spans[j][0] < te:
            s0, s1, w = spans[j]
            ov = max(0, min(te, s1) - max(ts, s0))
            if ov > 0:
                wsum += w * ov
                osum += ov
            j += 1

        weights.append(wsum / osum if osum > 0 else 1.0)

    return ids, weights, clean_text

def _resample_seq_to_len(x: torch.Tensor, L: int) -> torch.Tensor:
    if x.size(0) == L:
        return x
    if x.size(0) <= 1:
        return x.repeat(L, 1)
    import torch.nn.functional as F
    t = x.transpose(0, 1).unsqueeze(0)          # (1,H,S)
    t = F.interpolate(t, size=L, mode="linear", align_corners=False)
    return t.squeeze(0).transpose(0, 1)         # (L,H)

# ---------------------------
# AND parser
# ---------------------------

def _has_top_level_AND(text: str) -> bool:
    if "AND" not in (text or ""):
        return False
    pr = br = 0
    s = text
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == "(":
            pr += 1
        elif ch == ")" and pr > 0:
            pr -= 1
        elif ch == "[":
            br += 1
        elif ch == "]" and br > 0:
            br -= 1

        if pr == 0 and br == 0 and i + 3 <= n and s[i:i+3] == "AND":
            prev = s[i-1] if i > 0 else " "
            nxt  = s[i+3] if i + 3 < n else " "
            prev_ok = prev.isspace() or (not prev.isalnum() and prev != "_")
            nxt_ok  = nxt.isspace()  or (not nxt.isalnum()  and nxt != "_")
            if prev_ok and nxt_ok:
                return True
        i += 1
    return False

def _split_top_level_AND(text: str) -> List[str]:
    out, buf = [], []
    pr = br = 0
    s = text or ""
    n = len(s)
    i = 0

    def boundary(ch: str) -> bool:
        return ch.isspace() or (not ch.isalnum() and ch != "_")

    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            buf.append(s[i]); buf.append(s[i+1])
            i += 2
            continue

        if ch == "(":
            pr += 1
        elif ch == ")" and pr > 0:
            pr -= 1
        elif ch == "[":
            br += 1
        elif ch == "]" and br > 0:
            br -= 1

        if pr == 0 and br == 0 and i + 3 <= n and s[i:i+3] == "AND":
            prev = s[i-1] if i > 0 else " "
            nxt  = s[i+3] if i + 3 < n else " "
            if boundary(prev) and boundary(nxt):
                seg = "".join(buf).strip()
                if seg:
                    out.append(seg)
                buf = []
                i += 3
                continue

        buf.append(ch)
        i += 1

    seg = "".join(buf).strip()
    if seg or not out:
        out.append(seg)
    return out

def _split_suffix_weight(seg: str) -> Tuple[str, float]:
    s = (seg or "").strip()
    if not s:
        return "", 1.0
    pr = br = 0
    last_colon = -1
    for i, ch in enumerate(s):
        if ch == "\\":
            continue
        if ch == "(":
            pr += 1
        elif ch == ")" and pr > 0:
            pr -= 1
        elif ch == "[":
            br += 1
        elif ch == "]" and br > 0:
            br -= 1
        elif ch == ":" and pr == 0 and br == 0:
            last_colon = i

    if last_colon == -1:
        return s, 1.0
    left = s[:last_colon].strip()
    right = s[last_colon+1:].strip()
    try:
        w = float(right)
    except Exception:
        return s, 1.0
    if not left:
        return s, 1.0
    return left, w

# ---------------------------
# main: Z-Image encode (ComfyUI)
# ---------------------------

class _TokenizerAdapter:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, text, add_special_tokens=False, truncation=False):
        ids = _encode_ids(self.tok, text)
        return type("Enc", (), {"input_ids": ids})

    def decode(self, ids, clean_up_tokenization_spaces=False, skip_special_tokens=False):
        try:
            return self.tok.decode(ids, clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                                   skip_special_tokens=skip_special_tokens)
        except TypeError:
            try:
                return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)
            except TypeError:
                return self.tok.decode(ids)


def _load_gpu_if_possible(clip):
    if hasattr(clip, "patcher"):
        model_management.load_model_gpu(clip.patcher)
        return


def _pick_token_key(tokenized: dict) -> str:
    if not isinstance(tokenized, dict) or len(tokenized) == 0:
        raise RuntimeError("TC_ADV_ZPrompt: clip.tokenize() returned invalid tokens")

    for k in ("qwen3_4b", "l", "g"):
        if k in tokenized:
            return k
        
    if len(tokenized) == 1:
        return next(iter(tokenized.keys()))

    raise RuntimeError(f"TC_ADV_ZPrompt: unknown token keys: {list(tokenized.keys())}")


def _iter_token_items(chunk):
    for it in chunk:
        if isinstance(it, (tuple, list)):
            if len(it) == 3:
                yield int(it[0]), float(it[1]), it[2]
            elif len(it) == 2:
                yield int(it[0]), float(it[1]), None
            else:
                yield int(it[0]), 1.0, None
        else:
            yield int(it), 1.0, None


def _is_usable_tokenizer(tok) -> bool:
    if tok is None:
        return False
    if not hasattr(tok, "decode"):
        return False
    if callable(tok):
        return True
    if hasattr(tok, "encode"):
        return True
    return False

def _maybe_wrap(tok):
    if tok is None:
        return None
    if _is_usable_tokenizer(tok) and callable(tok):
        return tok
    if _is_usable_tokenizer(tok) and (not callable(tok)):
        return _TokenizerAdapter(tok)
    return None

def _unwrap_hf_tokenizer_from_zimage_tokenizer(z_tok):
    seen = set()

    def push(cands, x):
        if x is None:
            return
        xid = id(x)
        if xid in seen:
            return
        seen.add(xid)
        cands.append(x)

    q = []
    push(q, z_tok)

    for name in ("hf_tokenizer", "_hf_tokenizer", "_tokenizer", "tokenizer_hf", "inner_tokenizer", "base_tokenizer"):
        push(q, getattr(z_tok, name, None))

    i = 0
    while i < len(q) and i < 32:
        cur = q[i]; i += 1

        wrapped = _maybe_wrap(cur)
        if wrapped is not None:
            return wrapped

        push(q, getattr(cur, "tokenizer", None))
        push(q, getattr(cur, "backend_tokenizer", None))
        push(q, getattr(cur, "hf_tokenizer", None))

        inner = getattr(cur, "tokenizer", None)
        if inner is not None:
            push(q, getattr(inner, "tokenizer", None))
            inner2 = getattr(inner, "tokenizer", None)
            if inner2 is not None:
                push(q, getattr(inner2, "tokenizer", None))

        tok_path = getattr(cur, "tokenizer_path", None)
        if isinstance(tok_path, str):
            try:
                from transformers import AutoTokenizer
                hf = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
                wrapped = _maybe_wrap(hf)
                if wrapped is not None:
                    return wrapped
            except Exception:
                pass

    raise RuntimeError(
        "TC_ADV_ZPrompt: cannot find usable tokenizer (needs decode + (callable or encode)). "
        "Your object chain exists but was not callable; add adapter as above."
    )


def _encode_single_zimage(
    clip,
    text: str,
    weight_strength: float,
    clamp_min: float,
    clamp_max: float,
):
    logger.warning(f"z_tok type={type(clip.tokenizer)}")
    logger.warning(f"z_tok.tokenizer type={type(getattr(clip.tokenizer,'tokenizer',None))}")
    logger.warning(f"z_tok.tokenizer.tokenizer type={type(getattr(getattr(clip.tokenizer,'tokenizer',None),'tokenizer',None))}")
    logger.warning(f"z_tok.tokenizer.tokenizer.tokenizer type={type(getattr(getattr(getattr(clip.tokenizer,'tokenizer',None),'tokenizer',None),'tokenizer',None))}")
    
    # 1) tokenize with ComfyUI CLIP API
    clean_text, _segs = _strip_attention_syntax(text or "")
    tokenized = clip.tokenize(clean_text, return_word_ids=True)

    key = _pick_token_key(tokenized)
    tok_chunks = tokenized[key]
    if tok_chunks is None:
        raise RuntimeError(f"TC_ADV_ZPrompt: tokens missing key '{key}' (keys={list(tokenized.keys())})")

    # 2) flatten token ids
    ids_flat = []
    for ch in tok_chunks:
        for tid, _w, _wid in _iter_token_items(ch):
            ids_flat.append(tid)

    # 3) get HF tokenizer（ZImageTokenizer -> Qwen3Tokenizer -> Qwen2Tokenizer）
    z_tok = getattr(clip, "tokenizer", None)
    if z_tok is None:
        raise RuntimeError("TC_ADV_ZPrompt: clip.tokenizer not found")
    hf_tok = _unwrap_hf_tokenizer_from_zimage_tokenizer(z_tok)

    # 4) calculate content token ids and weights
    content_ids, content_w, _ = _token_weights_from_segments(hf_tok, text)

    # 5) find content token sublist in ids_flat, and build weights_flat
    weights_flat = [1.0] * len(ids_flat)

    s_idx = _find_sublist(ids_flat, content_ids, 0) if content_ids else -1
    if s_idx == -1 and content_ids:
        logger.warning(
            "TC_ADV_ZPrompt: content token sublist not found; fallback to no weighting. "
            f"(key={key}, ids_flat={len(ids_flat)}, content_ids={len(content_ids)})"
        )
    else:
        n = min(len(content_w), len(ids_flat) - s_idx)
        for i in range(n):
            w = float(content_w[i])

            # NegPiP
            if w < 0.0:
                w = 1.0 + w

            # strength
            w = 1.0 + (w - 1.0) * float(weight_strength)

            # clamp
            w = max(float(clamp_min), min(float(clamp_max), w))
            weights_flat[s_idx + i] = w

    # 6) rebuild token chunks
    idx = 0
    new_chunks = []
    for ch in tok_chunks:
        new_ch = []
        for tid, _oldw, wid in _iter_token_items(ch):
            if idx < len(weights_flat):
                new_ch.append((tid, float(weights_flat[idx]), wid))
            else:
                new_ch.append((tid, 1.0, wid))
            idx += 1
        new_chunks.append(new_ch)

    weighted_tokens = dict(tokenized)
    weighted_tokens[key] = new_chunks

    # 7) encode
    cond, pooled = clip.encode_from_tokens(weighted_tokens, return_pooled=True)
    return cond, pooled


def _encode_with_AND_safe(
    clip,
    text: str,
    weight_strength: float,
    clamp_min: float,
    clamp_max: float,
    and_strength: float,
    base_bias: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    base_h, base_p = _encode_single_zimage(
        clip, text,
        weight_strength=weight_strength,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )

    if not _has_top_level_AND(text) or and_strength <= 0.0:
        return base_h, base_p

    parts = _split_top_level_AND(text)
    parsed = []
    for p in parts:
        t, w = _split_suffix_weight(p)
        t = (t or "").strip()
        if t:
            parsed.append((t, float(w)))

    if len(parsed) <= 1:
        return base_h, base_p

    base_seq = base_h[0]  # (L,H)
    L = base_seq.size(0)

    embs = [base_seq]
    ws = [float(base_bias)]
    pools = [base_p]
    pws = [float(base_bias)]

    for t, w in parsed:
        hi, pi = _encode_single_zimage(
            clip, t,
            weight_strength=weight_strength,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
        )
        si = hi[0]
        si = _resample_seq_to_len(si, L)
        embs.append(si)
        ws.append(float(w))
        pools.append(pi)
        pws.append(float(w))

    denom = sum(abs(x) for x in ws) or 1.0
    mixed = torch.zeros_like(base_seq)
    for e, w in zip(embs, ws):
        mixed = mixed + e * float(w)
    mixed = mixed / float(denom)

    pden = sum(abs(x) for x in pws) or 1.0
    pmix = torch.zeros_like(base_p)
    for p, w in zip(pools, pws):
        pmix = pmix + p * float(w)
    pmix = pmix / float(pden)

    mod_h = base_seq + (mixed - base_seq) * float(and_strength)
    mod_p = base_p + (pmix - base_p) * float(and_strength)

    return mod_h.unsqueeze(0), mod_p

def advanced_zprompt_encode(
    clip,
    text: str,
    schedule_steps: int = 30,
    use_schedule: bool = True,
    weight_strength: float = 2.0,
    clamp_min: float = 0.0,
    clamp_max: float = 3.0,
    and_strength: float = 0.6,
    base_bias: float = 4.0,
):
    """
    Returns ComfyUI CONDITIONING list:
      [ [emb, {"pooled_output": pooled, "start_percent":..., "end_percent":...}], ...]
    """
    _load_gpu_if_possible(clip)

    text = text or ""
    steps = max(1, int(schedule_steps))

    if not use_schedule:
        h, p = _encode_with_AND_safe(
            clip, text,
            weight_strength=weight_strength,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            and_strength=and_strength,
            base_bias=base_bias,
        )
        return [[h, {"pooled_output": p}]]

    # scheduled prompts
    sched = get_learned_conditioning_prompt_schedules([text], steps)[0]  # [[end_step, prompt], ...]
    if len(sched) <= 1:
        h, p = _encode_with_AND_safe(
            clip, text,
            weight_strength=weight_strength,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            and_strength=and_strength,
            base_bias=base_bias,
        )
        return [[h, {"pooled_output": p}]]

    out = []
    prev_end = 0
    for end_step, subtext in sched:
        end_step = int(end_step)
        start_p = float(prev_end) / float(steps)
        end_p = float(end_step) / float(steps)

        h, p = _encode_with_AND_safe(
            clip, subtext,
            weight_strength=weight_strength,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            and_strength=and_strength,
            base_bias=base_bias,
        )
        out.append([h, {"pooled_output": p, "start_percent": start_p, "end_percent": end_p}])
        prev_end = end_step

    return out

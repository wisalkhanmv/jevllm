"""The decoding loop: turn Choice calls into text, with as few round trips as possible.

Three things make a naive version slow, and all three are fixable:

1. A fresh TLS handshake per step — fixed by the persistent client in api.py.
2. Re-proposing candidates every step — fixed by caching the candidate set and
   refreshing it only when Jev's confidence says it no longer fits.
3. One round trip per word — softened by speculative fan-out. Jev evaluates
   every question in a request against the state in parallel, so alongside
   "what comes next?" we also ask "and if it turns out to be X, what then?"
   for the few most likely X. When the sampled word is one we guessed, the
   following distribution is already in hand and we advance two words for one
   round trip. This is ordinary speculative decoding: the proposer drafts,
   Jev verifies.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Iterator

from . import tree as T
from . import vocab as V
from .api import JevClient, choice_question, read_choice, sample


@dataclass
class Step:
    """One emitted unit — everything the UI and CLI need to render it."""
    unit: str
    is_punct: bool
    probs: dict[str, float]
    confidence: float | None
    options: int
    index: int
    speculated: bool          # emitted without its own round trip
    requests: int
    tokens: int
    cost: float
    elapsed: float

    @property
    def top_prob(self) -> float:
        return max(self.probs.values()) if self.probs else 0.0


@dataclass
class Settings:
    prompt: str
    mode: str = "word"
    strategy: str = "common"
    limit: int = 14
    temperature: float = 0.2
    top_k: int = 0
    top_p: float = 1.0
    repeat_penalty: float = 1.6
    repeat_window: int = 6
    #: Multiplier on every punctuation candidate. Below 1 discourages it.
    #:
    #: Jev finds a full stop plausible almost anywhere, so left at 1.0 it ends
    #: up as roughly half the output. Worth knowing before turning it down: the
    #: punctuation is doing real work. It chunks the output into short
    #: fragments that are each individually plausible, which hides the ordering
    #: weakness. At 0 you get "doing fine thanks you bye see over here there",
    #: which is more honest and much worse to read. 0.6 is the compromise.
    punct_bias: float = 0.6
    model: str = "jev-latest"
    proposer: str = "claude-sonnet-5"
    proposer_key: str = ""
    proposer_n: int = 200
    seed: int | None = None
    timeout: float = 45.0
    #: Stop once the generated text ends with any of these. Empty means run to
    #: the step count, and the person stops it if they want to.
    stop_at: tuple[str, ...] = ()
    escape_whitespace: bool = True
    #: How many recently generated units to keep in the state, alongside the
    #: original prompt. 0 keeps everything.
    #:
    #: Real language models do not slide by default — they keep the whole
    #: context and lean on a KV cache, so re-reading costs nothing. We have no
    #: cache across requests, so every token of state is re-billed and re-read
    #: on every step, which is quadratic in the length of the output.
    #:
    #: The shape comes from StreamingLLM: dropping the *oldest* tokens
    #: collapses quality, because early tokens act as attention sinks. Keeping
    #: a few initial tokens plus a recent window recovers nearly all of it. The
    #: prompt is our sink — it carries the topic and register — so it is always
    #: kept whole and the middle is what gets elided.
    context_words: int = 0

    #: How many speculative branches to attach to each request. 0 disables.
    #:
    #: Off by default. It buys roughly 20% fewer round trips and costs a great
    #: deal of coherence: a word cashed in from a speculation is drawn from the
    #: trimmed `speculate_width` option list rather than the full candidate set,
    #: and the difference is stark — "I'm fine. Thanks." becomes "Good. and And
    #: But Well is am." on the same prompt and seed.
    speculate: int = 0
    #: Re-propose candidates when the top probability falls below this, i.e.
    #: when the vocabulary no longer fits the text. 0 means never re-propose.
    refresh_below: float = 0.07
    #: Hard ceiling on how many steps a cached candidate set may serve.
    refresh_every: int = 16
    #: Options offered to a speculative branch. Each question carries its own
    #: criteria, so full-width speculation multiplies input tokens.
    speculate_width: int = 80

    parts: list[str] = field(default_factory=list)


@dataclass
class Stats:
    requests: int = 0
    units: int = 0
    speculated: int = 0
    proposals: int = 0
    reconnects: int = 0

    @property
    def acceptance(self) -> float:
        return self.speculated / self.units if self.units else 0.0

    @property
    def units_per_request(self) -> float:
        return self.units / self.requests if self.requests else 0.0


def _forbid_double_punct(probs: dict, state: str, parts: list[str]) -> dict:
    """Drop punctuation candidates when the text already ends in punctuation."""
    tail = (parts[-1] if parts else state.rstrip()[-1:] if state.strip() else "")
    if tail not in V.PUNCT:
        return probs
    kept = {o: p for o, p in probs.items() if o not in V.PUNCT}
    return kept or probs


def _shape_chars(probs: dict, parts: list[str]) -> dict:
    """Character mode needs almost none of the word-level shaping.

    Blocking a repeated unit is right for words and fatal for characters —
    it makes "ll", "ss" and "oo" unspellable — and blocking repeated bigrams
    is worse: once "th" has occurred, "h" could never follow "t" again. The
    one rule that does carry over is that a space may not follow a space,
    which is what the earlier drift into whitespace looked like.
    """
    if not parts or parts[-1] not in V.ESCAPES:
        return probs
    # probs is keyed by the labels actually sent — whitespace travels as
    # "<space>" / "<newline>", so filtering on the raw character matches
    # nothing and the run drifts into a wall of spaces.
    blocked = set(V.ESCAPES) | set(V.ESCAPES.values())
    kept = {o: p for o, p in probs.items() if o not in blocked}
    return kept or probs


def _block_loops(probs: dict, parts: list[str]) -> dict:
    """Forbid an immediate repeat, and any bigram already used.

    A flat frequency penalty is the wrong instrument here: set high enough to
    break a "but but but" loop, it also bans the ordinary recurrence of "you",
    "is" and "the" that English depends on. Blocking repeated bigrams — standard
    no-repeat-ngram decoding — kills loops while leaving natural reuse alone.
    """
    if not parts:
        return probs
    banned = {parts[-1]}
    seen = {(a, b) for a, b in zip(parts, parts[1:])}
    banned |= {b for (a, b) in seen if a == parts[-1]}
    # Punctuation used to be exempt here, which handed it a structural
    # advantage: every step that banned content words left punctuation
    # untouched, so full stops accumulated across a run.
    kept = {o: p for o, p in probs.items() if o not in banned}
    if kept:
        return kept
    # Everything is banned. Ending the clause beats resuming the loop.
    punct = {o: p for o, p in probs.items() if o in V.PUNCT}
    return punct or probs


def _bias_punct(probs: dict, settings: Settings) -> dict:
    if settings.punct_bias == 1.0:
        return probs
    scaled = {o: (p * settings.punct_bias if o in V.PUNCT else p)
              for o, p in probs.items()}
    return scaled if any(v > 0 for v in scaled.values()) else probs


def _penalise(probs, parts, settings):
    """A gentle nudge away from words used very recently, on top of _block_loops."""
    if settings.repeat_penalty <= 0 or not parts:
        return probs
    recent = set(parts[-settings.repeat_window:])
    return {o: (p / settings.repeat_penalty if o in recent else p)
            for o, p in probs.items()}


def _state_for(settings: Settings, parts: list[str]) -> str:
    """The text actually sent as state — prompt (the sink) plus a recent window."""
    if settings.context_words <= 0 or len(parts) <= settings.context_words:
        return _text(settings, parts)
    kept = parts[-settings.context_words:]
    if settings.mode == "char":
        return settings.prompt + " [...] " + "".join(kept)
    return settings.prompt.rstrip() + " [...]" + V.join(kept)


def _text(settings: Settings, parts: list[str]) -> str:
    if settings.mode == "char":
        return settings.prompt + "".join(parts)
    return settings.prompt.rstrip() + V.join(parts)


def decode(client: JevClient, settings: Settings, stats: Stats | None = None) -> Iterator[Step]:
    """Yield one Step per generated unit."""
    stats = stats or Stats()
    rng = random.Random(settings.seed)
    parts: list[str] = list(settings.parts)
    started = time.time()
    _descend.started = started

    char_criteria, char_back = (
        V.char_options(settings.escape_whitespace) if settings.mode == "char" else ({}, {})
    )

    recent_cats: list[str] = []     # tree mode: word classes just used
    cached: list[str] = []          # current candidate set
    draft: list[str] = []           # proposer's ranked head, used for speculation
    age = 0                         # steps the cached set has served
    emitted = 0

    while emitted < settings.limit:
        state = _state_for(settings, parts)

        # ---- tree descent: category, then word, in one round trip ---------
        if settings.strategy == "tree" and settings.mode == "word":
            step = _descend(client, settings, state, parts, rng, stats,
                            emitted, recent_cats)
            parts.append(step.unit)
            emitted += 1
            stats.units += 1
            yield step
            if _sentence_end(step.unit, parts, settings):
                break
            continue

        # ---- candidate set ------------------------------------------------
        if settings.mode == "char":
            criteria, back = char_criteria, char_back
            instructions = V.CHAR_INSTRUCTIONS
        else:
            stale = (settings.strategy != "tree") and (not cached
                     or age >= settings.refresh_every
                     or (settings.strategy != "dynamic" and settings.strategy == "hybrid"))
            if stale:
                cached, draft = V.word_options(
                    settings.strategy, settings.prompt, V.join(parts),
                    proposer_key=settings.proposer_key,
                    proposer_model=settings.proposer,
                    proposer_n=settings.proposer_n, timeout=settings.timeout,
                )
                if settings.strategy == "dynamic":
                    stats.proposals += 1
                age = 0
            criteria, back = {w: None for w in cached}, {}
            instructions = V.WORD_INSTRUCTIONS

        # ---- one request: the real question plus speculative branches ------
        questions = {"next": choice_question(instructions, criteria)}
        guesses: list[str] = []
        if settings.speculate and settings.mode == "word":
            pool = draft or sorted(cached, key=lambda w: w in V.PUNCT)
            guesses = [g for g in pool if g in criteria][: settings.speculate]
            narrow = dict(list(criteria.items())[: settings.speculate_width])
            for g in guesses:
                narrow.setdefault(g, None)
            for i, guess in enumerate(guesses):
                questions[f"spec{i}"] = choice_question(
                    f"{instructions}\n\nAssume the very next word after the text "
                    f"below is \"{guess}\". Which word or punctuation follows "
                    f"that one?", narrow)

        response = client.ask(state, questions, model=settings.model)
        stats.requests += 1
        probs, confidence = read_choice(response, "next")

        # ---- emit the real unit -------------------------------------------
        if settings.mode == "char":
            usable = _shape_chars(_forbid_double_punct(probs, state, parts), parts)
        else:
            usable = _bias_punct(
                _block_loops(_forbid_double_punct(probs, state, parts), parts), settings)
            usable = _penalise(usable, parts, settings)
        chosen = sample(usable,
                        temperature=settings.temperature, top_k=settings.top_k,
                        top_p=settings.top_p, rng=rng)
        unit = back.get(chosen, chosen)
        parts.append(unit)
        emitted += 1
        age += 1
        stats.units += 1

        yield Step(unit=unit, is_punct=settings.mode == "word" and unit in V.PUNCT,
                   probs={back.get(k, k): v for k, v in probs.items()},
                   confidence=confidence, options=len(criteria), index=emitted - 1,
                   speculated=False, requests=client.requests,
                   tokens=client.input_tokens, cost=client.cost,
                   elapsed=time.time() - started)

        if _sentence_end(unit, parts, settings) or emitted >= settings.limit:
            break

        # ---- cash in a speculation, if we guessed right --------------------
        if chosen in guesses:
            qid = f"spec{guesses.index(chosen)}"
            spec_probs, spec_conf = read_choice(response, qid)
            spec_usable = _penalise(_bias_punct(_block_loops(
                _forbid_double_punct(spec_probs, state, parts), parts), settings),
                parts, settings)
            spec_choice = sample(spec_usable,
                                 temperature=settings.temperature, top_k=settings.top_k,
                                 top_p=settings.top_p, rng=rng)
            spec_unit = back.get(spec_choice, spec_choice)
            parts.append(spec_unit)
            emitted += 1
            age += 1
            stats.units += 1
            stats.speculated += 1

            yield Step(unit=spec_unit,
                       is_punct=settings.mode == "word" and spec_unit in V.PUNCT,
                       probs={back.get(k, k): v for k, v in spec_probs.items()},
                       confidence=spec_conf, options=len(criteria), index=emitted - 1,
                       speculated=True, requests=client.requests,
                       tokens=client.input_tokens, cost=client.cost,
                       elapsed=time.time() - started)

            if _sentence_end(spec_unit, parts, settings):
                break

        # ---- has the vocabulary stopped fitting? ---------------------------
        if (settings.refresh_below > 0 and settings.strategy == "dynamic"
                and max(probs.values(), default=1.0) < settings.refresh_below):
            cached, draft, age = [], [], 0

    stats.reconnects = client.reconnects
    settings.parts = parts


def _descend(client, settings, state, parts, rng, stats, emitted, recent) -> Step:
    """One request: which category, plus which word inside every category.

    Jev evaluates all of them against the state in parallel, so only the
    branch the category answer picks is ever read — the rest cost tokens but
    no extra latency, and no extra round trip.
    """
    cats = T.categories()
    questions = {"cat": choice_question(T.CATEGORY_INSTRUCTIONS, cats)}
    for name in cats:
        questions[f"w_{name}"] = choice_question(
            T.WORD_INSTRUCTIONS.format(desc=T.describe(name)), T.words(name))

    response = client.ask(state, questions, model=settings.model)
    stats.requests += 1

    cat_probs, _ = read_choice(response, "cat")

    tail = parts[-1] if parts else state.rstrip()[-1:]
    viable = {}
    for c, p in cat_probs.items():
        if c == "punctuation":
            # Every word in this category is punctuation, so once the text
            # ends in punctuation the whole branch is dead — without this the
            # word level falls back and emits one anyway, which is what the
            # ".,.?.!" cycling was.
            if tail in V.PUNCT:
                continue
            p *= settings.punct_bias      # bias applies to the category here
        elif not _block_loops({w: 1.0 for w in T.words(c)}, parts):
            continue                       # nothing left in this branch
        # Real prose alternates word classes. Without this the same branch
        # keeps winning and you get "city street city town city river city".
        if recent and c == recent[-1]:
            p /= max(settings.repeat_penalty, 1.0) * 2.5
        elif c in recent[-3:]:
            p /= max(settings.repeat_penalty, 1.0)
        viable[c] = p

    category = sample(viable or cat_probs, temperature=settings.temperature, rng=rng)

    recent.append(category)
    del recent[:-6]

    word_probs, confidence = read_choice(response, f"w_{category}")
    usable = _block_loops(_forbid_double_punct(word_probs, state, parts), parts)
    chosen = sample(_penalise(usable, parts, settings),
                    temperature=settings.temperature, top_k=settings.top_k,
                    top_p=settings.top_p, rng=rng)

    return Step(
        unit=chosen, is_punct=chosen in V.PUNCT,
        # Show the category race in the UI: it is the more interesting judgment.
        probs={f"{c}": p for c, p in sorted(
            cat_probs.items(), key=lambda kv: kv[1], reverse=True)},
        confidence=confidence, options=sum(len(T.words(c)) for c in cats),
        index=emitted, speculated=False, requests=client.requests,
        tokens=client.input_tokens, cost=client.cost,
        elapsed=time.time() - _descend.started,
    )


def _sentence_end(unit: str, parts: list[str], settings: Settings) -> bool:
    """True once the text ends with something the caller asked to stop at."""
    if not settings.stop_at or len(parts) < 2:
        return False
    tail = _text(settings, parts)
    return any(tail.rstrip().endswith(t) for t in settings.stop_at if t)


def text_of(settings: Settings) -> str:
    return _text(settings, settings.parts)

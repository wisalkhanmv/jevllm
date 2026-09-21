"""Candidate sets: what Jev is allowed to choose from at each step.

Jev picks one option from a closed set. That set *is* the model's vocabulary,
so how you assemble it determines almost everything about the output.
"""

from __future__ import annotations

import json
import re
import string
import urllib.error
import urllib.request

from .api import MAX_OPTIONS

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

#: A standalone apostrophe is never a useful choice at word level, and semicolons
#: and colons almost never are. Fewer punctuation options also means fewer
#: chances for punctuation to win a step it should not.
PUNCT = [".", ",", "?", "!"]

CHAR_INSTRUCTIONS = (
    "The text below is an excerpt from a longer document, cut off mid-flow. "
    "Which single character comes next, immediately after the very last "
    "character shown? Consider spelling, grammar, and the style and subject of "
    "the text so far."
)

WORD_INSTRUCTIONS = (
    "The text below is an excerpt from a longer document, cut off mid-sentence. "
    "Which word or punctuation mark comes next, immediately after the last word "
    "shown? Consider grammar, meaning, and the subject of the text so far. "
    "Choose the option that most plausibly continues it."
)

PROPOSER_PROMPT = (
    "Here is the beginning of a passage:\n\n{text}\n\n"
    "List {n} plausible candidates for the next word (or punctuation mark), and "
    "for the words likely to follow it over the next sentence or so. Include "
    "grammatical function words and specific content words. "
    "Order the array by how likely each is to be the VERY NEXT word, most "
    "likely first. Reply with ONLY a JSON array of strings, no prose, no markdown."
)

#: Printable ASCII minus the characters that make poor JSON map keys.
ALPHABET = (
    string.ascii_lowercase + string.ascii_uppercase + string.digits + " .,!?'\"-:;()\n"
)

#: Whitespace travels as a visible label and is mapped back locally — a bare
#: " " as a JSON key is at the mercy of whitespace normalisation in transit.
ESCAPES = {" ": "<space>", "\n": "<newline>", "\t": "<tab>"}

# A Choice accepts 255 options, so the corpus can never ship whole. It is
# grouped by word class and sliced per step with a fixed budget for each, which
# matters more than size: an earlier version simply truncated a flat list and
# spent every content slot on pronouns and auxiliaries, producing "am are doing?
# being? are?" where the words it needed — good, thanks, fine — sat unused past
# the cutoff.

FUNCTION_CORE = """
the a an and or but so because if then that this these those which who what
is are was were be been being am has have had do does did will would can could should
may might must of in on at to from by with for about into over under after before between
i you he she it we they me him her us them my your his its our their
not no very too also just only even still yet again more most much many
one two three first next last other same
there here when where how why while all some any each both
""".split()

#: Openers, closers and discourse markers — what conversational replies are
#: actually made of, and what a frequency-ordered list buries.
#: Ordered by usefulness, not alphabetically or by raw frequency — each class is
#: truncated to its budget, so whatever sits past the line never gets offered.
REPLIES = """
good fine great well thanks yes no okay sure nice sorry please hi hello hey
really very pretty doing glad happy alright much too also lovely wonderful
yeah bye thank maybe perhaps course actually honestly certainly probably exactly
""".split()

VERBS = """
go goes going gone went come comes coming came take takes took make makes made
say says said tell tells told know knows knew think thinks thought see sees saw
look looks looked want wants wanted give gives gave find finds found use uses used
work works worked call calls called try tries tried ask asks asked need needs
feel feels felt become leave leaves left put puts mean means meant keep keeps kept
begin began seem seems help helps talk talks turn turns start starts started
show shows showed hear heard play plays run runs move moves like likes liked
live lives lived believe hold holds bring brings happen happens write writes wrote
sit sits stand stands lose lost pay pays meet meets met include continue set sets
learn learns change changes lead leads understand watch follow stop stops create
speak speaks read reads allow add adds spend grow open opens walk walks win wins
offer remember love loves consider appear buy bought wait waits send sends expect
build builds stay stays fall falls cut reach remain suggest raise pass sell
require report decide pull return explain hope hopes develop carry break receive
agree support hit produce eat eats drink sleep wake drive drove sing sang
""".split()

NOUNS = """
person people man woman child children friend family father mother brother sister
parent home house room door window city town country world place way thing things
part side end water food money work job business company team group school book
word name life story hand head eye face heart mind body car street road number
problem question answer idea reason fact case point line letter phone email message
note game music film light air news weather morning afternoon evening night day
week month year hour minute time moment coffee tea bread pizza fruit summer winter
""".split()

ADJECTIVES = """
bad better best worse worst new old young high low long short big small large little
happy sad glad sure certain clear easy hard difficult simple strong weak early late
right wrong true false real whole full empty free busy tired ready warm cold hot cool
bright dark heavy light quick slow quiet loud safe close open deep beautiful terrible
awful amazing interesting boring important serious
""".split()

ADVERBS = """
now then today tomorrow yesterday tonight here there always never often sometimes
usually rarely again soon later already almost nearly hardly quite rather enough
together anyway instead however therefore though although since until while
""".split()

#: How many slots each class gets once punctuation, the function core and any
#: words already present in the text have been placed.
BUDGET = [(REPLIES, 30), (VERBS, 40), (NOUNS, 32), (ADJECTIVES, 22), (ADVERBS, 14)]

INSTRUCTIONS = (
    "The text below is an excerpt from a longer document, cut off mid-flow. "
    "Which single character comes next, immediately after the very last "
    "character shown? Consider spelling, grammar, and the style and subject of "
    "the text so far."
)

WORD_INSTRUCTIONS = (
    "The text below is an excerpt from a longer document, cut off mid-sentence. "
    "Which word or punctuation mark comes next, immediately after the last word "
    "shown? Consider grammar, meaning, and the subject of the text so far. "
    "Choose the option that most plausibly continues it."
)

PROPOSER_PROMPT = (
    "Here is the beginning of a passage:\n\n{text}\n\n"
    "List {n} plausible candidates for the next word (or punctuation mark), and "
    "for the words likely to follow it over the next sentence or so. Include "
    "grammatical function words and specific content words. "
    "Order the array by how likely each is to be the VERY NEXT word, most "
    "likely first. Reply with ONLY a JSON array of strings, no prose, no markdown."
)

#: Printable ASCII minus the characters that make poor JSON map keys.
ALPHABET = (
    string.ascii_lowercase + string.ascii_uppercase + string.digits + " .,!?'\"-:;()\n"
)

#: Whitespace travels as a visible label and is mapped back locally — a bare
#: " " as a JSON key is at the mercy of whitespace normalisation in transit.
ESCAPES = {" ": "<space>", "\n": "<newline>", "\t": "<tab>"}

#: ~900 high-frequency English words, grouped so the selector can always keep
#: the grammatical skeleton and spend its remaining budget on content. A Choice
#: accepts 255 options, so the whole corpus never ships at once — see
#: `word_options`, which assembles a per-step slice.
CORPUS = """
i you he she it we they me him her us them my your his its our their mine yours theirs
myself yourself himself herself itself ourselves themselves one ones someone anyone everyone
nobody something anything everything nothing somebody anybody everybody each either neither
the a an this that these those some any all both few many much more most other another
is are was were be been being am do does did done doing have has had having will would
can could shall should may might must ought need dare let get got gets getting go goes
going gone went come comes coming came take takes taking took taken make makes making made
say says saying said tell tells telling told know knows knowing knew known think thinks
thought see sees seeing saw seen look looks looking looked want wants wanted give gives
gave given find finds found use uses used work works worked call calls called try tries
tried ask asks asked need needs needed feel feels felt become becomes became leave leaves
left put puts mean means meant keep keeps kept begin begins began seem seems seemed help
helps helped talk talks talked turn turns turned start starts started show shows showed
hear hears heard play plays played run runs ran move moves moved like likes liked live
lives lived believe believes believed hold holds held bring brings brought happen happens
happened write writes wrote written sit sits sat stand stands stood lose loses lost pay
pays paid meet meets met include includes continue continues set sets learn learns learned
change changes changed lead leads led understand understands understood watch watches
follow follows followed stop stops stopped create creates created speak speaks spoke read
reads allow allows allowed add adds added spend spends grow grows grew open opens opened
walk walks walked win wins won offer offers remember remembers love loves loved consider
appear appears buy buys bought wait waits waited serve serves send sends sent expect
expects build builds built stay stays stayed fall falls fell cut cuts reach reaches kill
remain remains suggest suggests raise raises pass passes sell sells require requires report
reports decide decides pull pulls return returns explain explains hope hopes develop carry
break breaks receive receives agree agrees support supports hit hits produce produces eat
eats ate drink drinks sleep sleeps wake wakes drive drives drove sing sings sang
of in on at to from by with for about into over under after before between through during
without within against across behind beyond near among around along beside despite toward
upon since until while because although though unless whether if then than as so but and
or nor yet still just only even also too very quite rather almost enough nearly hardly
never always often sometimes usually rarely again once twice already soon later now then
here there where when how why what which who whom whose everywhere anywhere somewhere
today tomorrow yesterday tonight morning afternoon evening night day week month year hour
minute second time moment while spring summer autumn winter monday friday weekend
good great bad better best worse worst new old young high low long short big small large
little great fine nice happy sad glad sorry sure certain clear easy hard difficult simple
strong weak early late right wrong true false real whole full empty free busy tired ready
warm cold hot cool bright dark heavy light quick slow quiet loud safe close open deep
beautiful lovely terrible awful wonderful amazing interesting boring important serious
person people man woman child children friend family father mother brother sister parent
home house room door window city town country world place way thing things part side end
water food money work job business company team group school book word name life story
hand head eye face heart mind body car street road number problem question answer idea
reason fact case point line letter phone email message note game music film light air
today news weather morning coffee tea water bread rice pizza fruit help thanks please
yes no okay hello hi hey bye well right sure maybe perhaps indeed course actually really
honestly obviously certainly probably possibly exactly absolutely definitely thanks thank
"""


FUNCTION_CORE = """
the a an and or but so because if then that this these those which who what
is are was were be been being am has have had do does did will would can could should
may might must of in on at to from by with for about into over under after before between
i you he she it we they me him her us them my your his its our their
not no very too also just only even still yet again more most much many
one two three first next last other same
there here when where how why while all some any each both
""".split()


def dedupe(seq):
    return list(dict.fromkeys(seq))


def char_options(escape_whitespace: bool = True) -> tuple[dict, dict]:
    """Return (criteria sent to Jev, label -> character)."""
    criteria: dict[str, object] = {}
    back: dict[str, str] = {}
    for ch in dedupe(ALPHABET):
        if escape_whitespace and ch in ESCAPES:
            label = ESCAPES[ch]
            criteria[label] = f"the {label.strip('<>')} character"
        else:
            label = ch
            criteria[label] = None
        back[label] = ch
    return criteria, back


def mine(text: str) -> list[str]:
    """Words appearing in the text, plus simple capitalisation variants."""
    out: list[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z'\-]*", text):
        out.append(word)
        out.append(word.lower() if word[0].isupper() else word.capitalize())
    return dedupe(out)


def propose(text: str, n: int, key: str, model: str, timeout: float) -> list[str]:
    """Ask a frontier model for candidate next words."""
    body = {
        "model": model,
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": PROPOSER_PROMPT.format(text=text, n=n)}],
    }
    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=json.dumps(body).encode(),
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        text_out = "".join(block.get("text", "") for block in data.get("content", []))
        match = re.search(r"\[.*\]", text_out, re.S)
        if not match:
            return []
        return dedupe([str(w) for w in json.loads(match.group(0)) if str(w).strip()])
    except (urllib.error.URLError, ValueError, KeyError):
        return []


STRATEGIES = ("tree", "common", "hybrid", "dynamic")

STRATEGY_HELP = {
    "tree": "picks a word class first, then the word inside it — one round trip, and a vocabulary far past the 255-option ceiling",
    "common": "a fixed list of high-frequency English words — fluent, but it "
              "can never name anything specific",
    "hybrid": "function words plus content words mined from the text so far — "
              "no second model, but it can only recycle what it has seen",
    "dynamic": "a frontier model proposes candidates, Jev picks one — the "
               "architecture Jev is actually designed for",
}


def word_options(strategy: str, prompt: str, generated: str, *, proposer_key: str = "",
                 proposer_model: str = "claude-sonnet-5", proposer_n: int = 200,
                 timeout: float = 45.0) -> tuple[list[str], list[str]]:
    """Return (candidate set, draft head).

    The draft head is the proposer's highest-ranked guesses, used as the
    speculation set. A frontier model drafting and Jev verifying is exactly the
    shape of ordinary speculative decoding.
    """
    draft: list[str] = []
    if strategy == "common":
        # The corpus is larger than a Choice allows, so assemble a slice: the
        # function-word skeleton always, then words already in play (so it can
        # answer in the text's own vocabulary), then frequency fill.
        vocab = PUNCT + FUNCTION_CORE + mine(f"{prompt} {generated}")
        for words, slots in BUDGET:
            vocab += words[:slots]
    elif strategy == "hybrid":
        vocab = PUNCT + FUNCTION_CORE + mine(f"{prompt} {generated}")
    elif strategy == "dynamic":
        proposed = propose(prompt.rstrip() + generated, proposer_n, proposer_key,
                           proposer_model, timeout)
        draft = proposed[:8]
        vocab = PUNCT + proposed + FUNCTION_CORE if proposed else PUNCT + FUNCTION_CORE
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return dedupe(vocab)[:MAX_OPTIONS], draft


def join(parts: list[str]) -> str:
    """Words carry their own leading space, so this appends cleanly to a prompt."""
    return "".join(p if p in PUNCT else " " + p for p in parts)

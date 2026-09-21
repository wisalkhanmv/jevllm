<img src="jevllm/logo.png" alt="" width="84">

# JevLLM

**A language model built out of a model designed never to be one.**

[Jev](https://docs.typesafe.ai) is a *System One* model. It does not write text.
You hand it some state and a closed set of options, and it returns one option
plus a calibrated probability for every candidate. No decoding loop, no
generation — that is the entire point of it.

So: what happens if you make the options *words*, ask "which comes next," append
the answer, and do it again?

```
It was a bright cold day in April and the clocks struck thirteen Winston
```

Orwell's opening has "the clocks were striking thirteen," and Winston is the man
who walks in next. JevLLM got there by asking a decision model the same
question a dozen times in a row.

---

## What this is

A working autoregressive decoder whose only inference primitive is a Jev Choice
call. It is a joke that turned into a measurement.

The joke: Jev is fast *because* it prefills once and reads candidate logits in
parallel instead of decoding token by token. Forcing it to generate text
reintroduces the autoregressive loop — over HTTP, re-sending the whole context
every step. You are paying network latency to undo the thing you were sold.

The measurement is more interesting than the joke. See
[Findings](#findings).

## Install

No dependencies. Python 3.10+.

```bash
git clone https://github.com/wisalkhanmv/jevllm
cd jevllm
cp .env.example .env    # add your key
```

Get a key at [console.typesafe.ai/settings/keys](https://console.typesafe.ai/settings/keys).

## Run

```bash
set -a; source .env; set +a

# the UI — http://localhost:8765
python3 -m jevllm --serve

# or the CLI
python3 -m jevllm "It was a bright cold day in April and the" -n 12
python3 -m jevllm "The capital of France is" -s common -t 0 --show-top 5
python3 -m jevllm "hello wor" -m char -n 6
```

The UI streams every step: the chosen word, the full distribution as bars, a
sparkline of confidence over time, and running cost. **Continue** carries on
from what is already there, feeding the text back in as the prompt, and the
counters accumulate across legs rather than resetting.

The `anti-repeat` control divides the probability of any word used in the last
few steps — higher means less repetition, 1 turns it off. Exact repeats and
repeated word-pairs are blocked outright regardless, so this only shapes the
softer cases.

Watch the bars, not the text — confidence is where the story is.

### Keys

If `JEV_API_KEY` is set in the server's environment — the usual case running
locally with a `.env` — the key field shows **from .env** and you can ignore it.
Otherwise paste a Jev key into the page. It overrides the environment, so you
can point your own key at someone else's deployment.

**A key typed into the page is kept in that browser until you clear it**, so a
reload does not cost you a paste. There is a Clear button beside the field.
It is never stored server-side and never logged. The field is masked with CSS
rather than `type="password"`, so browser password managers do not recognise it
as a credential and never offer to store it separately.

It is sent in a request header on a POST, never in a query string: URLs reach
proxy logs, browser history and `Referer` headers, and a credential has no
business in any of them. Request lines are not logged.

**Only the Jev key is ever accepted from the browser.** `ANTHROPIC_API_KEY` —
used by `dynamic` to propose candidates — is read from the environment and
nowhere else, so a public deployment cannot ask a visitor for a second
credential and `dynamic` is simply not offered there. Run it locally for that.

Steps are capped server-side at 60 per request when the run is spending the
server's key, and 400 when you have pasted your own — the lower cap exists to
protect whoever deployed it, not to restrain you. Both are overridable with
`JEVLLM_MAX_STEPS` and `JEVLLM_MAX_STEPS_OWN`.

What this does *not* protect against: whoever operates a deployment receives
whatever key you paste into it, in plaintext. That is inherent to any
bring-your-own-key web app. Paste keys into deployments you trust, or run it
yourself — which is two commands.

### Deploying

The request handler in `jevllm/server.py` serves the whole app — `/` for the
page, `/api/*` for config and decoding — so it deploys to Vercel as a single
Python function with no framework and no dependencies:

```toml
# pyproject.toml
[tool.vercel]
entrypoint = "jevllm.server:Handler"
```

**Leave `JEV_API_KEY` out of the deployment's environment.** With no server
key the page simply asks each visitor for their own, which is the point: a
public demo that costs you nothing and exposes no credential. Set it only if
you want a private instance that runs on your quota.

## How it works

```
state     = everything generated so far
question  = one Choice whose options are candidate next units
answer    = { choice, probabilities: {...}, confidence }
loop      = sample from probabilities, append, repeat
```

Because Jev returns a real probability per option, temperature, top-k and
nucleus sampling work on it unchanged. Temperature is applied as `p**(1/T)` and
renormalised, which is equivalent to scaling the underlying logits.

### Candidate strategies

The option set *is* the vocabulary, so how you build it decides everything.

| `-s` | what it does | result |
|---|---|---|
| `common` | a word-class-stratified corpus, sliced to fit | **the default** — deterministic, coherent, cheapest |
| `tree` | a categorised vocabulary walked two levels deep | reaches past the 255 ceiling; 3.4x the tokens for the same output |
| `hybrid` | function words + words mined from the text so far | can only recycle what it has already seen |
| `dynamic` | a frontier model proposes candidates, Jev picks | the only one that produces real content |

There are two **modes**, chosen separately from the strategy above:

| mode | candidates | note |
|---|---|---|
| `word` | assembled by the strategy above | the default |
| `char` | 75 printable characters | no strategy applies; confidence collapses |

Character mode needs none of the word-level shaping and is actively harmed by
it: blocking a repeated unit is right for words and makes `ll`, `ss` and `oo`
unspellable, and blocking repeated bigrams means that once `th` has occurred,
`h` could never follow `t` again. The one rule that carries over is that a
space may not follow a space. With that, `hello wor` completes to
`hello world`.

### The tree

A Choice accepts at most 255 options, which caps a flat vocabulary hard — the
`common` strategy offers 246 words and the rest of the corpus is unreachable on
that step. The `tree` strategy removes the ceiling by asking two questions
instead of one: which *category* the next word belongs to, then which word
inside it. Sixteen categories reach 741 words.

The descent costs no extra round trip. Jev ingests the state once and
evaluates every question in a request in parallel, so the category question and
one word question per category all travel together, and only the branch the
category answer picks is ever read.

It also suits the model better in principle. "Is the next word a person, an
action, or a connective?" is a judgment between options that mean genuinely
different things, which is what Jev is calibrated for; picking one of 246 flat
words is not.

**Measured, though, it is not better.** On the same prompt and seed:

| | tokens | cost | speed | mean top p |
|---|---|---|---|---|
| `tree` | 63,735 | $0.0027 | 1.34 w/s | 0.31 |
| `common` | 18,925 | $0.0008 | 2.00 w/s | 0.34 |

Seventeen questions per request is a lot of option tokens even when the state
is sent once, and the output reads about the same. Which is the finding: the
255-option ceiling was never the bottleneck. Ordering is, and a bigger
vocabulary reached more cleverly does not touch it.

Two bugs found on the way, both instructive. The punctuation *category* was
never excluded after a full stop, so the word level fell back and emitted one
anyway — output cycled `.,.?.!`. And the same category kept winning every
step, giving `city street city town city river city`, until the anti-repeat was
applied to categories as well as words. Real prose alternates word classes.

## Findings

Measured across the strategies and modes on the same prompts.

| strategy | mean top p | peak | sample output |
|---|---|---|---|
| `char` | ~0.20 | 0.25 | drifts into whitespace and never recovers |
| `hybrid` | 0.12–0.36 | 0.42 | `was were It.` |
| `common` | 0.23–0.51 | 0.71 | `fine thanks. and you fine am fine. too.` |
| `dynamic` | 0.34–0.54 | **0.82** | `clocks struck thirteen Winston` |

**1. Characters don't work, and the reason is the whole insight.** Jev is
calibrated to choose between options that *mean different things*. `a` versus
`b` is not a judgment — single letters carry no semantic content, so it spreads
probability nearly flat across the alphabet and the sampler wanders into a
whitespace fixed point. Move to words and confidence roughly triples.

It is not that Jev ignores context. Ask it directly and the signal is there:

| context | top candidates | correct |
|---|---|---|
| `...of France is Par` | **i**=0.25, a=.14, s=.13 | ✅ 1st |
| `...cat sat on the ma` | a=.20, **t**=.16, o=.12 | 2nd |
| `...2 plus 2 equals fo` | t=.21, **u**=.17, e=.13 | 2nd |

Real signal, honestly reported as weak. An LLM would say `i` at 0.98. Jev says
0.25 because 0.25 is what it actually knows. That is calibration working.

**2. It knows which words belong. It has no idea what order they go in.**
The clearest run produced every content word of Orwell's sentence —
`clocks`, `were`, `thirteen`, `striking`, `o'clock` — scrambled. Given good
candidates it reliably recognises the passage and reliably fails to sequence it.
Ordering is precisely the machinery a System One model does not have.

**3. `dynamic` is not reproducible, and `common` is.** The proposer returns a
different candidate list on every call, so the same prompt and seed give
`You?. I'm Been good.` on one run and `And you?? And you??` on the next.
`common` gives the same answer every time, which makes it the better one to
measure with — and, with the corpus stratified by word class, the better one to
read. A Choice takes 255 options, so the corpus can never ship whole; it is
sliced per step with a fixed budget for each class. An earlier version simply
truncated a flat frequency-ordered list and spent every content slot on
pronouns and auxiliaries, producing `am are doing? being? are?` while the words
it needed sat unused past the cutoff.

**4. Speculation costs more than it buys.** Cashing in a speculative branch
draws that word from the trimmed option list rather than the full candidate
set, and on the same prompt and seed the difference is `I'm fine. Thanks.`
against `Good. and And But Well is am.` It is off by default.

**5. The punctuation was load-bearing.** Full stops were roughly half the
output, partly through a bug — punctuation was exempt from the loop blocker, so
every step that banned a content word left it untouched — and partly because
Jev genuinely finds a full stop plausible almost anywhere. Suppressing it
entirely turns `thanks. fine. good. and you? fine too.` into `doing fine thanks
you bye see over here there the is i`. The dots were chunking the output into
fragments that each read as plausible, and hiding the ordering weakness. The
`punct` control sets the multiplier; 0.6 is the shipped compromise.

**6. `dynamic` wins on vocabulary, and how it wins is the point.** It only said "clocks"
because the proposer offered "clocks." Jev contributed selection, not
vocabulary — which is the architecture the
[TypeSafe cookbooks](https://docs.typesafe.ai) advocate in the first place:
generate candidates elsewhere, judge with Jev. We arrived at it by failing to
avoid it.

**7. Byte-level decoding is impossible by exactly one option.** The API caps a
Choice at 255 options. Bytes need 256.

## Speed

Latency, not cost, is the constraint — so most of the engineering here is about
removing round trips rather than tokens.

| fix | before | after |
|---|---|---|
| reuse one TLS connection | 1.01 s/call | **0.41 s/call** |
| cache the candidate set | 5 proposals / 10 words | **1 proposal / 16 words** |
| speculative fan-out | 1.00 words/request | 1.1–1.3 words/request — *off by default, see below* |
| **`common` overall** | ~1.0 words/sec | **2.5 words/sec** |
| **`dynamic` overall** | 0.12 words/sec | **0.55 words/sec** |

**Connection reuse is the single biggest win and it is free.** Against this API
a fresh TCP + TLS handshake costs ~610 ms versus ~390 ms of actual model time,
so a naive client spends nearly two thirds of its life reconnecting. One
persistent `HTTPSConnection` more than doubles throughput.

**Caching candidates matters more than it looks.** `dynamic` calls a frontier
model to propose words, which dominates everything else. Proposing once and
reusing the set — refreshing only on a confidence collapse, or every 16 steps —
took `dynamic` from 0.12 to 0.55 words/sec. The first threshold we tried (0.07
is the shipped value) was 0.22, which fired on almost every step, because Jev's
top probability on open continuations genuinely sits around 0.25. Tuning that
number *is* the optimisation.

**Speculative fan-out helps on round trips and hurts on output, so it ships
off.** Jev reads the state once and answers
every question in a request in parallel, so alongside "what comes next?" we ask
"and if it turns out to be X, what then?" for the proposer's top few guesses.
When the sampled word is one we guessed, the next distribution is already in
hand and we advance two words for one round trip. Acceptance runs 11–22%, worth
about 10–30% fewer requests. But a word cashed in from a speculative branch is
drawn from a trimmed 80-option list rather than the full candidate set — every
question carries its own criteria, and full-width speculation triples input
tokens — and the quality cost is severe: `I'm fine. Thanks.` becomes `Good. and
And But Well is am.` on the same prompt and seed. `--speculate N` turns it on.

It is the same structure as ordinary speculative decoding — a cheap model
drafts, an expensive one verifies — except the "expensive" model here costs
$0.042 per million tokens.

### Sliding the context window doesn't help, and that is the interesting part

Every step re-sends the whole text as state, with no KV cache between requests,
so the obvious worry is a quadratic: long outputs paying to re-read themselves
on every step. There is a `--context N` flag that keeps the prompt plus the last
N words, shaped after StreamingLLM — the prompt is the attention sink, so it is
kept whole and the middle is elided rather than the beginning.

Measured over a 36-word generation, it saves nothing:

| context | tokens | cost | mean top p |
|---|---|---|---|
| full | 68,806 | $0.0029 | **0.34** |
| prompt + last 24 | 68,777 | $0.0029 | 0.28 |
| prompt + last 10 | 68,544 | $0.0029 | 0.22 |

Identical to within 0.4%, and confidence falls as the window shrinks. The state
was never the expensive part: each request carries ~246 options at roughly two
tokens each, so about 1,900 tokens of *question* against 65 tokens of state.
The candidate list dominates, and trimming context only throws away quality
that was free.

It is off by default. The lever for cost is the size of the option set, not the
length of the context — which is also why `tree`, at seventeen questions per
request, costs 3.4x what `common` does.

### Still on the table

- **Pipeline the proposer.** The next candidate set could be fetched
  concurrently with the current Jev call, hiding it entirely.
- **Batch independent sequences.** Nothing here helps one sequence, but N
  prompts could share one request.

## Cost

Input is billed at $0.042 per million tokens; output tokens are free. Every
experiment in this README, end to end, cost **under three cents**.

## What this is not

Not a way to get cheap text generation — use a language model for that. Not a
criticism of Jev, which is doing exactly what it says on the tin and is honest
about its uncertainty throughout. Not a benchmark: these are small samples on a
handful of prompts, run against `jev-latest` in September 2026.

It is a demonstration, by construction, of what a System One model *is* — and
the most legible one we could find.

## Layout

```
jevllm/
  api.py      Jev client, sampling (temperature / top-k / nucleus)
  vocab.py    candidate strategies and the word lists
  tree.py     the categorised vocabulary and its two-level walk
  decode.py   the decoding loop — yields one Step per unit
  server.py   local SSE server
  ui.html     the interface
  __main__.py CLI
```

`api.py` is usable on its own if you just want a small, dependency-free Jev
client with sampling attached.

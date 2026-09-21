"""CLI: python -m jevllm ..."""

from __future__ import annotations

import argparse
import sys

from .api import JevClient, JevError, api_key
from .decode import Settings, Stats, decode, text_of
from .vocab import STRATEGIES, STRATEGY_HELP


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="jevllm",
        description="Generate text by asking Jev, one Choice at a time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("prompt", nargs="?", help="Text to continue.")
    ap.add_argument("-n", "--steps", type=int, default=14)
    ap.add_argument("-m", "--mode", default="word", choices=["word", "char"],
                    help="Predict a word at a time, or a character at a time.")
    ap.add_argument("-s", "--strategy", default="tree",
                    choices=[*STRATEGIES, "char"],
                    help="; ".join(f"{k}: {v}" for k, v in STRATEGY_HELP.items()))
    ap.add_argument("-t", "--temperature", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repeat-penalty", type=float, default=1.6)
    ap.add_argument("--punct-bias", type=float, default=0.6, metavar="X",
                    help="Multiplier on punctuation candidates. 1 disables; "
                         "below 1 discourages full stops.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--context", type=int, default=0, metavar="N",
                    help="Keep only the prompt plus the last N generated words "
                         "as state. 0 (default) keeps everything.")
    ap.add_argument("--stop-at", default="", metavar="TEXT",
                    help="Stop once the output ends with this. Several may be "
                         "separated by | . Empty runs to the step count.")
    ap.add_argument("--speculate", type=int, default=0, metavar="K",
                    help="Speculative branches per request. Fewer round trips, "
                         "markedly worse output. 0 (default) disables.")
    ap.add_argument("--show-top", type=int, default=0, metavar="N",
                    help="Print the top N candidates at each step.")
    ap.add_argument("--serve", action="store_true", help="Start the web UI instead.")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if args.serve:
        from .server import serve
        serve(args.port)
        return

    if not args.prompt:
        ap.error("a prompt is required (or pass --serve)")

    key = api_key()
    if not key:
        sys.exit("Set JEV_API_KEY (or TYPESAFE_API_KEY) first.")

    import os
    settings = Settings(
        prompt=args.prompt,
        mode="char" if (args.mode == "char" or args.strategy == "char") else "word",
        strategy="common" if args.strategy == "char" else args.strategy,
        limit=args.steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        punct_bias=args.punct_bias,
        seed=args.seed,
        proposer_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        speculate=args.speculate,
        context_words=args.context,
        stop_at=tuple(x for x in args.stop_at.split('|') if x),
    )

    print(args.prompt, end="", flush=True)
    tops = []
    stats = Stats()
    client = JevClient(key, timeout=settings.timeout)
    try:
        for step in decode(client, settings, stats):
            if args.show_top:
                top = sorted(step.probs.items(), key=lambda kv: kv[1], reverse=True)
                pretty = "  ".join(f"{k!r}={v:.2f}" for k, v in top[: args.show_top])
                print(f"\n\033[2m  [{pretty}]\033[0m", end="", file=sys.stderr, flush=True)
            sep = "" if (step.is_punct or settings.mode == "char") else " "
            print(sep + step.unit, end="", flush=True)
            tops.append(step.top_prob)
            last = step
    except JevError as exc:
        sys.exit(f"\n\n{exc}")
    except KeyboardInterrupt:
        print("\n\033[2m(interrupted)\033[0m", file=sys.stderr)
        return

    finally:
        client.close()

    if tops:
        mean = sum(tops) / len(tops)
        unit = "chars" if settings.mode == "char" else "words"
        print(f"\n\n\033[2m{stats.units} {unit} · {stats.requests} requests · "
              f"{stats.units_per_request:.2f} {unit}/request · "
              f"{stats.speculated} speculated ({stats.acceptance*100:.0f}% accepted) · "
              f"{stats.proposals} proposals\n"
              f"{last.tokens:,} tokens · ${last.cost:.4f} · {last.elapsed:.1f}s · "
              f"{stats.units/last.elapsed:.2f} {unit}/sec · "
              f"mean top p {mean:.2f} · peak {max(tops):.2f}\033[0m", file=sys.stderr)


if __name__ == "__main__":
    main()

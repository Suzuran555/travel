"""Compare Qwen-generated DSL vs oracle DSL over the pretranslate cache.

Reads cache/translation_Qwen3.6-27B_reflect/*.json (written by pretranslate.py,
which stashes the oracle under _oracle_hard_logic_py). Reports exact-set match
rate, per-constraint precision/recall, and syntax validity of generated code.
"""
import glob
import json
import os
import sys


def norm(c):
    return " ".join(str(c).split())


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Qwen3.6-27B"
    cache = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cache",
        f"translation_{name}_reflect",
    )
    files = sorted(glob.glob(os.path.join(cache, "*.json")))
    if not files:
        sys.exit(f"no cached translations in {cache}")
    exact = 0
    tp = fp = fn = 0
    syntax_bad = 0
    per_uid = []
    for f in files:
        d = json.load(open(f))
        gen = d.get("hard_logic_py") or []
        ora = d.get("_oracle_hard_logic_py") or []
        if isinstance(gen, dict):
            gen = list(gen.values())
        gset = {norm(c) for c in gen}
        oset = {norm(c) for c in ora}
        ok = gset == oset
        exact += ok
        tp += len(gset & oset)
        fp += len(gset - oset)
        fn += len(oset - gset)
        bad = 0
        for c in gen:
            try:
                compile(str(c), "<gen>", "exec")
            except SyntaxError:
                bad += 1
        syntax_bad += bad
        per_uid.append((os.path.basename(f)[:-5], ok, len(gset & oset), len(oset), bad))
    n = len(files)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    print(f"uids compared: {n}")
    print(f"exact-set match: {exact}/{n} = {exact/n:.1%}")
    print(f"constraint precision {prec:.1%}  recall {rec:.1%}  (tp {tp} fp {fp} fn {fn})")
    print(f"generated constraints with syntax errors: {syntax_bad}")
    misses = [r for r in per_uid if not r[1]]
    print(f"\nfirst mismatched uids ({min(len(misses),15)} of {len(misses)}):")
    for uid, _, hit, tot, bad in misses[:15]:
        print(f"  {uid}: {hit}/{tot} oracle constraints reproduced, {bad} syntax errors")


if __name__ == "__main__":
    main()

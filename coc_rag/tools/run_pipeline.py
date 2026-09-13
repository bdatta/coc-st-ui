"""
Run the whole CLI pipeline on one PDF: convert, validate, inventory, chunk, audit.

Streamlit caches imported modules, so a session started before a code change
silently converts with stale code. Running end to end here guarantees the code
on disk is the code that runs, and stops on the first hard failure rather than
carrying a broken conversion forward into embedding.

    python tools/run_pipeline.py coc.pdf
    python tools/run_pipeline.py coc.pdf --first 8 --doc-id acme_2026
    python tools/run_pipeline.py coc.pdf --outdir build --toc toc.txt
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

TOOLS = pathlib.Path(__file__).resolve().parent
# The project root must be importable: --config reads coc.pipeline_config here,
# and the child steps resolve it themselves.
sys.path.insert(0, str(TOOLS.parent))


def run(step: str, argv: list[str], allow_fail: bool = False) -> int:
    print(f"\n{'=' * 74}\n{step}\n{'=' * 74}")
    result = subprocess.run([sys.executable, *argv])
    if result.returncode and not allow_fail:
        print(f"\n!! {step} failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)
    return result.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full COC CLI pipeline.")
    ap.add_argument("pdf")
    ap.add_argument("--outdir", default=".", help="where to write outputs")
    ap.add_argument("--doc-id")
    ap.add_argument("--first", type=int, help="first PDF page (skip front matter)")
    ap.add_argument("--last", type=int)
    ap.add_argument("--toc", help="table-of-contents file, enables coverage check")
    ap.add_argument("--plan-id")
    ap.add_argument("--plan-year", type=int)
    ap.add_argument("--carrier")
    ap.add_argument("--bold-headings", action="store_true")
    ap.add_argument("--config", help="pipeline YAML")
    ap.add_argument("--embed", action="store_true",
                    help="also embed and store in Atlas (step 6)")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --embed, report cost and stop before embedding")
    ap.add_argument("--resume", action="store_true",
                    help="with --embed, skip chunks already stored")
    ap.add_argument("--force", action="store_true",
                    help="embed even if validation did not pass")
    args = ap.parse_args()

    cfg = None
    if args.config:
        from coc.pipeline_config import build_config

        cfg = build_config(args.config, {})
        for name in ("doc_id", "toc", "plan_id", "plan_year", "carrier"):
            if getattr(args, name, None) is None:
                setattr(args, name, getattr(cfg, name, None))
        if args.first is None:
            args.first = cfg.first_page
        if args.last is None:
            args.last = cfg.last_page
        if args.outdir == ".":
            args.outdir = cfg.outdir

    pdf = pathlib.Path(args.pdf).resolve()
    if not pdf.exists():
        print(f"No such file: {pdf}")
        sys.exit(2)

    outdir = pathlib.Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = pdf.stem.replace(" ", "_")
    md = outdir / f"{stem}.md"
    tables = outdir / f"{stem}.tables.json"
    chunks = outdir / f"{stem}.chunks.jsonl"
    report = outdir / f"{stem}.validation.json"
    doc_id = args.doc_id or stem.lower()

    convert = [str(TOOLS / "convert.py"), str(pdf), "-o", str(md),
               "--tables-json", str(tables)]
    if args.first:
        convert += ["--first", str(args.first)]
    if args.last:
        convert += ["--last", str(args.last)]
    if args.bold_headings:
        convert += ["--bold-headings"]
    run("1/5  CONVERT   PDF -> Markdown", convert)

    validate = [str(TOOLS / "validate.py"), str(md), "--pdf", str(pdf),
                "--json", str(report)]
    if args.toc:
        validate += ["--toc", args.toc]
    # exit 1 = review, 2 = quarantine; surface both but let the caller decide.
    verdict = run("2/5  VALIDATE  score + verdict", validate, allow_fail=True)

    run("3/5  INVENTORY named divisions",
        [str(TOOLS / "inventory.py"), str(md), "--divisions"], allow_fail=True)

    chunk = [str(TOOLS / "chunk.py"), str(md), "-o", str(chunks),
             "--doc-id", doc_id, "--pdf", str(pdf)]
    for flag, value in (("--plan-id", args.plan_id),
                        ("--plan-year", args.plan_year),
                        ("--carrier", args.carrier)):
        if value:
            chunk += [flag, str(value)]
    run("4/5  CHUNK     Markdown -> chunks", chunk)

    run("5/6  AUDIT     do split tables reassemble?" if args.embed
        else "5/5  AUDIT     do split tables reassemble?",
        [str(TOOLS / "table_audit.py"), "--from-jsonl", str(chunks)],
        allow_fail=True)

    if args.embed:
        if verdict != 0 and not args.force:
            print(f"\n{'=' * 74}")
            print("6/6  EMBED     SKIPPED")
            print("=" * 74)
            print("   Validation did not pass, so nothing was embedded. Embedding a"
                  "\n   document that failed validation costs money and produces an"
                  "\n   index nobody should trust. Fix the conversion and re-run, or"
                  "\n   pass --force if you have reviewed the findings and accept them.")
        else:
            embed = [str(TOOLS / "embed.py"), str(chunks), "--doc-id", doc_id,
                     "--markdown", str(md), "--tables", str(tables)]
            if args.config:
                embed += ["--config", args.config]
            if args.dry_run:
                embed += ["--dry-run"]
            if args.resume:
                embed += ["--resume"]
            run("6/6  EMBED     chunks -> Atlas", embed)

    print(f"\n{'=' * 74}\nDONE\n{'=' * 74}")
    print(f"   markdown   {md}")
    print(f"   tables     {tables}")
    print(f"   chunks     {chunks}")
    print(f"   validation {report}")
    if verdict == 0:
        print("\n   Validation PASSED. Ready to embed:")
        print("   upload the Markdown and tables JSON on Phase 2, or re-use this doc_id.")
    elif verdict == 1:
        print("\n   Validation returned REVIEW. Read the findings above before embedding.")
    else:
        print("\n   Validation QUARANTINED this document. Do not embed it.")
        print("   Fix the conversion settings or the code and re-run — do not hand-edit")
        print("   the Markdown, it is derived and the edit will be lost.")
    sys.exit(verdict)


if __name__ == "__main__":
    main()

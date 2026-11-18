"""Text-to-SQL latency: our model vs voice-to-sql's SQLCoder-7B (PLAN-week3 X7, RQ3).

Both run nf4 4-bit with HF generate, batch 1, greedy, on the same GPU, one system per process.
Input: the 100 clean-SNR ASR transcripts. 5 warm-up queries are run first and not timed.
Each query is timed from prompt rendering to decoded text, with CUDA synchronised.

    ~/venvs/tinysql/bin/python scripts/bench_latency.py --system ours --adapter checkpoints/v2/checkpoint-900
    ~/venvs/tinysql/bin/python scripts/bench_latency.py --system sqlcoder

SQLCoder uses voice-to-sql's prompt, schema format, model settings and SQL extraction, copied below from
C:/Users/alper/Desktop/4_2/speech/project/src/text_to_sql.py (_build_prompt, build_schema,
_SQLCoderBackend, _extract_sql). As a reproduction check, its extracted SQL is compared with the SQL
voice-to-sql stored for the same transcripts.

End-to-end estimate: per row, voice-to-sql's recorded total minus its recorded SQL stage plus the
latency measured here.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from pathlib import Path

import torch

from tinysql.stats import latency_summary
from tinysql.voice import load_voice, voice_rows

ROOT = Path(__file__).resolve().parents[1]
SPIDER = ROOT / "data" / "spider"
SQLCODER = "defog/sqlcoder-7b-2"
# voice-to-sql downloaded SQLCoder on the Windows side; reuse that cache instead of a 13 GB re-download.
WINDOWS_HF_CACHE = "/mnt/c/Users/alper/.cache/huggingface/hub"
WARMUP = 5


# ---- copied from voice-to-sql src/text_to_sql.py ---------------------------------------------
def _spider_type_to_sql(spider_type: str) -> str:
    return {"text": "TEXT", "number": "REAL", "time": "TEXT", "boolean": "INTEGER",
            "others": "TEXT"}.get(spider_type.lower(), "TEXT")


def build_schema(entry: dict) -> str:
    table_names = entry["table_names_original"]
    col_types = entry["column_types"]
    primary_keys = set(entry.get("primary_keys", []))
    tables: dict[int, list] = {i: [] for i in range(len(table_names))}
    for col_idx, (tbl_idx, col_name) in enumerate(entry["column_names_original"]):
        if tbl_idx == -1:
            continue
        tables[tbl_idx].append((col_idx, col_name, _spider_type_to_sql(col_types[col_idx])))
    lines: list[str] = []
    for tbl_idx, tbl_name in enumerate(table_names):
        col_defs = [f"    {c} {t}{' PRIMARY KEY' if i in primary_keys else ''}" for i, c, t in tables[tbl_idx]]
        lines.append(f"CREATE TABLE {tbl_name} (")
        lines.append(",\n".join(col_defs))
        lines.append(");\n")
    return "\n".join(lines)


def sqlcoder_prompt(transcript: str, schema: str) -> str:
    return (
        "### Task\n"
        f"Generate a SQL query to answer [QUESTION]{transcript}[/QUESTION]\n\n"
        "### Database Schema\n"
        "The query will run on a database with the following schema:\n"
        f"{schema}\n"
        "### Answer\n"
        "Given the database schema, here is the SQL query that answers "
        f"[QUESTION]{transcript}[/QUESTION]\n"
        "[SQL]\n"
    )


def extract_sql(raw: str) -> str:
    fence = re.search(r"```(?:sql)?\s*(SELECT[\s\S]+?)```", raw, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    tag = re.search(r"\[SQL\]\s*(SELECT[\s\S]+)", raw, re.IGNORECASE)
    if tag:
        return re.split(r"\n\s*\n|\[/SQL\]", tag.group(1).strip())[0].strip()
    m = re.search(r"(SELECT[\s\S]+?;)", raw, re.IGNORECASE)
    return m.group(1).strip() if m else raw.strip()
# ------------------------------------------------------------------------------------------------


class SQLCoder:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                 bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.tok = AutoTokenizer.from_pretrained(SQLCODER, cache_dir=WINDOWS_HF_CACHE)
        self.model = AutoModelForCausalLM.from_pretrained(
            SQLCODER, cache_dir=WINDOWS_HF_CACHE, quantization_config=bnb, dtype=torch.bfloat16,
            device_map="cuda:0")
        self.model.eval()
        schemas = {e["db_id"]: e for e in json.loads((SPIDER / "tables.json").read_text(encoding="utf-8"))}
        self.schemas = schemas

    @torch.inference_mode()
    def __call__(self, row: dict) -> tuple[str, int]:
        prompt = sqlcoder_prompt(row["transcript"], build_schema(self.schemas[row["db_id"]]))
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        # voice-to-sql: text-generation pipeline, max_new_tokens=300, greedy, pad = eos.
        out = self.model.generate(**enc, max_new_tokens=300, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        new = out[0, enc["input_ids"].shape[1]:]
        return extract_sql(self.tok.decode(new, skip_special_tokens=True)), len(new)


class Ours:
    def __init__(self, model: str, adapter: Path | None):
        from tinysql.hf import HFGenerator
        from tinysql.prompt import build_messages, serialize_schema

        self.gen = HFGenerator(model, adapter)
        self.build_messages, self.serialize_schema = build_messages, serialize_schema
        self.schemas: dict[str, str] = {}

    def __call__(self, row: dict) -> tuple[str, int]:
        from tinysql.reward import extract_sql as ours_extract

        db = row["db_id"]
        if db not in self.schemas:  # schema text is cached, as in evaluate(); not part of the timing after warm-up
            self.schemas[db] = self.serialize_schema(SPIDER / "database" / db / f"{db}.sqlite")
        completion = self.gen([self.build_messages(self.schemas[db], row["transcript"])])[0]
        n_tokens = len(self.gen.tok(completion, add_special_tokens=False)["input_ids"])
        return ours_extract(completion) or "", n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["ours", "sqlcoder"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke test")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA not available: is the CPU torch wheel installed?"

    rows = voice_rows(load_voice(ROOT / "data" / "spokenspider_transcripts.csv"), "clean")[: args.limit]
    t_load = time.perf_counter()
    system = SQLCoder() if args.system == "sqlcoder" else Ours(args.model, args.adapter)
    load_s = time.perf_counter() - t_load

    for row in rows[:WARMUP]:
        system(row)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    per_row = []
    for row in rows:
        t0 = time.perf_counter()
        sql, n_tokens = system(row)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        per_row.append({
            "query_id": row["query_id"], "latency_s": dt, "new_tokens": n_tokens, "sql": sql,
            "same_sql_as_voice_to_sql": sql.strip() == row["sqlcoder_sql"].strip(),
            "e2e_estimate_s": float(row["latency_total_s"]) - float(row["latency_sql_s"]) + dt,
        })

    name = args.system if args.system == "sqlcoder" else (
        "ours_" + args.model.replace("/", "__") + (f"_{args.adapter.parent.name}-{args.adapter.name}" if args.adapter else ""))
    summary = {
        "system": name,
        "model": SQLCODER if args.system == "sqlcoder" else args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "setup": "nf4 4-bit, HF generate, batch 1, greedy, clean transcripts, "
                 f"{WARMUP} warm-up queries untimed",
        "max_new_tokens": 300 if args.system == "sqlcoder" else 256,
        "text_to_sql": latency_summary([r["latency_s"] for r in per_row]),
        "new_tokens": latency_summary([float(r["new_tokens"]) for r in per_row]),
        "e2e_estimate": latency_summary([r["e2e_estimate_s"] for r in per_row]),
        "voice_to_sql_recorded_sql_stage": latency_summary([float(r["latency_sql_s"]) for r in rows]),
        "voice_to_sql_recorded_total": latency_summary([float(r["latency_total_s"]) for r in rows]),
        "same_sql_as_voice_to_sql": sum(r["same_sql_as_voice_to_sql"] for r in per_row),
        "load_seconds": round(load_s, 1),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "limit": args.limit,
    }
    suffix = f"_limit{args.limit}" if args.limit else ""
    out = ROOT / "results" / f"latency_{name}{suffix}.json"
    out.write_text(json.dumps({"summary": summary, "rows": per_row}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

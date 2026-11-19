"""tiny-sql-rl demo: ask a Spider database a question by text or voice (PLAN-week4 Y5).

    TINYSQL_GGUF=~/gguf/v2-900-merged-Q4_K_M.gguf TINYSQL_TOKENIZER=checkpoints/v2-900-merged python demo/app.py

Set TINYSQL_VOICE=0 to hide the microphone (text-only Space).
Databases: a few Spider databases (Yu et al., 2018; CC BY-SA 4.0) in demo/databases/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runtime import load_generator, load_transcriber, timed  # noqa: E402

from tinysql.demo import answer_question  # noqa: E402

DB_DIR = Path(os.environ.get("TINYSQL_DB_DIR", HERE / "databases"))
VOICE = os.environ.get("TINYSQL_VOICE", "1") != "0"
SAMPLE_ROWS = int(os.environ.get("TINYSQL_SAMPLE_ROWS", "3"))
# Questions the released model answers correctly (checked by hand). It is wrong on roughly 4 in 10
# Spider dev questions, so free-form questions will sometimes fail; that is shown, not hidden.
EXAMPLES = {
    "concert_singer": ["How many singers do we have?", "List the names of singers older than 40.",
                       "How many concerts were held in 2014?"],
    "pets_1": ["What is the heaviest pet type?", "How many pets does each student have?"],
    "car_1": ["How many cars have more than 6 cylinders?"],
    "world_1": ["How many countries speak French?", "Which continent has the most countries?"],
    "dog_kennels": ["Which breed is the most common?", "How many dogs have never been treated?"],
    "singer": ["List the singers born after 1950.", "Who has the highest net worth?",
               "How many singers are from France?"],
}


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def run(db_id, question, audio):
    timings, transcript = {}, ""
    if VOICE and audio:
        transcript, timings["speech-to-text"] = timed(load_transcriber(), audio)
        question = transcript
    # 3 example rows per table in the prompt (PLAN-week7: +2.5 official EX on Spider dev, better on voice).
    result, timings["text-to-SQL + query"] = timed(
        lambda q, p: answer_question(q, p, load_generator(), sample_rows=SAMPLE_ROWS), question or "", db_path(db_id))
    status = result.error and f"⚠ {result.error}" or result.answer
    timing = " · ".join(f"{k}: {v:.1f} s" for k, v in timings.items())
    if result.repaired:
        first = f"\n\n`{result.first_sql}`" if result.first_sql else ""
        timing = (f"🔧 **Repaired:** the first answer failed (`{result.first_error}`), so the model was asked "
                  f"once to fix it.{first}\n\n{timing}")
    if result.grounding:
        fixes = ", ".join(f"`'{old}'` → `'{new}'`" for old, new in result.grounding)
        timing = f"🔤 **Matched to the database's values:** {fixes}\n\n{timing}"
    if result.rows:
        table = gr.update(value={"data": [list(r) for r in result.rows], "headers": list(result.columns)}, visible=True)
    else:
        table = gr.update(visible=False)
    return transcript, result.sql or "", table, status, timing


with gr.Blocks(title="tiny-sql-rl") as app:
    gr.Markdown("# tiny-sql-rl\nQwen3-1.7B trained with GRPO on an execution reward (59% execution accuracy on "
                "Spider dev). Ask a question about a Spider database; the model writes SQL, which runs read-only. "
                "Runs on CPU, so expect a few seconds. The examples are questions it gets right; expect it to be "
                "wrong on roughly 4 in 10 of your own.")
    with gr.Row():
        db = gr.Dropdown(sorted(EXAMPLES), value="concert_singer", label="Database")
        question = gr.Textbox(label="Question", scale=3)
    audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="…or ask by voice", visible=VOICE)
    go = gr.Button("Ask", variant="primary")
    transcript = gr.Textbox(label="Transcript", visible=VOICE, interactive=False)
    sql = gr.Code(label="Generated SQL", language="sql", interactive=False)
    answer = gr.Textbox(label="Answer", interactive=False)
    table = gr.Dataframe(label="Result", interactive=False, wrap=True, visible=False)
    timing = gr.Markdown()
    gr.Examples([[d, q] for d, qs in EXAMPLES.items() for q in qs], inputs=[db, question])
    go.click(run, [db, question, audio], [transcript, sql, table, answer, timing])
    question.submit(run, [db, question, audio], [transcript, sql, table, answer, timing])

if __name__ == "__main__":
    app.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))

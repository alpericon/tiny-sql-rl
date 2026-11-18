import pytest

from tinysql.llamacpp import LlamaServerGenerator


class FakeServer:
    def __init__(self, replies):
        self.replies, self.payloads = list(replies), []

    def __call__(self, payload):
        self.payloads.append(payload)
        return self.replies.pop(0)


def render(messages):
    return "<prompt>" + messages[-1]["content"]


def reply(text, n, stop_type="eos"):
    return {"content": text, "tokens_predicted": n, "stop_type": stop_type}


def test_greedy_payload_and_outputs_in_order():
    server = FakeServer([reply("```sql\nSELECT 1\n```", 9), reply("x", 256, "limit")])
    gen = LlamaServerGenerator(render, post=server, max_new_tokens=256)
    out = gen([[{"role": "user", "content": "q1"}], [{"role": "user", "content": "q2"}]])
    assert out == ["```sql\nSELECT 1\n```", "x"]
    p = server.payloads[0]
    assert p["prompt"] == "<prompt>q1"
    assert p["n_predict"] == 256 and p["temperature"] == 0 and p["top_k"] == 1
    assert gen.hit_limit == [False, True]
    assert gen.new_tokens == [9, 256]


def test_server_error_raises_instead_of_scoring_empty():
    gen = LlamaServerGenerator(render, post=FakeServer([{"error": {"message": "boom"}}]))
    with pytest.raises(RuntimeError):
        gen([[{"role": "user", "content": "q"}]])

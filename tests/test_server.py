"""Проверка HTTP-контракта без внешнего LLM и ключа."""
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from serve import MAX_BODY_BYTES, make_handler


@pytest.fixture
def server(tmp_path):
    class Bot:
        llm = False
        model = None

        def ask(self, question, history, context=None):
            return {"answer": question, "history": history, "context": context, "mode": "test"}

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Bot(), tmp_path))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield server
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)


def post(server, body, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    connection.request("POST", "/api/ask", body.encode(), headers or {"Content-Type": "application/json"})
    response = connection.getresponse()
    result = response.status, json.loads(response.read())
    connection.close()
    return result


@pytest.mark.parametrize("body", ['{', '[]', '{"question":null}', '{"question":""}', '{"question":"x","history":{}}', '{"question":"x","history":[null]}'])
def test_bad_requests_return_json_error(server, body):
    status, response = post(server, body)
    assert status == 400 and response["error"]


def test_large_body_is_rejected(server):
    status, _ = post(server, "x" * (MAX_BODY_BYTES + 1))
    assert status == 413


def test_cross_origin_post_is_rejected(server):
    status, _ = post(server, '{"question":"hello"}', {"Content-Type": "application/json", "Origin": "https://example.org"})
    assert status == 403


def test_plaintext_post_is_rejected(server):
    status, _ = post(server, '{"question":"hello"}', {"Content-Type": "text/plain"})
    assert status == 415


def test_valid_history_is_bounded_and_sanitized(server):
    history = [{"role": "user", "content": str(index), "extra": "ignored"} for index in range(9)]
    status, result = post(server, json.dumps({"question": " Вопрос ", "history": history}))
    assert status == 200 and result["answer"] == "Вопрос"
    assert len(result["history"]) == 6
    assert result["history"][0] == {"role": "user", "content": "3"}


@pytest.mark.parametrize("context", ['[]', '{"review": "1"}', '{"review": [1]}', '{"review": ["12a"]}',
                                     '{"selected": 5}', '{"review": ["1"], "selected": "x"}'])
def test_bad_screen_context_is_rejected(server, context):
    status, response = post(server, '{"question": "Кто собирает деньги с этих?", "context": %s}' % context)
    assert status == 400 and response["error"]


def test_screen_context_is_passed_to_assistant(server):
    body = {"question": "Кто собирает деньги с этих?", "context": {"review": ["100000000000000100"], "selected": None}}
    status, result = post(server, json.dumps(body))
    assert status == 200 and result["context"] == {"review": ["100000000000000100"], "selected": None}

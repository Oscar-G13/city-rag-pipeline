import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
import main
import security


class QuerySecurity(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"CITY_RAG_API_KEY": "unit-test-only-key-that-is-not-deployed"})
        self.env.start()
        security._requests.clear()
        self.engine = Mock()
        self.engine.query.return_value = "stubbed answer"
        self.mock = patch.object(main, "get_engine", return_value=self.engine)
        self.engine_factory = self.mock.start()
        self.client = TestClient(main.app)
        self.headers = {"Authorization": "Bearer " + os.environ["CITY_RAG_API_KEY"]}

    def tearDown(self):
        self.mock.stop()
        self.env.stop()

    def test_queries_and_warmup_require_authentication(self):
        for path, method in [("/query", "post"), ("/warmup", "get")]:
            self.assertEqual(getattr(self.client, method)(path).status_code, 401)
        self.engine_factory.assert_not_called()

    def test_unconfigured_service_fails_closed(self):
        with patch.dict(os.environ, {"CITY_RAG_API_KEY": ""}):
            self.assertEqual(self.client.post("/query", json={"query": "hello"}, headers=self.headers).status_code, 503)
        self.engine_factory.assert_not_called()

    def test_authorized_request_works_and_input_errors_stay_422(self):
        self.assertEqual(self.client.post("/query", json={"message": "hello"}, headers=self.headers).json(), {"response": "stubbed answer"})
        self.assertEqual(self.client.post("/query", json={}, headers=self.headers).status_code, 422)

    def test_wrong_key_cannot_consume_provider_api(self):
        self.assertEqual(self.client.post("/query", json={"query": "hello"}, headers={"Authorization": "Bearer wrong"}).status_code, 401)
        self.engine_factory.assert_not_called()

    def test_large_body_and_query_rejected(self):
        self.assertEqual(self.client.post("/query", json={"query": "x" * 2001}, headers=self.headers).status_code, 422)
        self.assertEqual(self.client.post("/query", content=b"x" * 16385, headers=self.headers).status_code, 413)
        self.engine_factory.assert_not_called()

    def test_provider_errors_do_not_leak_details(self):
        self.engine.query.side_effect = RuntimeError("private provider detail")
        response = self.client.post("/query", json={"query": "hello"}, headers=self.headers)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private provider detail", response.text)

    def test_rate_limit_and_no_wildcard_cors(self):
        for _ in range(10):
            self.assertEqual(self.client.get("/warmup", headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get("/warmup", headers=self.headers).status_code, 429)
        response = self.client.options("/query", headers={"Origin": "https://untrusted.invalid", "Access-Control-Request-Method": "POST"})
        self.assertNotIn("access-control-allow-origin", response.headers)

if __name__ == "__main__":
    unittest.main()

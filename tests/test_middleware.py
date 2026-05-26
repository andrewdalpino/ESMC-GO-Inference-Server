import logging
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.middleware import ExceptionHandler, TokenAuthentication, ResponseTime


class TestExceptionHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.middleware = ExceptionHandler(app=MagicMock())

    async def test_dispatch_catches_http_exception(self):
        async def call_next(request):
            raise HTTPException(status_code=404, detail="Not Found")

        response = await self.middleware.dispatch(MagicMock(), call_next)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.body, b'{"message":"Not Found"}')

    async def test_dispatch_catches_generic_exception(self):
        async def call_next(request):
            raise RuntimeError("server error")

        with patch.object(logging, "error") as mock_log:
            response = await self.middleware.dispatch(MagicMock(), call_next)

        mock_log.assert_called_once()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.body, b'{"message":"Something went wrong."}')

    async def test_dispatch_passthrough_on_success(self):
        async def call_next(request):
            return JSONResponse(content={"status": "ok"}, status_code=200)

        response = await self.middleware.dispatch(MagicMock(), call_next)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body, b'{"status":"ok"}')


class TestTokenAuthentication(unittest.TestCase):
    def test_init_stores_api_token(self):
        middleware = TokenAuthentication(app=MagicMock(), api_token="test-token")

        self.assertEqual(middleware.api_token, "test-token")

    def test_missing_token_returns_401(self):
        app = FastAPI()
        app.add_middleware(TokenAuthentication, api_token="secret123")

        @app.get("/test")
        async def test_route():
            return {"status": "ok"}

        response = TestClient(app).get("/test")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"message": "Unauthorized."})

    def test_wrong_token_returns_401(self):
        app = FastAPI()
        app.add_middleware(TokenAuthentication, api_token="secret123")

        @app.get("/test")
        async def test_route():
            return {"status": "ok"}

        response = TestClient(app).get(
            "/test", headers={"Authorization": "Bearer wrong-token"}
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"message": "Unauthorized."})

    def test_correct_token_passes(self):
        app = FastAPI()
        app.add_middleware(TokenAuthentication, api_token="secret123")

        @app.get("/test")
        async def test_route():
            return {"status": "ok"}

        response = TestClient(app).get(
            "/test", headers={"Authorization": "Bearer secret123"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


class TestResponseTime(unittest.TestCase):
    def test_response_has_x_response_time_header(self):
        app = FastAPI()
        app.add_middleware(ResponseTime)

        @app.get("/test")
        async def test_route():
            return {"status": "ok"}

        response = TestClient(app).get("/test")

        self.assertIn("X-Response-Time", response.headers)

    def test_response_time_is_valid_float(self):
        app = FastAPI()
        app.add_middleware(ResponseTime)

        @app.get("/test")
        async def test_route():
            return {"status": "ok"}

        response = TestClient(app).get("/test")
        duration = float(response.headers["X-Response-Time"])

        self.assertGreaterEqual(duration, 0.0)

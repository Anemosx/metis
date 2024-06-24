import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestBase:

    @pytest.mark.asyncio
    def test_get_root(self, client: TestClient) -> None:
        response = client.get(f"/")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["Content-Type"] == "text/html; charset=utf-8"

        html_content = response.text

        assert "html" in html_content
        assert "<body>" in html_content

import os

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

base_html_file_path = os.path.join("static", "base", "index.html")
with open(base_html_file_path, "r") as file:
    base_content = file.read()


@router.get("/", response_class=HTMLResponse)
async def read_root() -> HTMLResponse:
    """
    Serve the Base HTML page from the static directory.

    Returns
    -------
    response : HTMLResponse
        An HTML response containing the Base page content.
    """

    response = HTMLResponse(content=base_content)
    response.headers["Cache-Control"] = "public, max-age=3600"

    return response

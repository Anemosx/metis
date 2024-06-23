import os

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

mnist_html_file_path = os.path.join("static", "mnist", "index.html")
with open(mnist_html_file_path, "r") as file:
    mnist_content = file.read()


@router.get("/", response_class=HTMLResponse)
async def read_root() -> HTMLResponse:
    """
    Serve the MNIST HTML page from the static directory.

    Returns
    -------
    response : HTMLResponse
        An HTML response containing the MNIST page content.
    """

    response = HTMLResponse(content=mnist_content)
    response.headers["Cache-Control"] = "public, max-age=3600"

    return response

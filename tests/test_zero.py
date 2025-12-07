from playwright.sync_api import Page

from shiny.playwright import controller
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

app = create_app_fixture(r"..\peregrin_app\app.py")


def test_zero(page: Page, app: ShinyAppProc):

    page.goto(app.url)
    # Add test code here
    assert page.title() == "Peregrin"
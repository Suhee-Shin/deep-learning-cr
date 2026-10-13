"""Optional Playwright QA; not needed on the optimization computer."""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--browser', required=True)
args = ap.parse_args()
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=args.browser, headless=True)
    for name, width, height in [('desktop', 1440, 1000), ('mobile', 390, 844)]:
        page = browser.new_page(viewport=dict(width=width, height=height))
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto((args.out/'index.html').resolve().as_uri())
        page.wait_for_function("document.querySelector('#step').textContent.includes('2 / 2')")
        page.wait_for_function("document.querySelector('#plot').complete && document.querySelector('#plot').naturalWidth > 0")
        page.wait_for_function("document.querySelector('#shape').complete && document.querySelector('#shape').naturalWidth > 0")
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), name
        assert not errors, errors
        page.screenshot(path=str(args.out/(name+'.png')), full_page=True)
        print(name, page.locator('#step').inner_text(), page.locator('#ty').inner_text())
        page.close()
    browser.close()

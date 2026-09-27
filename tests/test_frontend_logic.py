"""Execute frontend pure functions in node to verify escaping behavior."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "app" / "static" / "app.js"


@pytest.mark.skipif(
    shutil.which("node") is None,
    reason="environmental guard: frontend logic tests need node; CI installs it via setup-node",
)
def test_safe_escapes_markup_and_quotes():
    source = APP_JS.read_text(encoding="utf-8")
    match = re.search(r"^ *const safe = .*$", source, re.MULTILINE)
    assert match, "safe() must be a single-line definition in app.js"
    payload = "<img src=x onerror=alert(1)> \" ' &"
    script = f"{match.group(0)}\nprocess.stdout.write(safe({json.dumps(payload)}));"
    result = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    assert result.stdout == "&lt;img src=x onerror=alert(1)&gt; &quot; &#39; &amp;"

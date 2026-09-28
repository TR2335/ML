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


def _extract(name: str) -> str:
    source = APP_JS.read_text(encoding="utf-8")
    match = re.search(rf"^ *const {name} = .*$", source, re.MULTILINE)
    assert match, f"app.js 缺少顶层纯函数 {name}"
    return match.group(0).strip().rstrip(";")


@pytest.mark.skipif(
    shutil.which("node") is None,
    reason="environmental guard: frontend logic tests need node; CI installs it via setup-node",
)
def test_select_score_never_falls_back_to_other_target():
    snippet = _extract("selectScore")
    program = f"""
    {snippet};
    const p = {{risks: {{deterioration_24h: {{score: 0.8, band: 'high'}}}}}};
    const missing = selectScore(p, 'readmission_30d');
    const present = selectScore(p, 'deterioration_24h');
    const absent = selectScore(null, 'deterioration_24h');
    console.log(JSON.stringify([missing, present?.score, absent]));
    """
    result = subprocess.run(
        ["node", "-e", program], capture_output=True, text=True, timeout=30, check=True
    )
    assert json.loads(result.stdout) == [None, 0.8, None]


@pytest.mark.skipif(
    shutil.which("node") is None,
    reason="environmental guard: frontend logic tests need node; CI installs it via setup-node",
)
def test_show_does_not_use_generic_risk_fallback():
    # 全文件语法守卫：防止行级改写留下未闭合的 try 等损坏
    check = subprocess.run(
        ["node", "--check", str(APP_JS)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert check.returncode == 0, check.stderr
    source = APP_JS.read_text(encoding="utf-8")
    # 目标分数回退 `|| p.risk` 必须移除（切换目标不得显示错误目标分数）
    assert "|| p.risk" not in source
    # 详情渲染必须经由 selectScore
    assert "selectScore(p, $('target').value)" in source
    # 静默失败治理：所有 fetch 经 getJSON 包装（response.ok 检查）
    assert "const getJSON =" in source

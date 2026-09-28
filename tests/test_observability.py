def test_metrics_render_labels_and_type_family():
    from app.observability import Metrics

    registry = Metrics()
    registry.inc("medintel_http_responses_total", labels={"code": "200"})
    registry.inc("medintel_http_responses_total", labels={"code": "200"})
    registry.inc("medintel_http_responses_total", labels={"code": "503"})
    registry.observe("medintel_http_request_seconds", 0.25)
    text = registry.render()
    assert "# TYPE medintel_http_responses_total counter" in text
    assert text.count("# TYPE medintel_http_responses_total counter") == 1  # 指标族只发一次
    assert 'medintel_http_responses_total{code="200"} 2' in text
    assert 'medintel_http_responses_total{code="503"} 1' in text
    assert "# TYPE medintel_http_request_seconds summary" in text
    assert "medintel_http_request_seconds_sum 0.250000" in text

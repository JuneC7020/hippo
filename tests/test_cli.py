from typer.testing import CliRunner

from hippo.cli import app

runner = CliRunner()


def test_run_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    result = runner.invoke(app, ["run", "hello"])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output

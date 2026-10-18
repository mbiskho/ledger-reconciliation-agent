import pytest
from click.testing import CliRunner

from ledger_rec.cli import main


class TestCLI:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Ledger Reconciliation" in result.output

    def test_generate_help(self, runner):
        result = runner.invoke(main, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--transactions" in result.output

    def test_generate_small(self, runner):
        with runner.isolated_filesystem():
            result = runner.invoke(main, [
                "generate", "--transactions", "100", "--break-rate", "0.1",
                "--output", "data/test",
            ])
            assert result.exit_code == 0

    def test_reconcile_help(self, runner):
        result = runner.invoke(main, ["reconcile", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_reconcile_dry_run(self, runner):
        with runner.isolated_filesystem():
            runner.invoke(main, [
                "generate", "--transactions", "50", "--break-rate", "0.2",
                "--output", "data/test",
            ])
            result = runner.invoke(main, [
                "reconcile", "--data-dir", "data/test", "--output", "results",
                "--dry-run",
            ])
            assert result.exit_code == 0

    def test_evaluate_help(self, runner):
        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0

    def test_serve_help(self, runner):
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0

    def test_query_help(self, runner):
        result = runner.invoke(main, ["query", "--help"])
        assert result.exit_code == 0

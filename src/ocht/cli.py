"""Command-line interface entry points for the OChaT TUI application."""
import click

from ocht.core.migration import migrate_to
from ocht.core.version import get_version
from ocht.services.chat import start_chat
from ocht.services.config import export_conf, import_conf, open_conf
from ocht.services.model_manager import list_llm_models, sync_llm_models
from ocht.services.workspace import create_workspace


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context):
    """Modular Python TUI for controlling LLMs via LangChain."""
    if ctx.invoked_subcommand is None:
        start_chat()


@cli.command()
@click.argument("name")
def init(name):
    """Creates a new chat workspace with configuration file and history."""
    create_workspace(name)


@cli.command()
def chat():
    """Starts an interactive chat session based on the current workspace."""
    start_chat()


@cli.command()
def config():
    """Opens the configuration in the default editor."""
    open_conf()


@cli.command()
@click.argument("datei")
def export_config(datei):
    """Exports the current settings as YAML or JSON file."""
    export_conf(datei)


@cli.command()
@click.argument("datei")
def import_config(datei):
    """Imports settings from a YAML or JSON file."""
    import_conf(datei)


@cli.command()
def list_models():
    """Lists available LLM models via LangChain."""
    list_llm_models()


@cli.command()
@click.option(
    '--delete-missing',
    is_flag=True,
    help='Delete models from database that are no longer available in providers',
)
def sync_models(delete_missing):
    """Synchronizes model metadata from external providers into the database."""
    sync_llm_models(delete_missing=delete_missing)


@cli.command()
@click.argument("zielversion")
def migrate(zielversion):
    """Runs Alembic migrations to the specified target version."""
    migrate_to(zielversion)


@cli.command()
def version():
    """Shows the current CLI/package version."""
    version = get_version()
    click.echo(f"OChaT version: {version}")


@cli.command()
@click.argument("command", required=False)
def help(command):
    """Shows detailed help for a command."""
    if command:
        click.echo(f"Help for {command}")
    else:
        click.echo(
            "Available commands: init, chat, config, list-models, sync-models, "
            "export-config, import-config, migrate, version"
        )


if __name__ == "__main__":
    cli()

import click
from ocht.core.workspace import create_workspace
from ocht.core.chat import start_chat
from ocht.core.config import open_config, export_config, import_config
from ocht.core.model_manager import list_models, sync_models
from ocht.core.migration import migrate_to
from ocht.core.version import get_version


@click.group()
def cli():
    """Modulare Python-TUI zur Steuerung von LLMs via LangChain."""
    pass


@cli.command()
@click.argument("name")
def init(name):
    """Erstellt einen neuen Chat-Workspace mit Konfigurationsdatei und Historie."""
    create_workspace(name)


@cli.command()
def chat():
    """Startet eine interaktive Chat-Session basierend auf dem aktuellen Workspace."""
    start_chat()


@cli.command()
def config():
    """Öffnet die Konfiguration im Standard-Editor."""
    open_config()


@cli.command()
@click.argument("datei")
def export_config(datei):
    """Exportiert die aktuellen Einstellungen als YAML- oder JSON-Datei."""
    export_config(datei)


@cli.command()
@click.argument("datei")
def import_config(datei):
    """Importiert Einstellungen aus einer YAML- oder JSON-Datei."""
    import_config(datei)


@cli.command()
def list_models():
    """Listet verfügbare LLM-Modelle über LangChain auf."""
    list_models()


@cli.command()
def sync_models():
    """Synchronisiert die Modell-Metadaten von externen Providern in die Datenbank."""
    sync_models()


@cli.command()
@click.argument("zielversion")
def migrate(zielversion):
    """Führt Alembic-Migrationen auf die angegebene Zielversion aus."""
    migrate_to(zielversion)


@cli.command()
def version():
    """Zeigt die aktuelle CLI-/Paket-Version an."""
    version = get_version()
    click.echo(f"OChaT version: {version}")


@cli.command()
@click.argument("command", required=False)
def help(command):
    """Zeigt detaillierte Hilfe zu einem Command an."""
    if command:
        click.echo(f"Help for {command}")
    else:
        click.echo(
            "Available commands: init, chat, config, list-models, sync-models, export-config, import-config, migrate, version"
        )


if __name__ == "__main__":
    cli()

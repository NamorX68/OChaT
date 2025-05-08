# OChaT

[![PyPI version](https://img.shields.io/pypi/v/ocht.svg)](https://pypi.org/project/ocht/) [![Build Status](https://github.com/dein-username/OChaT/actions/workflows/ci.yml/badge.svg)](https://github.com/dein-username/OChaT/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**OChaT** ist eine modulare Python-TUI-Applikation, die lokal (Ollama) und online (ChatGPT, Claude, Grok) verfügbare Large Language Models über LangChain orchestriert.

---

## 📦 Installation

1. **Repository klonen**

   ```bash
   git clone https://github.com/dein-username/OChaT.git
   cd OChaT
   ```
2. **Virtuelle Umgebung erstellen & aktivieren**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Projekt initialisieren & Abhängigkeiten installieren**

   ```bash
   uv init
   uv sync
   ```

> **Hinweis:** Standardmäßig installiert `uv sync` bereits alle Abhängigkeiten aus `pyproject.toml`, einschließlich:
>
> * `langchain>=0.3.25`
> * `ollama>=0.4.8`
> * `rich>=14.0.0`
> * `sqlmodel>=0.0.8` (für DB-Modelle mit SQLModel)
> * `alembic>=1.10.0` (für Datenbank-Migrationen)

---

## ⚡ Schnelleinstieg

* **`ocht init [PFAD]`**
  Erstellt einen neuen Chat-Workspace mit Konfigurationsdatei und Historie.
* **`ocht chat`**
  Startet eine interaktive Chat-Session basierend auf dem aktuellen Workspace.
* **`ocht config`**
  Öffnet die `ocht.yaml` im Standard-Editor (API-Keys, Modell-Einstellungen).
* **`ocht list-models`**
  Listet verfügbare LLM-Modelle über LangChain auf.

<details>
<summary>Beispiel</summary>

```bash
# Neuen Chat-Workspace "mein-chat" anlegen
ocht init mein-chat
cd mein-chat
# Chat starten
ocht chat
```

</details>

---

## 🗂️ Projektstruktur

```text
OChaT/
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── src/
    └── ocht/
        ├── __init__.py        # Paketdefinition
        ├── cli.py             # CLI-Entrypoints (Click-Commands)
        ├── core/              # Kernlogik & DB-Anbindung
        │   ├── db.py          # SQLite-Verbindung & Session-Setup (SQLModel)
        │   └── models.py      # SQLModel-Klassen: Workspace & Message
        ├── adapters/          # Schnittstellen zu LLM-Providern (LangChain Adapters)
        └── tui/               # Textbasierte UI-Komponenten (Rich)
```

---

## 🤝 Mitwirken

Beiträge sind willkommen!:

1. **Fork** das Repository.
2. **Feature-Branch** erstellen:

   ```bash
   ```

git checkout -b feature/mein-feature

````
3. **Änderungen committen**:  
```bash
git commit -m "feat: Beschreibung meines Features"
````

4. **Push** zum Fork:

   ```bash
   ```

git push origin feature/mein-feature

```
5. **Pull Request** öffnen.

Bitte halte dich an unsere Coding Guidelines und füge Tests hinzu.

---

## 📄 Lizenz

Dieses Projekt steht unter der [MIT License](LICENSE).

```


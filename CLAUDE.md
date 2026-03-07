# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**StartUpResearch** — a Python 3.10 project. The codebase is in early/empty state; source files have not been created yet.

## Environment

- Python 3.10, virtual environment at `.venv/`
- Activate: `source .venv/Scripts/activate` (Windows/bash)
- Install dependencies: `pip install -r requirements.txt` (once a requirements file exists)

## Tooling

- **Formatter**: Black (configured in PyCharm via `.idea/misc.xml`)
- Run Black: `black .`
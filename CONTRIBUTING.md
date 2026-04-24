# Contributing to ISAT

## Development Setup

```bash
git clone https://github.com/SID-Devu/isat.git
cd isat
pip install -e ".[dev,all]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Code Style

We use `ruff` for linting:

```bash
ruff check isat/
ruff format isat/
```

## Adding a New Search Dimension

1. Create `isat/search/your_dimension.py`
2. Define a `YourConfig` dataclass with a `label` field
3. Create a `YourSearchDimension` class with a `candidates()` method
4. Register it in `isat/search/engine.py`
5. Add tests in `tests/`

## Adding a New Integration

1. Create `isat/integrations/your_integration.py`
2. Add CLI flags in `isat/cli.py` if needed
3. Add optional dependencies in `pyproject.toml`

## Pull Request Checklist

- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Linter clean (`ruff check isat/`)
- [ ] New features have tests
- [ ] CLI help text is updated
- [ ] CHANGELOG.md is updated

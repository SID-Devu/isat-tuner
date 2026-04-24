# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in ISAT, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email: sudheerdevu4work@gmail.com
3. Include: description, steps to reproduce, and impact assessment

We will respond within 48 hours and work with you on a fix.

## Scope

ISAT interacts with:
- Local filesystem (model files, config files, output reports)
- GPU drivers (via ORT execution providers)
- SQLite databases (results storage)
- Network (REST API server, if enabled)

## Best Practices

- Do not expose the ISAT API server to the public internet without authentication
- Store API tokens and credentials outside of config files
- Review model files before tuning untrusted models (ONNX models can contain arbitrary ops)

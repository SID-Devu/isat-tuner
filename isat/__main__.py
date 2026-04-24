"""Allow running ISAT as: python3 -m isat

Fallback when 'isat' command is not on PATH after pip install.
"""
import sys
from isat.cli import main

sys.exit(main())

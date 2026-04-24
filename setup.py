"""Custom setup.py to show post-install banner and PATH instructions.

All metadata is in pyproject.toml -- this only adds the post-install hook.
"""
import shutil
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop

POST_INSTALL_MSG = r"""
=====================================================================

  ___ ____    _  _____
 |_ _/ ___|  / \|_   _|
  | |\___ \ / _ \ | |
  | | ___) / ___ \| |
 |___|____/_/   \_\_|

  Inference Stack Auto-Tuner
  by Sudheer Ibrahim Daniel Devu

  55 commands for ONNX inference optimization.

=====================================================================

  INSTALLED SUCCESSFULLY!

  Quick start:

    isat --help              # Show all 55 commands
    isat tune model.onnx     # Auto-tune a model
    isat hwinfo              # Check your hardware
    isat inspect model.onnx  # Analyze a model

  If 'isat' is not found, add ~/.local/bin to PATH:

    export PATH="$HOME/.local/bin:$PATH"

  Or run directly:

    python3 -m isat --help

  Docs: https://github.com/SID-Devu/isat-tuner
  PyPI: https://pypi.org/project/isat-tuner/

=====================================================================
"""


class PostInstall(install):
    def run(self):
        install.run(self)
        print(POST_INSTALL_MSG)


class PostDevelop(develop):
    def run(self):
        develop.run(self)
        print(POST_INSTALL_MSG)


setup(
    cmdclass={
        "install": PostInstall,
        "develop": PostDevelop,
    },
)

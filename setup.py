"""
Minimal setup.py for RLLM (infra / inference / quantization)
"""

import ast
import os
import re
from pathlib import Path

from setuptools import find_packages, setup


def parse_requirements():
    install_requires = []
    dependency_links = []

    with open("./requirements.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("--extra-index-url"):
                _, url = line.split()
                dependency_links.append(url)
            else:
                install_requires.append(line)

    return install_requires, dependency_links


def get_package_version():
    init_path = (
        Path(os.path.dirname(os.path.abspath(__file__)))
        / "src"
        / "RLLM"
        / "__init__.py"
    )
    with open(init_path, "r", encoding="utf-8") as f:
        version_match = re.search(
            r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE
        )
    return ast.literal_eval(version_match.group(1))


def get_long_description():
    readme_path = Path(os.path.dirname(os.path.abspath(__file__))) / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


extras_require = {
    "quantization": [
        "llmcompressor==0.5.1",
        "auto-gptq==0.5.1",
    ],
    "logging": [
        "mlflow",
    ],
    "dev": [
        "pytest",
        "black",
        "isort",
        "ruff",
    ],
}

install_requires, dependency_links = parse_requirements()

setup(
    name="RLLM",
    version=get_package_version(),
    description="RLLM: Reinforcement Learning infrastructure with inference and quantization support",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="ved1beta",
    url="https://github.com/ved1beta/RLLM",
    license="MIT",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=install_requires,
    dependency_links=dependency_links,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "rllm-train=RLLM.main:train",
            "rllm-inference=RLLM.inference.run_inference:main",
            "rllm-quantize=RLLM.inference.quant:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="reinforcement-learning ppo inference quantization pytorch",
)

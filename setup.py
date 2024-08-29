# © 2024 Lucas Faudman.
# Licensed under the MIT License (see LICENSE for details).
# For commercial use, see LICENSE for additional terms.
from setuptools import setup, find_namespace_packages
from setuptools.command.build_ext import build_ext

EXT_MODULES = []
# try:
#     from mypyc.build import mypycify

#     EXT_MODULES.extend(
#         mypycify(
#             [
#                 "src/simplifai/simplifai.py",
#                 "src/simplifai/concurrent_executor.py",
#                 "src/simplifai/decompiler.py",
#                 "src/simplifai/secret_scanner.py",
#             ]
#         )
#     )
# except Exception as e:
#     print(f"Failed to compile with mypyc: {e}")

setup(
    name="simplifai",
    version="0.1.0",
    use_scm_version=True,
    setup_requires=["setuptools>=42", "setuptools_scm>=8", "wheel"],
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Lucas Faudman",
    author_email="lucasfaudman@gmail.com",
    url="https://github.com/LucasFaudman/simplifai.git",
    packages=find_namespace_packages(where="src", exclude=["tests*"]),
    package_dir={"": "src"},
    package_data={
        "": ["LICENSE"],
    },
    include_package_data=True,
    exclude_package_data={"": [".gitignore", ".pre-commit-config.yaml"]},
    install_requires=["pydantic"],
    ext_modules=EXT_MODULES,
    cmdclass={"build_ext": build_ext},
    extras_require={
        "dev": ["pytest"],
        "openai": ["openai"],
        "ollama": ["ollama"],
        "anthropic": ["anthropic"],
    },
    entry_points={
        "console_scripts": [],
    },
    python_requires=">=3.10",
    license="LICENSE",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="",
    project_urls={
        "Homepage": "https://github.com/LucasFaudman/simplifai.git",
        "Repository": "https://github.com/LucasFaudman/simplifai.git",
    },
)

from setuptools import setup

setup(
    name="minictorch",
    version="0.0.1",
    install_requires=["lark-parser"],
    extras_require={
        "develop": []
    },
    entry_points={
        "console_scripts": [
            "minictorch_translator = minictorch.converter:main"
        ]
    }
)

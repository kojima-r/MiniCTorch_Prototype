from setuptools import setup

setup(
    name="minictorch",
    version="0.0.1",
    author="Ryosuke Kojima",
    author_email="kojima.ryosuke.8e@kyoto-u.ac.jp",
    description="controllable deep neural state-space model library",
    long_description="controllable deep neural state-space model library",
    long_description_content_type="text/markdown",
    url="https://github.com/clinfo/ConDeNS",
    packages=setuptools.find_packages(),
    install_requires=["lark-parser"],
    #extras_require={
    #    "develop": []
    #},
    entry_points={
        "console_scripts": [
            "minictorch_translator = minictorch.converter:main"
        ]
    }
)

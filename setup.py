from setuptools import setup, find_packages

setup(
    name="ComfyUI-EasyNodes",
    version="0.2",
    packages=find_packages(where="easy_nodes"),
    install_requires=[
        "torch",
        "pillow",
        "colorama",
        "numpy",
    ],
)

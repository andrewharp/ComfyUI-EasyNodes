from setuptools import setup, find_packages

setup(
    name="ComfyUI-EasyNodes",
    version="0.3",
    description='Makes creating new nodes for ComfyUI a breeze.',
    packages=find_packages(where="easy_nodes"),
    url='https://github.com/andrewharp/ComfyUI-EasyNodes',
    author='Andrew Harp',
    author_email='andrew.harp@gmail.com',
    install_requires=[
        "torch",
        "pillow",
        "colorama",
        "numpy",
    ],
)

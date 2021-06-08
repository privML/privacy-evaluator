from setuptools import find_packages
from setuptools import setup

setup(
    name="privacy-evaluator",
    version="0.1",
    description="Tool to assess ML model's levels of privacy.",
    url="https://github.com/privML/privacy-evaluator",
    license="MIT",
    packages=find_packages(include=["privacy_evaluator*"]),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "torchvision",
        "tensorflow",
        "adversarial-robustness-toolbox[pytorch,tensorflow]==1.6.1",
    ],
    include_package_data=True,
)

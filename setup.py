from setuptools import find_packages
from setuptools import setup

setup(
    name="privacy-evaluator",
    version="0.1",
    description="Tool to assess ML model's levels of privacy.",
    url="https://github.com/privML/privacy-evaluator",
    license="MIT",
    packages=list(filter(lambda s: s.startswith("privacy_evaluator"), find_packages())),
    install_requires=["adversarial-robustness-toolbox[pytorch,tensorflow]==1.6.1"],
)

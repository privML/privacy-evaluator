from setuptools import setup

setup(
    name="privacy-evaluator",
    version="0.1",
    description="Tool to assess ML model's levels of privacy.",
    url="https://github.com/privML/privacy-evaluator",
    license="MIT",
    packages=["privacy_evaluator"],
    install_requires=["adversarial-robustness-toolbox[pytorch,tensorflow]==1.6.1"],
)

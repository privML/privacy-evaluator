from setuptools import find_packages
from setuptools import setup

install_requires = [
    "adversarial-robustness-toolbox[pytorch,tensorflow]==1.7.1",
    "matplotlib",
    "numpy",
    "pandas",
    "torch",
    "tensorflow",
    "torchvision",
    "pillow==8.2.0",
]

development_requires = [
    "black",
    "googledrivedownloader",
    "pytest",
]

setup(
    name="privacy-evaluator",
    version="0.1",
    description="Tool to assess ML model's levels of privacy.",
    url="https://github.com/privML/privacy-evaluator",
    license="MIT",
    packages=find_packages(include=["privacy_evaluator*"]),
    install_requires=install_requires,
    extras_require={"development": development_requires},
    include_package_data=True,
)

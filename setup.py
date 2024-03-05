from setuptools import setup, find_packages

from setuptools import setup, find_packages

setup(
    name="interactda",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "pydantic == 2.6.3",
        "pandas == 2.2.1",
        "scikit-learn == 1.4.1.post1",
        "imbalanced-learn == 0.12.0",
        "category-encoders == 2.6.3",
        "numpy == 1.26.4",
        # List your dependencies here
    ],
)

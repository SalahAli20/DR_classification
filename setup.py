from setuptools import setup, find_packages

# Project metadata
setup(
    name="diabetic_retinopathy_classifier",
    version="0.1.0",
    description="A machine learning project for classifying diabetic retinopathy",
    author="Salah Ali",
    author_email="salahmflh@gmail.com",
    packages=find_packages(exclude=["tests", "*.ipynb"]),  # Exclude tests and notebooks
    install_requires=[
        "torch",
        "torchvision",
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
        "seaborn",
        "matplotlib",
    ], 
)

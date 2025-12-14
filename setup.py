from setuptools import find_packages, setup  # type: ignore
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """Return list of requirements from requirements.txt"""
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and req.strip() != "-e ."]
    return requirements

setup(
    name="medical_chatbot",
    version="0.0.1",
    author="mukesh",
    author_email="www.worldwide.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
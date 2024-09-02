import os
import re

from setuptools import find_packages, setup


def get_version():
    with open(os.path.join("fastie", "extras", "env.py"), "r", encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("VERSION")
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


def main():
    setup(
        name="fastie",
        version=get_version(),
        author="xusenlin",
        author_email="15797687258" "@" "163.com",
        description="Easy-to-use Information Extraction Framework",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords=["NLP", "Information Extraction", "NER", "Text Classification", "Event Extraction", "transformer", "pytorch", "deep learning"],
        license="Apache 2.0 License",
        url="https://github.com/xusenlinzy/FastIE",
        packages=find_packages(),
        python_requires=">=3.8.0",
        install_requires=get_requires(),
        entry_points={"console_scripts": ["fastie-cli = fastie.cli:main"]},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main()

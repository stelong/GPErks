#!/usr/bin/env python3
import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))
home_page = "https://github.com/stelong/GPErks"


def read_requirements(file_name):
    reqs = []
    with open(os.path.join(here, file_name)) as in_f:
        for line in in_f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            reqs.append(line)
    return reqs


with open(os.path.join(here, "README.md")) as f:
    readme = f.read()


setup(
    name="GPErks",
    version="0.1.2",
    url=home_page,
    author="Stefano Longobardi, Gianvito Taneburgo",
    author_email="stefano.longobardi.8@gmail.com, taneburgo+shadowtemplate@gmail.com",
    license="MIT",
    description="A Python library to (bene)fit Gaussian Process Emulators.",
    long_description=readme,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests"]),
    install_requires=read_requirements(
        os.path.join("requirements", "requirements.txt")
    ),
    project_urls={
        "Bug Tracker": os.path.join(home_page, "issues"),
        "Source Code": home_page,
    },
    extras_require={
        "dev": read_requirements(
            os.path.join("requirements", "requirements-dev.txt")
        ),
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.repo"]
    },
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)

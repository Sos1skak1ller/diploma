#!/usr/bin/env python3
"""
Setup script для системы анализа спутниковых снимков
"""

from setuptools import setup, find_packages
import os

# Читаем README файл
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Система анализа спутниковых снимков"

# Читаем requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="satellite-image-analyzer",
    version="2.0.0",
    author="Diplom Project",
    author_email="diplom@example.com",
    description="Система анализа спутниковых снимков с тремя алгоритмами",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/diplom/satellite-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'satellite-analyzer=launch_integrated:main',
            'satellite-demo=demo_interface:main',
            'satellite-test=test_interface:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.md', '*.txt', '*.ui'],
    },
    keywords="satellite, image processing, computer vision, PyQt5, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/diplom/satellite-analyzer/issues",
        "Source": "https://github.com/diplom/satellite-analyzer",
        "Documentation": "https://github.com/diplom/satellite-analyzer/blob/main/README.md",
    },
)

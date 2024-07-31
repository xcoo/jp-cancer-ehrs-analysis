import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="emr-analysis",
    version="0.0.1",
    author="Xcoo, Inc.",
    author_email="developer@xcoo.jp",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    install_requires=[
        "gensim==4.2.0",
        "mojimoji==0.0.12",
        "numpy==1.23.4",
        "pandas==1.4.4",
        "python-dateutil==2.8.2",
        "pyyaml==6.0",
        "scikit-learn==1.5.0",
        "scipy==1.12.0",
        "stanza==1.4.2",
        "torch==2.4.0",
    ],
    extras_require={
        "dev": [
            "ipywidgets==8.0.1",
            "jupyterlab==3.4.3",
            "tabulate[widechars]==0.9.0",
        ],
    },
    python_requires='>=3.9',
)

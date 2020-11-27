import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="vog_optimizer",
    version="0.0.1",
    author="Matthew Conniff",
    author_email="lol@nah.com",
    description="A package for using VoG in an optimizer",
    long_description="A long description",
    long_description_content_type="text/markdown",
    url="https://github.com/mttcnnff/vog-optimizer",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)

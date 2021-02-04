import setuptools

setuptools.setup(
    name="mwutils",  # Replace with your own username
    version="0.4.1-fix",
    author="mw123",
    description="use in mw",
    url="https://github.com/mkkb473/mwutils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["PyJWT", "requests", "pynvml", "psutil"],
    python_requires='>=3.6',
)

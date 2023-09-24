import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name="liquidity",
    version="1.0.0",
    author="Anastasia Bugeenko",
    author_email="anabugaenko@gmail.com",
    license="MIT",
    description="Fit and calibrate nonlinear repsonse functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anabugaenko/liquidity",
    install_requires=[
        "scipy",
        "numpy",
        "pandas",
        "typing",
        "statsmodels",
        "matplotlib",
        "powerlaw",
        "hurst_exponent",
        "powerlaw_function",
    ],
    keywords="liquidity price-impact order-flow-imbalance",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
    ],
)

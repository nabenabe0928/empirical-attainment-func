import setuptools


requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        requirements.append(line.strip())


setuptools.setup(
    name="empirical-attainment-func",
    version="0.4.3",
    author="nabenabe0928",
    author_email="shuhei.watanabe.utokyo@gmail.com",
    url="https://github.com/nabenabe0928/empirical-attainment-func",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    platforms=["Linux"],
    install_requires=requirements,
    include_package_data=True,
)

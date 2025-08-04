import pathlib

from setuptools import find_packages, setup


def get_version():
    """Gets the vmas version."""
    path = CWD / "src" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


CWD = pathlib.Path(__file__).absolute().parent


setup(
    name="rl_x_coverage",
    version="0.1.0",
    description="Reinforcement Learning for Coverage Control",
    long_description=open("README.md").read(),
    url="https://github.com/Giovannibriglia/rl_x_coverage",
    license="GPLv3",
    author="Giovanni Briglia",
    author_email="giovanni.briglia@phd.unipi.it",
    packages=find_packages(),
    install_requires=["torch"],
    include_package_data=True,
)

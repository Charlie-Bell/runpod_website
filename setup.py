from setuptools import setup, find_packages


# Modified from https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/setup.py
def parse_requirements():
    _install_requires = []
    _dependency_links = []
    with open("./requirements.txt", encoding="utf-8") as requirements_file:
        lines = [r.strip() for r in requirements_file.readlines()]
        for line in lines:
            if line.startswith("--extra-index-url") or line.startswith("--index-url"):
                # Handle custom index URLs
                _, url = line.split()
                _dependency_links.append(url)
            elif line and line[0] != "#":
                # Handle standard packages
                _install_requires.append(line)

    return _install_requires, _dependency_links


install_requires, dependency_links = parse_requirements()

setup(
    name="runpod_website",
    version="0.0.1",
    packages=find_packages(),
    install_requires=install_requires,
    dependency_links=dependency_links,
    entry_points={
        'console_scripts': [
            'runpod_website=runpod_website.app:main',
        ],
    },
)
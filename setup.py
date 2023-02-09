from setuptools import setup, find_packages
from os import path as p


DISTNAME = 'autogater'
DESCRIPTION = 'a tool for gating flow cytometry data using only FSC/SSC channels'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'netrias'
MAINTAINER_EMAIL = 'meslami@netrias.com'

def read(filename, parent=None):
    parent = (parent or __file__)

    try:
        with open(p.join(p.dirname(parent), filename)) as f:
            return f.read()
    except IOError:
        return ''

def parse_requirements(filename, parent=None):
    parent = (parent or __file__)
    filepath = p.join(p.dirname(parent), filename)
    content = read(filename, parent)

    for line_number, line in enumerate(content.splitlines(), 1):
        candidate = line.strip()

        if candidate.startswith('-r'):
            for item in parse_requirements(candidate[2:].strip(), filepath):
                yield item
        else:
            yield candidate

setup(
    name=DISTNAME,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    packages=['autogater_src'] + ['autogater_src/' + s for s in find_packages('autogater_src')],
    include_package_data=True,
    install_requires=[list(parse_requirements('requirements.txt'))]
)

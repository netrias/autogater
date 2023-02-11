
# Autogater

## Why use Autogater?
AutoGater is a weakly supervised deep learning model that can separate healthy populations from unhealthy and dead populations using only light-scatter channels. In addition, AutoGater harmonizes different measurements of dead cells such as Sytox and CFUs.
A link to the paper on biorxiv is here: [Link](https://www.biorxiv.org/content/10.1101/2022.12.07.519491v1)

## Using the Test Harness

Python 3 should be used when using autogater.

### Installation
1. Clone this repository into the environment of your choice (directory, conda env, virtualenv, etc)
2. Using command-line, navigate to the directory in which you cloned this repo (not inside the repo itself).
3. Run `pip3 install test-harness` or `pip3 install -e autogater` .
This will install the `autogater` package and make it visible to all other repositories/projects
you have in the current environment. The `-e` option stands for "editable". This will install the package
in a way where any local changes to the package will automatically be reflected in your environment.
See [this link](https://stackoverflow.com/questions/41535915/python-pip-install-from-local-dir/41536128)
for more details.
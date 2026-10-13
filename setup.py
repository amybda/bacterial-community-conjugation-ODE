from setuptools import setup, find_packages

def get_readme():
    """Load README.md text for use as description."""
    with open('README.md') as f:
        return f.read()

setup(name='bacterial-community-conjugation-ODE',
      version='0.1',
      description='A package to model the spread of a plasmid by conjugation within a bacterial community perturbed by antibiotics',
      long_description=get_readme(),
      author='AnMei Daniels',
      url='https://github.com/amybda/bacterial-community-conjugation-ODE',
      # Packages to include
      packages=find_packages(include=('ODE-model')),
      include_package_data=True,
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'matplotlib',
                        'plotly',
                        'shiny',
                        'shinywidgets',
                        'faicons',
                        'htmltools'],
      extras_require={'docs': ['sphinx>=1.5, !=1.7.3',  # Sphinx for doc generation (v.1.7.3 has a bug)
                               'sphinx_rtd_theme']},
      license='GPLv3')
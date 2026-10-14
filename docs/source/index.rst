.. ODE-based modelling of conjugation in a bacterial community in response to drug perturbation documentation master file, created by
   sphinx-quickstart on Tue Oct 13 15:12:55 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ODE-based modelling of conjugation in a bacterial community in response to drug perturbation documentation
==========================================================================================================
The experiment we are working on is to determining the effect of various drugs on the gut microbiome, which then may enable or prevent the spread of antibiotic resistance genes (ARGs). The current theory is that if drugs such as antibiotics successfully peturb the microbiome, if the donor and recipient are resistant, they can overgrow, increasing the odds they would come in contact with each other, allowing conjugation to occur. We also want to understand the influence of spatial structure within the gut - where different species are localised in the gut microbiome - on this phenomenon. While we are currently conducting *in vitro* and *in vivo* mouse experiments, we would also like to develop a model which explores the dynamics of this.

There are multiple different modelling approaches for studying the microbiome (`Kumar, 2019 <https://www.nature.com/articles/s41564-019-0491-9#Sec11>`_). The two most relevant methods are ordinary differential equation (ODE)-based modelling, and agent-based modelling (ABM). Here we will focus on ODE-based modelling. ODE-based modelling has been used to model monocultures or simple co-culture systems (minimal species), using rate equations to measure the change in different variables in time - such as species growth or plasmid transfer. The rate equations can be used to create mass balances (a set of ODEs that can be solved using standard computational solvers). These ODEs allow the simulation of dynamic interactions between species, but they often have to be simplified to reduce the number of variables and parameters.

Details on the mathematics behind the model can be found in the walkthroughs of the code or on our `GitHub Wiki <https://github.com/amybda/bacterial-community-conjugation-ODE/wiki>`_.


How to Use This Code
-------------------------
All the code for this model is contained in the folder ``ODEconjugation``. . Within this folder is::

   |-- data
   |-- figures
   |-- app.py
   |-- modelling-ODE.ipynb
   |-- ode-model.py

Run Online in Browser
^^^^^^^^^^^^^^^^^^^^^^^
The simplest method to interact with the code is to run it in our webapp at our `GitHub pages <https://amybda.github.io/bacterial-community-conjugation-ODE/>`_ The app contains full details on the mathematical background, a walkthrough of each of the parameters and how it affects the model outputs, as well as an option to run a simulation with fully customised parameters.

Run Local App on Your Computer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want to explore the code on your computer, with the same user-friendly GUI and walkthrough options as the browser app, you can run the app locally.

1. Download this repository to your computer
2. Install the requirements in your Python environment of choice using ``pip install ./`` from within the repository folder
3. In your environment, using an application like VScode that is able to support Shiny apps, run ``app.py``

Run Scripts Locally on Your Computer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want full freedom to run and explore the code contained in this repository, you can run the code locally on your computer.

1. Download this repository to your computer
2. Install the requirements in your Python environment of choice using ``pip install ./`` from within the repository folder
3. A walkthrough of how the code works is located in the Jupyter Notebook ``modelling-ODE.ipynb``, with details on the background and step-by-step explanations of all the parameters
4. Otherwise, you can create your own Python script to run the code, which is documented below

The Code
----------
The class containing the functions to run the model is in ``ODE_model.py`` as a class, with details on how the functions within it work below.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ODEmodel
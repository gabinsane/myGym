# Robothon 2021 Project - Robots and Artificial Intelligence 

This repository contains our submission code for Robothon 2021. It was forked from myGym, a modular toolkit for robotic RL simulator, developed by the same authors as this project. To read more about myGym, please visit [the original repository](https://github.com/incognite-lab/myGym).


## Supported systems

Ubuntu 18.04, 20.04
Python 3
GPU acceleration strongly reccomended



## Installation

Clone the repository:

`git clone https://github.com/gabinsane/myGym.git`

`cd mygym`

We recommend to create conda environment:

`conda env create -f environment.yml`

`conda activate mygym`

Install mygym:

`python setup.py develop`


## Visualization

You can visualize the virtual gym prior to the training.

`python test.py`


## Basic Training

Run the default training without specifying parameters:

`python train.py`

The training will start with gui window and standstill visualization. Wait until the first evaluation after 10000 steps to check the progress: 



## Authors


![alt text](myGym/images/incognitelogo.png "test_work")


[Incognite lab - CIIRC CTU](https://incognite.ciirc.cvut.cz) 

Team members:

[Megi Mejdrechova](https://www.linkedin.com/in/megi-mejdrechova)

[Gabriela Sejnova](https://kognice.wixsite.com/vavrecka)

[Nikita Sokovnin](https://kognice.wixsite.com/vavrecka)

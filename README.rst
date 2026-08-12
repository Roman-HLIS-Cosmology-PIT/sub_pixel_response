|badge1| |badge2|

.. |badge1| image:: https://codecov.io/gh/Roman-HLIS-Cosmology-PIT/sub_pixel_response/graph/badge.svg

.. |badge2| image:: https://github.com/Roman-HLIS-Cosmology-PIT/sub_pixel_response/actions/workflows/smoke-test.yml/badge.svg

Project Description
###################

Goal
====

- Produce a calibration file with the moments of the response of every pixel in the focal plane. 
- If necessary, we want to include the calibration file in the Level 2 to 2.1 processing pipeline. 

Methods
=======

- Making image simulations of the Galactic Bulge Time Domain Survey and calibration fields in Section 2.5 of WFI On-Orbit Calibration plan.
- Creating a pipeline to simultaneously solve for 1.8 billion pixel moments from a set of images.
- Validate that this pipeline works using our simulated images. 
- Coordinating with the Calibration Working Group and SOC representatives on optimizing the observations and dithering plan, as well as analysis methods.

Running the code
################

*work in progress*

``imageSim.py`` contains the code to make oversampled/high resolution Roman images. An example configuration file is provided in ``example_test.yaml``.

To run `imageSim.py`, use the following command in this repository:

.. code-block:: bash

    sbatch makeimage.sh

with the correct edits to ``makeimage.sh`` so that you are running on the correct account. 

Post processing functions live in ``processImage.py``. In your local repository, you can import functions 

.. code-block:: python

    from processImage import compute_pixel_weights

and call it as needed. (We are working on making this installable.)

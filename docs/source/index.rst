.. d documentation master file, created by
   sphinx-quickstart on Wed Jul 13 14:44:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the GPM-API documentation!
======================================

Here, you'll find everything you need to navigate through the data archive provided by the **Global Precipitation Measurement (GPM)** mission.

The GPM data archive currently includes satellite data records that extend back to 1987.
This extensive archive is the result of contributions from **2** spaceborne **radars** and a fleet of **35 passive microwave (PMW)** sensors
that forms the so-called GPM constellation.

The data are organized into various product levels, encompassing raw and calibrated observations (Level 1),
intermediate geophysical retrieval products (Level 2), and integrated datasets from multiple satellites (Level 3).

The **GPM-API** is a Python package designed to make your life easier, whether you aim to download some data, search for specific files, or jump into scientific analysis.
Our goal is to empower you to focus more on what you can discover and create with the data, rather than getting bogged down by the process of handling it.

With our software, you can:

1. **Download GPM Products**: Quickly search and download the data you need.

2. **Locate Files Easily**: Find the downloaded files on your local storage without digging through folders.

3. **Get Analysis-Ready Data**: Load your data into xarray, ready to be analyzed.

4. **Process Data Efficiently**: Dask enables you to perform fast and efficient lazy distributed processing.

5. **Visualize Like a Pro**: Create beautiful visualizations tailored to the unique product characteristics.

6. **Label Precipitation Events**: Identify specific precipitation events within the entire data archive.

7. **Extract Spatio-Temporal Patches**: Create your own database to develop new algorithms and discover something new.

8. **Geographically Aggregate Data**: Collect sensor measurements into a grid or polygons to perform statistical analysis.

9. **Access Scientific Retrievals**: Access community-based retrievals to enhance your analysis.

10. **Share Your Work**: Share your discoveries and findings with the community, making it easier for others to access and build upon collective knowledge.


We're excited to see how you'll use this tool to push the boundaries of what's possible, all while making your workflow smoother, enjoyable, and more productive.


**Ready to jump in?**

Consider joining our `Slack Workspace <https://join.slack.com/t/gpmapi/shared_invite/zt-28vkxzjs1-~cIYci2o3G0qEEoQJVMQRg>`__ to say hi or ask questions.
It's a great place to connect with others and get support.

Let's get started and unlock the full potential of the GPM data archive together!

.. warning::

   The GPM-API is still in development. Feel free to try it out and to report issues or to suggest changes.


Documentation
=============

.. toctree::
   :maxdepth: 2

   00_introduction
   02_installation
   03_quickstart
   04_tutorials
   05_advanced_tools
   06_theory
   06_contributors_guidelines
   07_maintainers_guidelines
   08_authors


API Reference
===============

.. toctree::
   :maxdepth: 1

   API <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

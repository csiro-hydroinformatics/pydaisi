pydaisi
=======

Python Data Assimilation Informed model Structure Improvement (PyDAISI): Python
code to run the PyDAISI method applied to the GR2M monthly rainfall-runoff
model.

What is pydaisi?
~~~~~~~~~~~~~~~~



Installation
~~~~~~~~~~~~
- Install the `hydrodiy <https://github.com/csiro-hydroinformatics/hydrodiy>`__ package.
- Install the `pygme <https://github.com/csiro-hydroinformatics/pygme>`__ package.
- Download the `source code <https://github.com/csiro-hydroinformatics/pydaisi>`__ and run ``python setup.py install``.

Basic use
~~~~~~~~~

To access the data:
   .. code:: 
       from pydaisi import daisi_data
       
       # Select a site id 
       # For example the Jamieson River at Gerrang Bridge,
       # (site ID 405218)
       siteid = 405218
       monthly_data = dais_data.get_data(siteid)

       print(monthly_data) 
       # This command shows:
       #                 Rain      Evap      Qobs
       # 
       # 1970-07-01  135.9429   33.5447  186.7124
       # 1970-08-01  253.6129   46.9463  207.2174
       # 1970-09-01   87.7752   69.9889  119.4157
       # 1970-10-01   51.4173  114.8741   56.1140
       # 1970-11-01  110.2354  145.4254   27.9042
       # ...              ...       ...       ...
       # 2019-02-01   34.0550  146.5809    3.1161
       # 2019-03-01   62.1733  117.4158    2.5748
       # 2019-04-01   27.0699   75.8672    2.8548
       # 2019-05-01  138.3442   44.7638    8.6404
       # 2019-06-01  141.6448   32.5938   43.1690

To run DAISI applied to GR2M:
TODO

License
~~~~~~~~~

The source code and documentation of the pydaisi package is licensed under the
`BSD license <https://opensource.org/license/bsd-3-clause/>`__.


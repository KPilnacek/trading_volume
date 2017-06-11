# Modelling and predicting trading volume

A simple modelling and predicting of day to day S&P 500 trading volume.

## How to run

So far this code has been tested only under Linux.

I recommend run this code using a virtual environment, which can be created as follows:

    $ virtualenv -p /usr/bin/python3.6 env/finance
    $ source ./env/finance/bin/activate
    $ pip install -r requirements.txt

Then, in the `virtualenv`, use the `main.py` file:

    (finance)$ ./src/main.py

## Report

The technical report is written in LaTeX in Czech.

In order to generate the report, one needs to generate all the plots.
This can be done using the following commands:

    $ ./src/main.py --save-plots
    $ python3.6 ./src/preprocessing.py
    $ python3.6 ./src/model/statespace_models.py
    
The resulting plots will be then located in `./plots` directory.

### Note
Python >= 3.6.0 required.

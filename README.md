# 659 project


[![Build Status](https://travis-ci.org/dchui1/659-project.svg?branch=master)](https://travis-ci.org/dchui1/659-project)


## Setup
Setup can be done with globally installed packages (default way to setup a python project) or can be setup in a virtual environment (cleaner / easier to maintain, but more up-front work).

### No virtual environment
To setup in the standard way:
```bash
pip3 install tensorflow tensorflow_probability numpy matplotlib
```

### Virtual environment (the right way)
To setup in a virtual environment:
```bash
pip3 install virtualenv
cd Path/to/project/directory
virtualenv -p python3 659env
```

To "login" to the virtualenv before running the code:
```bash
. 659env/bin/activate
```

Finally, install the packages:
```bash
pip install tensorflow tensorflow_probability numpy matplotlib
```

## running code
To run the code, first activate the virtual environment (if you are using one).
Then run:
```bash
python q_learning.py
```

# Designing Search Spaces for Unbounded Bayesian Optimization via Transfer Learning (ECML PKDD 2025)

## Installation
The easiest way to install the required dependencies is to use Anaconda on Window (code was tested on Windows 11). In this directory, run
```
conda env create -f environment.yaml
```

The environment can then be used with
```
conda activate transfer_search_space
```
Alternatively, the dependencies can be installed manually using ```environment.yaml``` as reference.


## Experiments
The experiments contain in the file [test_proposed.py](https://github.com/Fsoft-AIC/BO-transfer-search-space/blob/main/test_proposed.py)

#### Example: Running Scenario 1 with Ackley function

```
python test_proposed.py --test_func ackley_s1 --restart 20 --max_evals 50 --save_result False
```



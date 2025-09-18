# Optimization Framework

## Citing
TODO:



## Installation (user)

To install the python package, please run:
```bash
python install -e .
```

Then, you should have a package called OptimizationFramework, which can be imported via:

```python
import OptimizationFramework as OF
```

Have fun!

## Installation (developer)

### Quick Install (Arch Linux)
```bash
bash install.sh --IUseArchBtw
```

### Quick Install (Ubuntu)
```bash
bash install.sh
```

If you have used any of the automated installation methods, you can activate the virutal environment via:
```bash
source env.sh
```

### Manual Installation (All Systems)
Ensure you have **Python 3.12** installed, then:

```bash
git clone git@github.com:mengsig/OptimizationFramework.git
cd OptimizationFramework
python -m venv opt_venv
source opt_venv/bin/activate
pip install -r requirements.txt
```

## Usage

To run the optimization pipeline, execute the following command - i.e. all the optimization methods:

```bash
python src/basic_example.py
```

I have also included a non-trivial example of how to implement one's own loss function and use this repo
to directly optimize that landscape. That can be seen via the call:

```bash
python src/circle_example.py
```


## Structure
The repository can be seen hereunder. You will find all of the optimizer methods in *src/optimization/optimizers.py*,
and you can see how to invoke them by looking at the examples in *src/<...>_example.py*.
```tree
├── env.sh
├── examples
│   ├── basic_example.py
│   └── circle_example.py
├── figs
├── install.sh
├── macinstall.sh
├── pyproject.toml
├── README.md
├── requirements.txt
└── src
    ├── OF
    │   └── __init__.py
    └── OptimizationFramework
        ├── __init__.py
        ├── lossFunctions.py
        ├── optimizers.py
        └── plottingUtils.py```

## Examples

### Result from *src/circle_example.py*
Here we use a difficult optimization problem to showcase the generalizeability of this repo. The objective is to optimally cover as much weight of a 2D weighted matrix (image) as possible. Here is the saved image from the script:

![](figs/circle_winner.png)

Where the terminal output was as follows:

```bash
[pgo         ] [##############################] 100%  best=-566.805797
[pgo         ] score=566.805797  nfev= 10000  time=  2.37s  centers=[(27, 40), (8, 54), (28, 10), (54, 28), (46, 51), (5, 23), (3, 4)]
[cma         ] [##############################] 100%  best=-562.281486
[cma         ] score=562.281486  nfev= 10000  time=  2.39s  centers=[(37, 50), (37, 9), (22, 34), (54, 29), (5, 46), (8, 58), (25, 19)]
[de          ] [##############################] 100%  best=-534.572375
[de          ] score=534.572375  nfev=  9870  time=  0.83s  centers=[(26, 46), (53, 29), (8, 52), (22, 15), (35, 9), (44, 58), (14, 8)]
[ga          ] [##############################] 100%  best=-547.649116
[ga          ] score=547.649116  nfev= 10000  time=  1.39s  centers=[(26, 11), (9, 43), (52, 28), (44, 54), (8, 58), (30, 43), (14, 22)]
[abc         ] [##############################] 100%  best=-562.588330
[abc         ] score=562.588330  nfev= 10000  time=  1.23s  centers=[(27, 40), (8, 54), (52, 28), (44, 54), (34, 6), (26, 17), (30, 58)]
[dual_anneal ] [##############################] 100%  best=-554.948243
[dual_anneal ] score=554.948243  nfev= 10000  time=  1.06s  centers=[(28, 39), (9, 8), (8, 47), (36, 7), (42, 57), (25, 16), (5, 60)]
[nelder_mead ] [##############################] 100%  best=-420.580967
[nelder_mead ] score=420.580967  nfev=   670  time=  0.05s  centers=[(30, 18), (47, 9), (9, 15), (26, 8), (1, 47), (60, 7), (29, 38)]
[powell      ] [##############################] 100%  best=-533.841087
[powell      ] score=531.619334  nfev=  1605  time=  0.13s  centers=[(49, 29), (35, 47), (37, 9), (18, 39), (28, 30), (5, 48), (24, 19)]
[pso         ] [##############################] 100%  best=-541.040252
[pso         ] score=541.040252  nfev=  9960  time=  0.68s  centers=[(28, 40), (8, 44), (53, 29), (44, 53), (6, 58), (55, 46), (16, 11)]
[bho          ] [##############################] Best Loss: -539.707374
[bho         ] score=539.707374  nfev= 10000  time=  2.01s  centers=[(10, 50), (38, 53), (30, 9), (27, 38), (47, 10), (22, 23), (39, 33)]

=== Scoreboard (higher is better) ===
 1. pgo           score=566.805797  nfev= 10000  time=  2.37s
 2. abc           score=562.588330  nfev= 10000  time=  1.23s
 3. cma           score=562.281486  nfev= 10000  time=  2.39s
 4. dual_anneal   score=554.948243  nfev= 10000  time=  1.06s
 5. ga            score=547.649116  nfev= 10000  time=  1.39s
 6. pso           score=541.040252  nfev=  9960  time=  0.68s
 7. bho           score=539.707374  nfev= 10000  time=  2.01s
 8. de            score=534.572375  nfev=  9870  time=  0.83s
 9. powell        score=531.619334  nfev=  1605  time=  0.13s
10. nelder_mead   score=420.580967  nfev=   670  time=  0.05s

Winner: pgo  score=566.805797
```

### Result from *src/simple_example.py*
Here we optimize the simple rosenbrock function, to showcase the simplest case. For dimension of 20, the output of the *src/basic_example.py* was as follows:

Where the terminal output was as follows:

```bash
[cma         ] [##############################] 100%  best=0.9949593
[cma         ] fx=0.994959  nfev=  1000  time=  0.66s
[de          ] [##############################] 100%  best=0.0023312
[de          ] fx=0.00233121  nfev=   990  time=  0.03s
[ga          ] [##############################] 100%  best=0.206952
[ga          ] fx=0.206952  nfev=  1000  time=  0.07s
[abc         ] [##############################] 100%  best=0.000066
[abc         ] fx=6.57672e-05  nfev=  1000  time=  0.05s
[dual_anneal ] [##############################] 100%  best=0.0000008
[dual_anneal ] fx=7.10543e-15  nfev=  1000  time=  0.06s
[nelder_mead ] [##############################] 100%  best=16.914203
[nelder_mead ] fx=16.9142  nfev=   157  time=  0.01s
[powell      ] [##############################] 100%  best=0.0000006
[powell      ] fx=0  nfev=    80  time=  0.00s
[pso         ] [##############################] 100%  best=0.994962
[pso         ] fx=0.994962  nfev=  1000  time=  0.02s
[pgo         ] [##############################] 100%  best=0.0116657
[pgo         ] fx=0.0116648  nfev=  1000  time=  0.16s
[bho          ] [##############################] Best Loss: 0.079601
[bho         ] fx=0.0796007  nfev=  1000  time=  0.10s
```

## How to use:
Simply write your own loss function with the correct input arguments, and simply pass it to one of the optimizers as done in the examples, and enjoy! You can also use the examples as a baseline to test your loss function for all the different methods with various ```Budgets``` to see which version works best for your use case!

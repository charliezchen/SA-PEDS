
## Introduction
This code is for the project **On the global minimum convergence of non-convex deterministic functions via Stochastic Approximation**, as part of the NYU AM-SURE program.


## Draw the graph in the slides

To draw the graph in the Experiment section, run the following lines. Change 2 to 10 if you want to see the case `m=10`.
```bash
# restart
python3 optimize_peds.py \
    --test_function Ackley \
    --m 2 --N 32 --alpha 0 --alpha-inc 1e-3 --lower 5 --upper 10 --independent \
    --class torch.optim.SGD --debug 

# PEDS
python3 optimize_peds.py \
    --test_function Ackley \
    --m 2 --N 32 --alpha 0 --alpha-inc 1e-3 --init_noise 5  --lower 5 --upper 10  \
    --class torch.optim.SGD --debug 

# SA-PEDS
# alpha-inc should be at least 1e-2
python3 optimize_peds.py \
    --test_function Ackley \
    --m 2 --N 32 --alpha 0 --alpha-inc 1e-2 --init_noise 10  --lower 5 --upper 10  --rv \
    --class torch.optim.Adam --debug
```

The grid experiments results can be get from running
```bash
# This runs the experiments
# for t=Ackley, m=1,2,4,...,128, optim=Adam,SGD, rv=--rv, --independent, ""(nothing)
python3 -u optimize_peds.py \
    --test_function $t \
    --m $m --N \$N --alpha 0 --alpha-inc 0.01 $rv \
    --class torch.optim.$optim 

# To draw the grid in the slides
# You may want to change certain settings in the codes
# Search for "CHANGE ME"
python3 grid_plot.py
```

To draw the graph in Appendix A
```bash
# SA-PEDS with VGD
python3 optimize_peds.py \
    --test_function Ackley \
    --m 2 --N 32 --alpha 0 --alpha-inc 1e-2 --init_noise 10  --lower 5 --upper 10  --rv \
    --class torch.optim.SGD --debug

# PEDS with Adam
python3 optimize_peds.py \
    --test_function Ackley \
    --m 2 --N 32 --alpha 0 --alpha-inc 1e-4 --init_noise 5  --lower 5 --upper 10  \
    --class torch.optim.Adam --debug 
```
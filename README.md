# Safe Domains of Attraction for Discrete-Time Autonomous Nonlinear  Systems: Characterization and Verifiable Neural Network Estimation

# Code

## Installation

Create a conda environment and install the dependencies except those for verification:
```bash
conda create --name lnc python=3.11
conda activate lnc
pip install -r requirements.txt
```

We use [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA.git) and [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN.git) for verification. To install both of them, run:
```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
(cd alpha-beta-CROWN/auto_LiRPA && pip install -e .)
(cd alpha-beta-CROWN/complete_verifier && pip install -r requirements.txt)
```

To set up the path:
```
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/alpha-beta-CROWN:$(pwd)/alpha-beta-CROWN/complete_verifier"
```

## Verification

We have provided the models we trained and the specifications for verification we generated.
To run verification with our **pre-trained models** and specifications:

```bash
cd verification
export CONFIG_PATH=$(pwd)
cd complete_verifier

# Run the following for each of the systems
python abcrown.py --config $CONFIG_PATH/van_der_pol.yaml
python abcrown.py --config $CONFIG_PATH/two_machine.yaml
python abcrown.py --config $CONFIG_PATH/power.yaml

```

The verification will output a summary of results. For example, here are the
results we obtained on Pendulum Output Feedback using `pendulum_output_feedback_lyapunov_in_levelset.yaml`:
```
############# Summary #############
Final verified acc: 100.0% (total 8 examples)
Problem instances count: 8 , total verified (safe/unsat): 8 , total falsified (unsafe/sat): 0 , timeout: 0
mean time for ALL instances (total 8):12.023795893354652, max time: 23.111693859100342
mean time for verified SAFE instances(total 8): 12.023810923099518, max time: 23.111693859100342
safe (total 8), index: [0, 1, 2, 3, 4, 5, 6, 7]
```
It shows that the 8 examples (sub-regions for verification) are all verified
and no example is falsified or timeout. Therefore, the verification fully succeeded.

Our verification configurations have been tested on a GPU with 48GB memory.
If you are using a GPU with less memory, you may decrease the batch size
of verification by modifying the `batch_size` item in the configuration files
or passing an argument `--batch_size BATCH_SIZE`,
until it fits into the GPU memory.

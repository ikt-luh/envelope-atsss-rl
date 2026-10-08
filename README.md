# RL Agent for MPQUIC Traffic Steering

This repository provides a Reinforcement Learning (RL) agent for MPQUIC traffic steering between two interfaces (WiFi and 5G) on the [ENVELOPE](https://envelope-project.eu/) testbed, using [RLlib](https://docs.ray.io/en/latest/rllib/index.html) and the ENVELOPE ATSSS enabler.

The agent observes network metrics and outputs a continuous ratio that determines how to split the traffic between the two interfaces, aiming to minimize round trip time and packet loss rate.

> [!NOTE]
> **This code is intended for execution on a real B5G testbed.** For local development, it also provides a `--dry-run` mode that does not require testbed access but only covers limited functionality.


## Prerequisites


### Minimal Requirements (local development)

* Set up a Python 3.12 environment (e.g., via conda)
* Install the requirements
  ```
  (atsss-rl) $ pip install -r requirements.text
  ```

### Testbed Deployment

Before deployment, the following external projects need to be set up
* ENVELOPE Performance Measurement Function (PMF): Provides network measurements that form the observations of the agent.
* [ENVELOPE MPQUIC/ATSSS](https://github.com/ikt-luh/envelope-atsss-mpquic): The RL agent sends its decisions to the ATSSS AUE, which interacts with MPQUIC.

## Project Structure

| Path | Description |
|------|-------------|
| `aue_client/` | Posts the decisions of the agent to the ATSSS UE API |
| `configs/` | YAML parameter configs for the scenarios and agents |
| `pmf_client/` | Polling service for the Performance Measurement Function (PMF) |
| `rl_agent/` | PPO-FF, PPO-LSTM, and heuristic agents + shared RL environment |
| `scenarios/` | Network scenario generation via tc/netem |
| `scripts/` | Contains `train.py`, `eval.py`, `check_pmf.py` and plotting utilities |

## Local Development

> [!NOTE]
> Local development does not consider application traffic or the actual network conditions. The agent observes network conditions based on a simple stateful simulation that is initialized with values sampled from the scenario configuration files. No interaction between the agent's decisions and the network conditions is modeled here.

Important arguments and environment variables for local development:
* `--dry-run`: Simulate the PMF instead of connecting to it
* `DECISION_INTERVAL=None`: Do not wait between individual decisions of the agent (otherwise the code will just sleep without any effect)
* `CUDA_VISIBLE_DEVICES=0`: Use GPU 0 to accelerate training and evaluation. Can be set to `-1` for CPU-only training.

Relevant scenario configurations:
* Local development and debugging
  * `configs/scenarios/train.yaml`: Random network configurations used during training
  * `configs/scenarios/eval.yaml`: Fixed network configurations for testing
  * `configs/scenarios/eval_baseline.yaml`: Test without additional delay
* Local development and debugging with testbed-like conditions: `configs/scenarios/train_testbed_dry_run.yaml` and `configs/scenarios/eval_testbed_dry_run.yaml`

### Training
```bash
PYTHONPATH=. DECISION_INTERVAL=None CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --dry-run --agent ppo --config=configs/training/ppo.yaml --with-scenarios --scenario-config=configs/scenarios/train.yaml
```

### Evaluation

In all scenarios (but note that traffic generation is not supported locally)
```bash
PYTHONPATH=. DECISION_INTERVAL=None CUDA_VISIBLE_DEVICES=0 python scripts/eval.py --reward-alpha=0 --agent ppo --checkpoint=<PATH_OF_AGENT_MODEL> --steps=120 --with-scenarios --scenario-config=configs/scenarios/eval.yaml --dry-run --settle-s=0 --no-exploration
```

## Experiments on the Testbed

> [!IMPORTANT]
> **Note that most experiment scenarios generate artificial application traffic that will be forwarded by MPQUIC.** It is not possible to perform multiple experiments in parallel, as they would share the same interfaces. If you want to use live application data instead, make sure that your selected experiment scenario does not generate additional traffic, as this would affect the results.  

Important arguments and environment variables for local development:
* `DECISION_INTERVAL=1`: The decision interval should be larger than the time the PMF requires to probe the network and return the measurements. If the PMF cannot keep up, warnings will be printed. 

Relevant scenario configurations:
* `configs/scenarios/train_testbed.yaml`: Random network configurations used during training
* `configs/scenarios/eval_testbed.yaml`: Fixed network configurations for testing
* `configs/scenarios/eval_baseline.yaml`: Test without additional delay

### Preparation

**1) Check PMF Functionality**

Before starting any evaluation or training run, always verify that the Performance Measurement Function (PMF) is reachable and works as expected by running

```bash
python3 scripts/check_pmf.py
```

> [!WARNING] 
> The check will return the default idle network conditions. **Make sure that these conditions are in line with the experiment configurations you want to run**. The configuration files `train_testbed.yaml` and `eval_testbed.yaml` depend on the idle network conditions.

Troubleshooting: If the check fails, verify `PMF_UE_URL` / `PMF_UPF_URL` in `.env`. If IPs have changed, update `PMF_UPF_IP`, `PMF_UE_WIFI_IP`, `PMF_UE_5G_IP` accordingly.
It may also be necessary to restart the PMF or update the PMF config.

**2) Check end-to-end functionality**

To confirm that traffic generation and MPQUIC work, you can investigate the `bytes_sent.png` plot created in `results/evaluation/heuristic/[experiment_folder]` after running a heuristic agent on all evaluation traffic types

```bash
PYTHONPATH=. DECISION_INTERVAL=1 python3 scripts/eval.py --reward-alpha=0 --agent heuristic --heuristic=lb-utility --steps=30 --with-scenarios --scenario-config=configs/scenarios/eval_baseline.yaml
```

### Training

Train the RL agent on the testbed (warning: will take a long time) with artificial application traffic and randomized network conditions (on top of the real network conditions) 
```bash
PYTHONPATH=. DECISION_INTERVAL=1 python3 scripts/train.py --agent ppo --config=configs/training/ppo.yaml --with-scenarios --scenario-config=configs/scenarios/train_testbed.yaml
```

### Evaluation

Baseline evaluation under live network conditions and application traffic

```bash
PYTHONPATH=. DECISION_INTERVAL=1 python scripts/eval.py --reward-alpha=0  --agent=heuristic --heuristic=[lb-rtt-min|lb-plr-min|lb-utility|lb-random] - --steps=120
```

Evaluation of the agent under live network conditions and application traffic
```bash
PYTHONPATH=. DECISION_INTERVAL=1 python scripts/eval.py --reward-alpha=0  --agent ppo --checkpoint=<PATH_OF_AGENT_MODEL_DIR> --steps=120 --no-exploration
```

Full evaluation of a PPO agent under the network conditions and traffic profiles as defined in `eval_testbed.yaml`:
```bash
PYTHONPATH=. DECISION_INTERVAL=1 python scripts/eval.py --reward-alpha=0  --agent ppo --checkpoint=<PATH_OF_AGENT_MODEL> --steps=120 --no-exploration --with-scenarios --scenario-config=configs/scenarios/eval_testbed.yaml
```

## Acknowledgements

We thank Abebe S. Feleke for his major contributions to this project during his time as a research assistant at IKT.

This work has received funding from the European Union’s Horizon Europe research and innovation program under the ENVELOPE project (Grant Agreement No. 101139048).


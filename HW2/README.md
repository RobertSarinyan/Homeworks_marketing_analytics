# Homework 2 — A/B testing with multi-armed bandits

Compare ε-greedy and Thompson sampling over 20,000 trials per algorithm. Each trial shows one of four ads with true mean rewards of 1, 2, 3, and 4.

## Run

From the `HW2` directory, create a local virtual environment and run both scripts. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe Bandit.py
.\.venv\Scripts\python.exe bonus.py
```

On macOS and Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`. If you already have a Python environment activated, install the dependencies with `python -m pip install -r requirements.txt` and run `python Bandit.py` and `python bonus.py`. Tested with Python 3.13.5.

## Experiment design

- Rewards are Gaussian with variance 1 (known precision 1). The policies observe only the reward from the selected ad. The true means define the simulation and regret benchmark.
- **ε-greedy:** Explore uniformly with probability `1/t` on trial `t`. Otherwise choose an ad with the highest estimated mean. Untried ads receive optimistic scores until they are sampled. Each observed reward updates the selected ad's sample mean.
- **Thompson sampling:** Start with independent normal priors with mean 0 and variance 100. Sample a mean for each ad, choose the largest draw, and update the selected ad's posterior with the observed reward.
- **Regret:** Each trial adds the best ad's true mean minus the selected ad's true mean. Cumulative regret sums these gaps.
- **Reproducibility:** The main comparison uses seed 42 and gives both policies the same Gaussian reward noise on each trial. Policy decisions use separate random streams from the reward noise.

## Results

`Bandit.py` logs cumulative and average reward and regret for both algorithms. It regenerates these files in `output/`:

| File | Contents |
| --- | --- |
| `rewards.csv` | 40,000 trial rows with columns `Bandit, Reward, Algorithm`. Bandit IDs run from 1 to 4. |
| `learning_curves.png` | Running average reward on linear and logarithmic trial scales. |
| `cumulative_comparison.png` | Cumulative reward and cumulative regret over time. |

| Algorithm | Cumulative reward | Cumulative regret |
| --- | ---: | ---: |
| ε-greedy | 79,895.88 | 28 |
| Thompson sampling | 79,892.88 | 31 |

## Bonus — repeat the comparison across seeds

`bonus.py` runs 50 paired simulations with fresh bandits. It reports means and standard deviations across runs, median cumulative regret, and the paired difference in regret. The median describes a typical run, while the mean reflects the cost of occasional high-regret runs.

Across these simulations, ε-greedy's mean cumulative regret was 464.26 and its median was 20.00. Thompson sampling's mean was 31.20 and its median was 30.50. The mean paired difference was 433.06 in favor of Thompson sampling, while the median paired difference favored ε-greedy by 10. These results describe the simulated reward model and settings above.

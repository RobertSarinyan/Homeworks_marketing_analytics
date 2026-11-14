"""Compare two policies for selecting among four advertisements."""

import csv
from abc import ABC, abstractmethod
from itertools import accumulate
from math import inf, sqrt
from pathlib import Path
from random import Random

from loguru import logger


BANDIT_REWARDS = (1, 2, 3, 4)
NUMBER_OF_TRIALS = 20_000
REWARD_PRECISION = 1.0
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        """Initialize the bandit with the true mean rewards in p."""
        pass

    @abstractmethod
    def __repr__(self):
        """Return the algorithm name used in reports."""
        pass

    @abstractmethod
    def pull(self):
        """Select an ad and observe its reward."""
        pass

    @abstractmethod
    def update(self):
        """Update the selected ad's estimate with the observed reward."""
        pass

    @abstractmethod
    def experiment(self):
        """Run the requested number of trials."""
        pass

    @abstractmethod
    def report(self):
        """Write rewards to CSV and log performance metrics."""
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


class GaussianBandit(Bandit):
    """Share the reward model and reporting across both selection policies."""

    def __init__(self, p, precision=REWARD_PRECISION, seed=SEED):
        """Set the ad means, known reward precision, and random streams."""
        self.reward_means = tuple(p)
        self.best_mean = max(self.reward_means)
        self.precision = precision
        self.arms = range(len(self.reward_means))
        self.policy_rng = Random(f"policy:{seed}")
        self.reward_rng = Random(seed)
        self.chosen_arms = []
        self.rewards = []
        self.regrets = []

    def __repr__(self):
        """Return the concrete algorithm's name."""
        return type(self).__name__

    @abstractmethod
    def select_arm(self):
        """Return the index of the ad to show on the next trial."""

    def pull(self):
        """Select an ad and draw a Gaussian reward from that ad."""
        arm = self.select_arm()
        reward = self.reward_rng.gauss(self.reward_means[arm], 1 / sqrt(self.precision))
        return arm, reward

    def experiment(self, trials=NUMBER_OF_TRIALS):
        """Run trials, update the policy, and record rewards and regret."""
        for _ in range(trials):
            arm, reward = self.pull()
            self.update(arm, reward)
            self.chosen_arms.append(arm)
            self.rewards.append(reward)
            self.regrets.append(self.best_mean - self.reward_means[arm])

    def report(self, writer):
        """Write one row per trial and log cumulative and average metrics."""
        writer.writerows(
            (arm + 1, reward, repr(self))
            for arm, reward in zip(self.chosen_arms, self.rewards)
        )
        total_reward = sum(self.rewards)
        total_regret = sum(self.regrets)
        trials = len(self.rewards)
        logger.info(
            f"{self!r} | cumulative reward: {total_reward:.2f} | "
            f"cumulative regret: {total_regret:.2f} | "
            f"average reward: {total_reward / trials:.4f} | "
            f"average regret: {total_regret / trials:.4f}"
        )


class EpsilonGreedy(GaussianBandit):
    """Explore with probability 1/t and otherwise use the best estimate."""

    def __init__(self, p, precision=REWARD_PRECISION, seed=SEED):
        """Initialize sample means and counts for all ads."""
        super().__init__(p, precision, seed)
        self.estimates = [0.0] * len(self.arms)
        self.counts = [0] * len(self.arms)

    def select_arm(self):
        """Explore uniformly or choose uniformly among the best estimates."""
        epsilon = 1 / (len(self.rewards) + 1)
        if self.policy_rng.random() < epsilon:
            return self.policy_rng.choice(self.arms)

        # Optimistic estimates let every ad get an initial observation.
        scores = [mean if count else inf for mean, count in zip(self.estimates, self.counts)]
        best_score = max(scores)
        best_arms = [arm for arm in self.arms if scores[arm] == best_score]
        return self.policy_rng.choice(best_arms)

    def update(self, arm, reward):
        """Update the selected ad's sample mean with one observation."""
        self.counts[arm] += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]


class ThompsonSampling(GaussianBandit):
    """Use Gaussian posteriors for ad means with known reward precision."""

    def __init__(self, p, precision=REWARD_PRECISION, seed=SEED):
        """Start each unknown ad mean with a normal prior N(0, 10^2)."""
        super().__init__(p, precision, seed)
        self.posterior_means = [0.0] * len(self.arms)
        self.posterior_precisions = [0.01] * len(self.arms)

    def select_arm(self):
        """Draw once from each posterior and select the largest draw."""
        samples = [
            self.policy_rng.gauss(mean, 1 / sqrt(precision))
            for mean, precision in zip(self.posterior_means, self.posterior_precisions)
        ]
        return max(self.arms, key=samples.__getitem__)

    def update(self, arm, reward):
        """Combine the current estimate and new observation by precision."""
        old_precision = self.posterior_precisions[arm]
        new_precision = old_precision + self.precision
        self.posterior_means[arm] = (
            old_precision * self.posterior_means[arm] + self.precision * reward
        ) / new_precision
        self.posterior_precisions[arm] = new_precision


class Visualization:
    """Save learning curves and cumulative comparisons of the policies."""

    def __init__(self, bandits, output_dir):
        """Use the completed experiments and the report output directory."""
        self.bandits = bandits
        self.output_dir = output_dir

    def plot1(self):
        """Plot each policy's running average reward on linear and log scales."""
        from matplotlib.figure import Figure

        fig = Figure(figsize=(11, 4), layout="constrained")
        axes = fig.subplots(1, 2)
        for bandit in self.bandits:
            trials = range(1, len(bandit.rewards) + 1)
            averages = [
                total / trial for trial, total in enumerate(accumulate(bandit.rewards), 1)
            ]
            for ax in axes:
                ax.plot(trials, averages, label=repr(bandit))

        for ax, scale in zip(axes, ("linear", "log")):
            ax.axhline(
                self.bandits[0].best_mean, linestyle="--", color="gray", label="Best ad mean"
            )
            ax.set(
                xscale=scale,
                xlabel="Trial",
                ylabel="Average reward to date",
                title=f"Learning curve ({scale} scale)",
            )
            ax.legend()
        fig.savefig(self.output_dir / "learning_curves.png", dpi=150)

    def plot2(self):
        """Plot cumulative reward and cumulative regret by trial."""
        from matplotlib.figure import Figure

        fig = Figure(figsize=(11, 4), layout="constrained")
        axes = fig.subplots(1, 2)
        for bandit in self.bandits:
            trials = range(1, len(bandit.rewards) + 1)
            axes[0].plot(trials, list(accumulate(bandit.rewards)), label=repr(bandit))
            axes[1].plot(trials, list(accumulate(bandit.regrets)), label=repr(bandit))

        for ax, metric in zip(axes, ("Cumulative reward", "Cumulative regret")):
            ax.set(xlabel="Trial", ylabel=metric, title=metric)
            ax.legend()
        fig.savefig(self.output_dir / "cumulative_comparison.png", dpi=150)


def comparison():
    """Run both policies and save the homework CSV and figures."""
    bandits = [EpsilonGreedy(BANDIT_REWARDS), ThompsonSampling(BANDIT_REWARDS)]
    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / "rewards.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("Bandit", "Reward", "Algorithm"))
        for bandit in bandits:
            bandit.experiment()
            bandit.report(writer)

    visualization = Visualization(bandits, OUTPUT_DIR)
    visualization.plot1()
    visualization.plot2()
    logger.info(f"Saved CSV and figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    comparison()

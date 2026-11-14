"""Evaluate both policies across repeated, paired simulations.

One run can favor a policy that happened to get good rewards early. Repeat the
same experiment with fresh bandits for each seed and summarize the results
across runs. Within each seed, both policies face the same reward noise at
each trial, making their reward difference attributable to their ad choices.
"""

from statistics import mean, median, stdev

from loguru import logger

from Bandit import BANDIT_REWARDS, EpsilonGreedy, ThompsonSampling


REPETITIONS = 50


def repeated_comparison():
    """Log final reward and regret distributions from 50 paired runs."""
    regret_by_policy = []
    for policy in (EpsilonGreedy, ThompsonSampling):
        totals = []
        regrets = []
        for seed in range(REPETITIONS):
            bandit = policy(BANDIT_REWARDS, seed=seed)
            bandit.experiment()
            totals.append(sum(bandit.rewards))
            regrets.append(sum(bandit.regrets))

        regret_by_policy.append(regrets)
        logger.info(
            f"{policy.__name__} over {REPETITIONS} runs | "
            f"cumulative reward: mean {mean(totals):.2f}, SD {stdev(totals):.2f} | "
            f"cumulative regret: mean {mean(regrets):.2f}, "
            f"median {median(regrets):.2f}, SD {stdev(regrets):.2f}"
        )

    epsilon_regrets, thompson_regrets = regret_by_policy
    regret_differences = [
        epsilon - thompson for epsilon, thompson in zip(epsilon_regrets, thompson_regrets)
    ]
    logger.info(
        f"Paired regret difference (EpsilonGreedy minus ThompsonSampling) | "
        f"mean {mean(regret_differences):.2f} | "
        f"median {median(regret_differences):.2f} | SD {stdev(regret_differences):.2f}"
    )


if __name__ == "__main__":
    repeated_comparison()

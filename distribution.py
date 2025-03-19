import numpy as np
from collections import Counter

class Distribution:
    def __init__(self, sample_space, absolute_frequencies=None):
        if absolute_frequencies is None:
            absolute_frequencies = [1/len(sample_space)] * len(sample_space)

        if len(sample_space) != len(absolute_frequencies):
            raise ValueError("Sample space and absolute frequencies must have same length")
        corrected_sample_space = []
        for state in sample_space:
            if isinstance(state, str):
                corrected_sample_space.append(str(state))
            elif isinstance(state, list):
                corrected_sample_space.append(tuple(state))
            elif isinstance(state, int):
                corrected_sample_space.append(int(state))
            else:
                corrected_sample_space.append(state)

        relative_frequencies = np.array(absolute_frequencies)/np.array(absolute_frequencies).sum()
        self.distribution = {sample: float(probability) for sample, probability in zip(corrected_sample_space, relative_frequencies)}

    def __repr__(self):
        rows = "\n".join(f"  {sample}: {probability*100:.1f}%" for sample, probability in self.distribution.items())
        return f"Distribution(\n{rows}\n)"

    def __getitem__(self, key):
        return self.distribution[key]

    @classmethod
    def from_dict(cls, dictionary):
        return Distribution(list(dictionary.keys()), list(dictionary.values()))

    @classmethod
    def from_sample(cls, sample_list):
        corrected_samples = []
        for sample in sample_list:
            if isinstance(sample, str):
                corrected_samples.append(str(sample))
            elif isinstance(sample, list):
                corrected_samples.append(tuple(sample))
            elif isinstance(sample, int):
                corrected_samples.append(int(sample))
            else:
                corrected_samples.append(sample)
        counter = Counter(corrected_samples)
        return Distribution(list(counter.keys()), list(counter.values()))

    @classmethod
    def extend_to_same_space(cls, distribution1: "Distribution", distribution2: "Distribution"):
        total_list = []
        total_list += distribution1.sample_space()
        total_list += distribution2.sample_space()

        for key in total_list:
            if key not in distribution1.sample_space():
                distribution1.distribution[key] = 0.
            if key not in distribution2.sample_space():
                distribution2.distribution[key] = 0.

        return distribution1, distribution2

    @classmethod
    def convex_combination(cls, distribution_list, coefficient_list):
        if len(distribution_list) == 0:
            raise ValueError("distribution_list cannot be empty")
        if len(distribution_list) != len(coefficient_list):
            raise ValueError("distribution_list and coefficient_list must have same length")
        if min(coefficient_list) < 0:
            raise ValueError("coefficient_list must be non-negative")
        first_space = distribution_list[0].sample_space()
        for i in range(1, len(distribution_list)):
            if first_space != distribution_list[i].sample_space():
                raise ValueError("Each distribution in distribution_list must have the same sample space")

        new_distribution = Distribution(first_space, np.sum([coefficient * np.array(list(distribution.probabilities())) for coefficient, distribution in zip(coefficient_list, distribution_list)], axis=0))
        return new_distribution

    def sample_space(self):
        return self.distribution.keys()

    def probabilities(self):
        return self.distribution.values()

    def sample(self):
        sample_idxs = range(len(self.distribution.keys()))
        return list(self.distribution.keys())[np.random.choice(sample_idxs, 1, p=list(self.distribution.values()))[0]]

    def extend_space(self, new_sample_space):
        new_distribution = Distribution(new_sample_space)
        extended = self.extend_to_same_space(self, new_distribution)
        self.distribution = extended[0].distribution
        return self

    def restrict_space(self, new_sample_space):
        if not set(new_sample_space).issubset(set(self.distribution.keys())):
            raise ValueError("Can only restrict the sample space to a subspace of the sample space.")
        return Distribution.from_dict({key: value for key, value in zip(self.distribution.keys(), self.distribution.values()) if key in new_sample_space})

    # RV X on A (distribution P), RV Y on B (distribution B), f: A -> B. Calculate P[X = x | f(x) =^d Y]
    def get_conditioned_on_Q(self, dist_q, f):
        # Group X by f(x)
        groups = {}

        for x in self.sample_space():
            y = f(x)
            if y not in groups:
                groups[y] = []
            groups[y].append(x)

        # Identify reachable values in Y
        reachable_y = set(groups.keys())
        reachable_y = [y for y in reachable_y if y is not None]  # exclude None

        # Restrict dist_q to reachable Y
        total_reachable = sum([dist_q[y] for y in reachable_y])
        if total_reachable < 1e-10:
            raise ValueError("The distribution cannot be constructed. None of the y is reachable.")

        dist_q_restricted = {y: dist_q[y] / total_reachable for y in reachable_y}

        # Compute Z_y for each y
        Z = {y: sum(self[x] for x in group) for y, group in groups.items()}
        Z_sum = sum([dist_q_restricted[y]*Z[y] for y in reachable_y])

        # Compute R(x)
        R = {}
        for x in self.sample_space():
            y = f(x)
            R[x] = dist_q_restricted[y] * self[x] / Z_sum if y is not None else 0.

        # for exact sum == 1
        R_SUM = sum(R.values())
        for x in self.sample_space():
            R[x] = R[x] / R_SUM

        return self.from_dict(R)



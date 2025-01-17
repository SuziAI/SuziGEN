import numpy as np

class Distribution:
    def __init__(self, sample_space, absolute_frequencies):
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
        relative_frequencies = np.array(absolute_frequencies)/sum(absolute_frequencies)
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
    def extend_to_same_space(cls, distribution1: "Distribution", distribution2: "Distribution"):
        total_list = []
        total_list += distribution1.keys()
        total_list += distribution2.keys()

        for key in total_list:
            if key not in distribution1:
                distribution1[key] = 0.
            if key not in distribution2:
                distribution2[key] = 0.

        return distribution1, distribution2

    def sample_space(self):
        return self.distribution.keys()

    def probabilities(self):
        return self.distribution.values()

    def sample(self):
        sample_idxs = range(len(self.distribution.keys()))
        return list(self.distribution.keys())[np.random.choice(sample_idxs, 1, p=list(self.distribution.values()))[0]]

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
        reachable_y = [y for y in reachable_y if y]  # exclude None

        # Restrict dist_q to reachable Y
        total_reachable = sum([dist_q[y] for y in reachable_y])
        dist_q_restricted = {y: dist_q[y] / total_reachable for y in reachable_y}

        # Compute Z_y for each y
        Z = {y: sum(self[x] for x in group) for y, group in groups.items()}

        # Compute R(x)
        R = {}
        for x in self.sample_space():
            y = f(x)
            R[x] = dist_q_restricted[y] * self[x] / Z[y] if Z[y] > 1e-8 else 0

        # for exact sum == 1
        R_SUM = sum(R.values())
        for x in self.sample_space():
            R[x] = R[x] / R_SUM

        return self.from_dict(R)



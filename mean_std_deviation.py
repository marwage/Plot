import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mean_std_deviation(samples: [np.array], losses: [np.array]):
    num_runs = len(samples)
    new_samples = []
    std_deviations = []

    if num_runs == len(losses):
        first_samples = samples[0]
        num_data_points = first_samples.shape[0]
        for i in range(num_data_points):
            values = [losses[0][i]]
            for j in range(1, num_runs):
                samples_j = samples[j]
                indices_j = np.asarray(first_samples[i] == samples_j).nonzero()
                if indices_j[0].size == 0:
                    break
                else:
                    index = indices_j[0][0]
                    values.append(losses[j][index])

            if len(values) == num_runs:
                std_deviation = np.std(np.array(values))

                new_samples.append(first_samples[i])
                std_deviations.append(std_deviation)

        return new_samples, std_deviations

    return None
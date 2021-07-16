import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def exp_moving_avg(values, alpha=0.1):
    smoothed_values = np.zeros_like(values)
    num_values = values.shape[0]

    for i in range(num_values):
        if i == 0:
            smoothed_values[i] = values[i]
        else:
            last_value = smoothed_values[i - 1]
            smoothed_values[i] = (1 - alpha) * last_value + alpha * values[i]

    return smoothed_values


def mean_loss_workers(path: str, num_workers=4):
    losses = []
    steps = np.array([0])
    for i in range(num_workers):
        file_path = os.path.join(path, "scalar_{}.csv".format(i))
        if not os.path.exists(file_path):
            file_path = os.path.join(path, "summary_{}.csv".format(i))
        data_frame = pd.read_csv(file_path)
        if "value" in data_frame:
            loss = data_frame["value"].values
        else:
            loss = data_frame["loss"].values
        losses.append(loss)

        if i == 0:
            steps = data_frame["step"].values

    loss_all = np.stack(losses)
    loss_mean = np.mean(loss_all, axis=0)

    return steps, loss_mean


def mean_loss_runs(runs: list):
    steps = runs[0][0]
    losses = [run[1] for run in runs]

    loss_all = np.stack(losses)
    loss_mean = np.mean(loss_all, axis=0)

    return steps, loss_mean


def get_steps_loss(path: str):
    data_frame = pd.read_csv(path)

    loss = data_frame["value"].values
    steps = data_frame["step"].values

    return steps, loss


def sliding_window_mean(values: np.array, window_size=100):
    num_values = values.shape[0]

    new_num_values = num_values - window_size
    new_values = np.zeros((new_num_values,))
    for i in range(new_num_values):
        new_values[i] = np.mean(values[i:i + window_size])

    return new_values


def get_std_deviations(samples: [np.array], losses: [np.array]):
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


def plot(batch_size: int,
         workers=[int],
         ema=False,
         sliding_window=False,
         seed=False,
         no_shuffle=False,
         sampling=1,
         kungfu=False,):
    num_runs = 3

    comparison = {}
    # num_workers
    # run

    for num_workers in workers:
        runs = []

        for i in range(1, num_runs + 1):
            path = "./data/squad_distr_{}_{}".format(batch_size, num_workers)
            if kungfu:
                path = "{}_kf".format(path)
            if seed:
                path = "{}_seed".format(path)
            if no_shuffle:
                path = "{}_no_shuffle".format(path)
            path = "{}/{}".format(path, i)
            if num_workers > 1:
                runs.append(mean_loss_workers(path, num_workers))
            else:
                path = "{}/{}".format(path, "scalar.csv")
                runs.append(get_steps_loss(path))

        comparison[num_workers] = mean_loss_runs(runs)

    if ema:
        alpha = 0.1
        for num_workers in workers:
            em_avg = exp_moving_avg(comparison[num_workers][1], alpha)
            comparison[num_workers] = comparison[num_workers][0], em_avg

    if sliding_window:
        window_size = 100
        for num_workers in workers:
            swm = sliding_window_mean(comparison[num_workers][1], window_size)
            steps = comparison[num_workers][0][0:-window_size]
            comparison[num_workers] = steps, swm

    if sampling > 1:
        for num_workers in workers:
            num_samples = comparison[num_workers][0].shape[0]
            steps = []
            loss = []
            for j in range(num_samples):
                if comparison[num_workers][0][j] % sampling == 0:
                    steps.append(comparison[num_workers][0][j])
                    loss.append(comparison[num_workers][1][j])
            comparison[num_workers] = np.array(steps), np.array(loss)

    figure, axis = plt.subplots()

    title = "Different number of workers"
    if ema:
        title = title + ", EMA ({})".format(alpha)
    if sliding_window:
        title = title + ", Sliding window ({})".format(window_size)
    if sampling > 1:
        title = title + ", Sampling ({})".format(sampling)
    if kungfu:
        title = title + ", KungFu"
    plt.title(title)

    linewidth = 0.75
    for num_workers in workers:
        axis.plot(comparison[num_workers][0],
                  comparison[num_workers][1],
                  label=num_workers,
                  linewidth=linewidth)

    axis.legend()
    axis.set_xlabel("Step")
    axis.set_ylabel("Loss")

    figure.set_size_inches(8, 4.5)
    figure.tight_layout()

    path = "./output/squad_distr_{}_{}".format(batch_size, "diff")
    if kungfu:
        path = path + "_kf"
    if seed:
        path = path + "_seed"
    if no_shuffle:
        path = path + "_no_shuffle"
    if sampling > 1:
        path = path + "_sampling"
    if ema:
        path = path + "_ema"
    if sliding_window:
        path = path + "_sw"
    path = path + ".pdf"
    figure.savefig(path)

    if ema:
        print("EMA ({})".format(alpha))
    if sliding_window:
        print("Sliding window ({})".format(window_size))
    if sampling > 1:
        print("Sampling ({})".format(sampling))

    for num_workers in workers:
        final_loss = comparison[num_workers][1][-1]
        print("{}: Final loss: {}".format(num_workers, final_loss))

    samples = []
    losses = []
    for num_workers in workers:
        samples.append(comparison[num_workers][0] * batch_size)
        losses.append(comparison[num_workers][1])
    dev_samples, deviations = get_std_deviations(samples, losses)
    msd = np.mean(deviations)
    print("Mean standard deviation: {}".format(msd))


def main():
    batch_size = 24
    workers = [1, 2, 3, 4]
    ema = False
    sliding_window = False
    seed = True
    no_shuffle = False
    sampling = 100
    kungfu = True

    plot(batch_size,
         workers,
         ema,
         sliding_window,
         seed,
         no_shuffle,
         sampling,
         kungfu)


if __name__ == "__main__":
    main()

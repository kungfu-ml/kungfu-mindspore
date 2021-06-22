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
    

def main():
    num_runs = 3

    data_frames = []
    for i in range(1, num_runs + 1):
        file_path = "./bs16/scalar_{}.csv".format(i)
        data_frame = pd.read_csv(file_path)
        data_frames.append(data_frame)

    steps = []
    samples = []
    losses = []
    for i in range(num_runs):
        steps_i = data_frames[i]["step"].values
        steps.append(steps_i)
        samples.append(steps_i * 16)
        losses.append(data_frames[i]["value"].values)

    new_samples, std_deviations = mean_std_deviation(samples, losses)

    mean_std_dev = np.mean(std_deviations)
    print("Mean standard deviation: {}".format(mean_std_dev))

    figure, axis = plt.subplots()

    linewidth=1.0
    axis.plot(new_samples, std_deviations, linewidth=linewidth)

    # axis.legend()
    axis.set_xlabel("Samples")
    axis.set_ylabel("Standard deviation")

    figure.set_size_inches(16, 9)
    figure.tight_layout()
    figure.savefig("mean_std_deviation_bs16.pdf")


if __name__ == "__main__":
    main()

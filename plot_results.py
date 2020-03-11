from write_and_read_results import read_results_from_files, TOTAL_PIXELS
from matplotlib import pyplot as plt

results = read_results_from_files()

median_colours = dict()

## Plot Median Filter
plt.subplot(1, 2, 2)
plt.title("Average Time Taken To Do Median Filter")

for key in results.keys():
    try:
        p = plt.plot(results[key]["median filter"], label=key, marker=".")
        median_colours[key] = p[-1].get_color()
    except KeyError:
        continue

plt.yscale("log")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.legend()


def truediv(a, b):
    return a / b


# Plot Adding Speed Difference
plt.subplot(1, 2, 1)
plt.title("Speed Change For Median Filter When Compared With scipy")

for key in results.keys():
    if key == "scipy":
        continue
    try:
        diff = list(
            map(
                truediv,
                results["scipy"]["median filter"],
                results[key]["median filter"],
            )
        )
        plt.plot(diff, label=key, marker=".", color=median_colours[key])
    except KeyError:
        continue
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.xlabel("Number of Pixels/Elements")

plt.show()

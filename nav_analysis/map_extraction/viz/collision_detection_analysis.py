import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import collections
import gzip

sns.set(style="whitegrid")


def main():
    with gzip.open("collision_detection_and_pruning.json.gz", "rt") as f:
        data = json.load(f)

    sns_data = collections.defaultdict(list)
    for v in data:
        #  if v["bal_acc"] < 0.:
        #  continue

        if v["n_sig"] > 20:
            continue

        sns_data["l1_pen"].append(v["l1_pen"])
        sns_data["bal_acc"].append(v["bal_acc"])
        sns_data["acc"].append(v["acc"])
        sns_data["num_neurons"].append(v["n_sig"])

    print(min(sns_data["num_neurons"]))
    print(min(sns_data["l1_pen"]))
    print(max(v["bal_acc"] for v in data))
    sns_data = pd.DataFrame.from_dict(sns_data)

    fig = plt.figure()
    ax1 = plt.gca()
    sns.lineplot(x="num_neurons", y="bal_acc", data=sns_data, ax=ax1)
    ax1.invert_xaxis()
    #  ax2 = ax1.twiny()
    #  sns.lineplot(x="l1_pen", y="bal_acc", data=sns_data, ax=ax2)

    plt.ylabel("Class Balanced Accuracy")
    plt.xlabel("Numer of Significant Neurons")
    sns.despine(fig=fig)
    plt.savefig("collision_detection.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

import h5py as h5
import numpy as np
import json
import gzip

split = "val"
fname = "data/map_extraction/collisions/collision-dset-2022-20-01_{split}.h5"


with gzip.open("collision_detection_and_pruning.json.gz", "rt") as f:
    pruning_res = json.load(f)


max_acc = max(v["bal_acc"] for v in pruning_res)
print(max_acc)


min_neurons_within_thresh = 1e4
acc_by_n_sig = {}
for v in pruning_res:
    acc_by_n_sig[v["n_sig"]] = acc_by_n_sig.get(v["n_sig"], []) + [v["bal_acc"]]


acc_by_n_sig = {k: sum(v) / len(v) for k, v in acc_by_n_sig.items()}
print(acc_by_n_sig)

for k, v in acc_by_n_sig.items():
    if v > 0.97 * max_acc:
        min_neurons_within_thresh = min(k, min_neurons_within_thresh)

print(sorted(list(acc_by_n_sig.keys())))
print(min_neurons_within_thresh)
print(acc_by_n_sig[min_neurons_within_thresh])


accs_per_neuron = {}
pens = set()
for v in pruning_res:
    if v["n_sig"] != min_neurons_within_thresh:
        continue

    for n in v["sig_nueron_inds"]:
        accs_per_neuron[n] = accs_per_neuron.get(n, []) + [v["bal_acc"]]

    pens.add(v["l1_pen"])


votes_per_neuron = {n: len(v) for n, v in accs_per_neuron.items()}

print(votes_per_neuron)
print(list(votes_per_neuron.keys()))

top_n_neurons = np.asarray(
    sorted(
        list(
            sorted(
                votes_per_neuron.keys(), key=lambda n: votes_per_neuron[n], reverse=True
            )
        )[0:min_neurons_within_thresh]
    ),
    dtype=np.int64,
)
print(top_n_neurons)


with h5.File(fname.format(split=split), "r") as fin, h5.File(
    "data/map_extraction/collisions/collision_pca.h5", "w"
) as f:
    xs = fin["hidden_states"][:, top_n_neurons]
    ys = fin["collision_labels"][()]

    f["top_n_neurons"] = top_n_neurons
    f["collision_labels"] = ys
    f["hidden_states"] = xs
    f["prev_actions"] = fin["prev_actions"][()]

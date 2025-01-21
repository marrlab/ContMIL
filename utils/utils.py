import gzip
import pickle


def loaddata(small=False):
    if not small:
        print("Loading features", end="... ", flush=True)
        with gzip.open("dataset/features.pkl", "rb") as f:
            features = pickle.load(f)
        print("[done]")
    else:
        print("Loading features", end="... ", flush=True)
        with gzip.open("dataset/reduced_data/features-small.pkl", "rb") as f:
            features = pickle.load(f)
        print("[done]")

    return features

# import pickle
# import gzip
#
#
#
# def loaddata(small=False):
#     if not small:
#         print("Loading patient lists and labels", end="... ", flush=True)
#         with open("../dataset/split.dat", "rb") as f:
#             pat_lists = pickle.load(f)
#         with open("../dataset/patients.dat", "rb") as f:
#             patients = pickle.load(f)
#         print("[done]")
#         print("Loading features", end="... ", flush=True)
#         with open("../dataset/features.pkl", "rb") as f:
#             features = pickle.load(f)
#         print("[done]")
#     else:
#         print("Loading features", end="... ", flush=True)
#         with gzip.open("../dataset/features-small.pkl", "rb") as f:
#             features = pickle.load(f)
#         print("[done]")
#
#         print("Loading patient lists and labels", end="... ", flush=True)
#         with open("../dataset/split-small.dat", "rb") as f:
#             pat_lists = pickle.load(f)
#         with open("../dataset/patients-small.dat", "rb") as f:
#             patients = pickle.load(f)
#         print("[done]")
#
#     return features, patients, pat_lists
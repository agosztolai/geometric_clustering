"""Functions for saving and loading"""
import pickle


def save_curvatures(times, kappas, filename="curvature"):
    """save curvatures in a pickle"""
    pickle.dump([times, kappas], open(filename + ".pkl", "wb"))


def load_curvature(filename="curvature"):
    """load curvatures from a pickle"""
    times, kappas = pickle.load(open(filename + ".pkl", "rb"))
    return times, kappas

def save_embedding(embedding, filename=None):
    """save embedding results"""
    pickle.dump(embedding, open(filename + "_embed.pkl", "wb"))

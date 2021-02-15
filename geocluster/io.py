"""Functions for saving and loading."""
import pickle


def save_curvatures(times, kappas, filename="curvature.pkl"):
    """Save curvatures in a pickle."""
    pickle.dump([times, kappas], open(filename, "wb"))


def load_curvature(filename="curvature"):
    """Load curvatures from a pickle."""
    times, kappas = pickle.load(open(filename, "rb"))
    return times, kappas

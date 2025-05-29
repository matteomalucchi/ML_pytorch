import numpy as np

def binned_arctanh_2bins(x, bin_edge):
    """
    Apply binned arctanh transformation to the input array x based on the provided middle bin edge.
    The transformation is defined as:
    lambda x: np.digitize(np.arctanh(x), [np.min(np.arctanh(x))-1, bin_edge, np.max(np.arctanh(x))+1])-1
    """
    
    arctanh_x = np.arctanh(x)
    bin_edges = [np.min(arctanh_x) - 1, bin_edge, np.max(arctanh_x) + 1]
    return np.digitize(arctanh_x, bin_edges) - 1


functions_dict={
    "binned_arctanh_2bins": binned_arctanh_2bins,
}
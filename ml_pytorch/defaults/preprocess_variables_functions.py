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

def pad_arctanh(x, pad_limit, pad_value=-999.0):
    """
    Apply arctanh transformation to the input array x and pad it with a specified value if it exceeds the pad_limit.
    """
    
    arctanh_x = np.arctanh(x)
    # if np.any(np.abs(arctanh_x) > pad_limit):
    arctanh_x = np.where(arctanh_x > pad_limit, pad_value, arctanh_x)
        
    return arctanh_x


functions_dict={}
# automatically add all functions in the current module to the functions_dict
import inspect
current_module = inspect.getmodule(inspect.currentframe())
for name, obj in inspect.getmembers(current_module):
    if inspect.isfunction(obj) and name not in functions_dict:
        functions_dict[name] = obj

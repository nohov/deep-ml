import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    if mode == "row":
        results = a_t.mean(dim=1)
    elif mode == "column":
        results = a_t.mean(dim=0)
    else:
        raise ValueError("mode must be 'row' or 'column'")


    return results


import torch
from torchsort.ops import _arange_like, isotonic_kl, isotonic_l2, isotonic_l2_backward, isotonic_kl_backward

class SoftSortByColumn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, regularization="l2", regularization_strength=1.0, column=0):
        tensor_t = torch.transpose(tensor, -2, -1)
        b, h, w = tensor_t.shape
        tensor_t_merged = tensor_t.contiguous().view(-1, w)
        
        ctx.sign = -1
        ctx.regularization = regularization
        w_ = (_arange_like(tensor_t_merged, reverse=True) + 1) / regularization_strength
        tensor_t_merged = ctx.sign * tensor_t_merged  # for ascending

        relevant_columns = tensor_t[..., column, :]
        permutation_columns = torch.argsort(relevant_columns, -1)
        # permutation_reshaped = permutation_columns.repeat_interleave(h, -2)
        permutation_reshaped = permutation_columns.view(b, 1, w).expand(b, h, w).contiguous().view(-1, w)
        s = tensor_t_merged.gather(-1, permutation_reshaped)

        # note reverse order of args
        if ctx.regularization == "l2":
            sol = isotonic_l2[s.device.type](w_ - s)
        else:
            sol = isotonic_kl[s.device.type](w_, s)
        ctx.save_for_backward(s, sol, permutation_reshaped)
        out = ctx.sign * (w_ - sol)
        
        # Separate into batches again and re-transpose
        out = out.view(b, h, w).transpose(-2, -1)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_t = torch.transpose(grad_output, -2, -1)
        b, h, w = grad_output_t.shape
        grad_output_t_merged = grad_output_t.contiguous().view(-1, w)

        s, sol, permutation = ctx.saved_tensors
        inv_permutation = torch.argsort(permutation, -1)

        if ctx.regularization == "l2":
            grad = isotonic_l2_backward[s.device.type](s, sol, grad_output_t_merged)
        else:
            grad = isotonic_kl_backward[s.device.type](s, sol, grad_output_t_merged)

        # Reshape back
        grad_gathered = grad.gather(-1, inv_permutation)
        grad_gathered = grad_gathered.view(b, h, w).transpose(-2, -1)
        
        return grad_gathered, None, None, None


def soft_sort_by_column(values, regularization="l2", regularization_strength=1.0, column=0):
    return SoftSortByColumn.apply(values, regularization, regularization_strength, column)
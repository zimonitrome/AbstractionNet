import torch
from torchsort.ops import _arange_like, isotonic_kl, isotonic_l2, _inv_permutation, isotonic_l2_backward, isotonic_kl_backward, soft_sort

class SoftSortByColumn1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, regularization="l2", regularization_strength=1.0, column=0):
        ctx.sign = -1
        ctx.regularization = regularization
        w = (_arange_like(tensor, reverse=True) + 1) / regularization_strength
        tensor = ctx.sign * tensor  # for ascending
        # idk I hope this works lol
        permutation_1d = torch.argsort(tensor[:,column], descending=True)
        permutation = permutation_1d.unsqueeze(-1).expand(-1, tensor.shape[-1])
        s = tensor.gather(0, permutation)


        # note reverse order of args
        if ctx.regularization == "l2":
            sol = isotonic_l2[s.device.type](w - s)
        else:
            sol = isotonic_kl[s.device.type](w, s)

        print(tensor)
        print(w)
        print(s)
        print(sol)
        print((w - sol))
        exit()
        ctx.save_for_backward(s, sol, permutation_1d)
        return ctx.sign * (w - sol)

    @staticmethod
    def backward(ctx, grad_output):
        s, sol, permutation_1d = ctx.saved_tensors
        inv_permutation_1d = _inv_permutation(permutation_1d.unsqueeze(0)).squeeze(0)
        inv_permutation = inv_permutation_1d.unsqueeze(-1).expand(-1, s.shape[-1])
        if ctx.regularization == "l2":
            grad = isotonic_l2_backward[s.device.type](s, sol, grad_output)
        else:
            grad = isotonic_kl_backward[s.device.type](s, sol, grad_output)
        return grad.gather(0, inv_permutation), None, None, None

def soft_sort_by_column1(tensor, regularization="l2", regularization_strength=1.0, column=0):
    t_shape = tensor.shape
    t_2ds = tensor.view([-1, *t_shape[-2:]]) # [..., H, W] -> [N, H, W]
    t_2ds_sorted = torch.stack([SoftSortByColumn1.apply(t, regularization, regularization_strength, column) for t in t_2ds])
    return t_2ds_sorted.view(t_shape)

########################################################################################################################################################################
########################################################################################################################################################################

class SoftSortByColumn15(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, regularization="l2", regularization_strength=1.0, column=0):
        ctx.sign = -1
        ctx.regularization = regularization
        w = (_arange_like(tensor, reverse=True) + 1) / regularization_strength
        tensor = ctx.sign * tensor  # for ascending
        # idk I hope this works lol
        permutation_1d = torch.argsort(tensor[:,column], descending=True)
        permutation = permutation_1d.unsqueeze(-1).expand(-1, tensor.shape[-1])
        s = tensor.gather(0, permutation)

        # note reverse order of args
        if ctx.regularization == "l2":
            sol = isotonic_l2[s.device.type](w - s)
        else:
            sol = isotonic_kl[s.device.type](w, s)
        ctx.save_for_backward(s, sol, permutation_1d)
        return ctx.sign * (w - sol)

    @staticmethod
    def backward(ctx, grad_output):
        s, sol, permutation_1d = ctx.saved_tensors
        inv_permutation_1d = _inv_permutation(permutation_1d.unsqueeze(0)).squeeze(0)
        inv_permutation = inv_permutation_1d.unsqueeze(-1).expand(-1, s.shape[-1])
        if ctx.regularization == "l2":
            grad = isotonic_l2_backward[s.device.type](s, sol, grad_output)
        else:
            grad = isotonic_kl_backward[s.device.type](s, sol, grad_output)
        return grad.gather(0, inv_permutation), None, None, None

def soft_sort_by_column15(tensor, regularization="l2", regularization_strength=1.0, column=0):
    t_shape = tensor.shape
    t_2ds = tensor.view([-1, *t_shape[-2:]]) # [..., H, W] -> [N, H, W]
    t_2ds_sorted = torch.stack([SoftSortByColumn1.apply(t, regularization, regularization_strength, column) for t in t_2ds])
    return t_2ds_sorted.view(t_shape)

########################################################################################################################################################################
########################################################################################################################################################################

from einops import rearrange, repeat

class SoftSortByColumn2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, regularization="l2", regularization_strength=1.0, column=0):
        # Save original shape
        b, h, w = tensor.shape
        # Reshape original tensor [B, H, W] -> [B*H, W]
        tensor_reshaped = rearrange(tensor, "b h w -> (b h) w")

        ctx.sign = -1
        ctx.regularization = regularization
        w_ = (_arange_like(tensor_reshaped, reverse=True) + 1) / regularization_strength
        tensor = ctx.sign * tensor  # for ascending
        tensor_reshaped = ctx.sign * tensor_reshaped  # for ascending

        # Sort all items by given column [B, H]
        permutation = torch.argsort(tensor[...,column], descending=True)
        permutation += repeat(h*torch.arange(0, b, device=tensor.device), "b -> b h", h=h)
        # Reshape permutation [B, H] -> [B*H, W]
        permutation_reshaped = repeat(permutation, "b h -> (b h) w", w=w)
        # Sort reshaped tensor with reshaped permutation
        s_ = tensor_reshaped.gather(0, permutation_reshaped)

        # note reverse order of args
        if ctx.regularization == "l2":
            sol = isotonic_l2[s_.device.type](w_ - s_)
        else:
            sol = isotonic_kl[s_.device.type](w_, s_)
        ctx.save_for_backward(s_, sol, permutation)

        out = ctx.sign * (w_ - sol)
        out_correct_shape = rearrange(out, "(b h) w -> b h w", b=b)
        return out_correct_shape

    @staticmethod
    def backward(ctx, grad_output):
        b, h, w = grad_output.shape
        grad_output = rearrange(grad_output, "b h w -> (b h) w")
        s_, sol, permutation = ctx.saved_tensors

        # Inverse permutation
        # inv_permutation = _inv_permutation(permutation) # [B, H]
        permutation -= repeat(h*torch.arange(0, b, device=grad_output.device), "b -> b h", h=h)
        # inv_permutation = torch.argsort(permutation)
        inv_permutation = _inv_permutation(permutation)
        inv_permutation_reshaped = repeat(inv_permutation, "b h -> (b h) w", w=w)
        
        if ctx.regularization == "l2":
            grad = isotonic_l2_backward[s_.device.type](s_, sol, grad_output)
        else:
            grad = isotonic_kl_backward[s_.device.type](s_, sol, grad_output)

        # print(grad.shape)

        # Maybe needs to go other way around
        out = grad.gather(0, inv_permutation_reshaped)
        out_correct_shape = rearrange(out, "(b h) w -> b h w", b=b)

        return out_correct_shape, None, None, None

def soft_sort_by_column2(tensor, regularization="l2", regularization_strength=1.0, column=0):
    return SoftSortByColumn2.apply(tensor, regularization, regularization_strength, column)

########################################################################################################################################################################
########################################################################################################################################################################

class SoftSortByColumn3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, regularization="l2", regularization_strength=1.0, column=0):
        b, h, w = tensor.shape
        device = tensor.device

        # Reshape original tensor [B, H, W] -> [B*H, W]
        tensor_reshaped = tensor.view(-1, w)

        ctx.sign = -1
        ctx.regularization = regularization
        w_ = torch.arange(w, 0, -1, dtype=tensor.dtype, device=device).expand(b*h, w) / regularization_strength

        tensor = ctx.sign * tensor  # for ascending
        tensor_reshaped = ctx.sign * tensor_reshaped  # for ascending

        # Sort all items by given column [B, H]
        permutation = torch.argsort(tensor[...,column], descending=True)
        permutation += h*torch.arange(b, device=device).expand(h, b).T
        # Reshape permutation [B, H] -> [B*H, W]
        permutation_reshaped = permutation.view(b*h).expand(w, b*h).T
        # Sort reshaped tensor with reshaped permutation
        s_ = tensor_reshaped.gather(0, permutation_reshaped)

        # note reverse order of args
        if ctx.regularization == "l2":
            sol = isotonic_l2[device.type](w_ - s_)
        else:
            sol = isotonic_kl[device.type](w_, s_)
        ctx.save_for_backward(s_, sol, permutation)

        out = ctx.sign * (w_ - sol)
        out_correct_shape = out.view(b, h, w)
        return out_correct_shape

    @staticmethod
    def backward(ctx, grad_output):
        b, h, w = grad_output.shape
        device = grad_output.device

        # b h w -> (b h) w
        grad_output = grad_output.view(-1, w)
        s_, sol, permutation = ctx.saved_tensors

        # Inverse permutation
        permutation -= h*torch.arange(b, device=device).expand(h, b).T
        inv_permutation = torch.argsort(permutation)
        # Reshape permutation [B, H] -> [B*H, W]
        inv_permutation_reshaped = inv_permutation.view(b*h).expand(w, b*h).T
        
        if ctx.regularization == "l2":
            grad = isotonic_l2_backward[device.type](s_, sol, grad_output)
        else:
            grad = isotonic_kl_backward[device.type](s_, sol, grad_output)


        # Maybe needs to go other way around
        out = grad.gather(0, inv_permutation_reshaped)
        out_correct_shape = out.view(b, h, w)

        return out_correct_shape, None, None, None

def soft_sort_by_column3(tensor, regularization="l2", regularization_strength=1.0, column=0):
    return SoftSortByColumn3.apply(tensor, regularization, regularization_strength, column)

########################################################################################################################################################################
########################################################################################################################################################################

def soft_sort_by_column_dirty(tensor, regularization="l2", regularization_strength=1.0, column=0):
    out = []
    for t in tensor:
        h, w = t.shape
        sorted_column = soft_sort(t[:, column].unsqueeze(0)).squeeze(0)
        permutation = torch.argsort(sorted_column)
        permutation = permutation.unsqueeze(-1).expand(h, w)
        sorted_t = torch.gather(t, 0, permutation)
        sorted_t[:, column] = sorted_column
        out.append(sorted_t)
    return torch.stack(out)

########################################################################################################################################################################
########################################################################################################################################################################

def soft_sort_by_column_identity(tensor, regularization="l2", regularization_strength=1.0, column=0):
    # out = []
    # for t in tensor:
    #     h, w = t.shape
    #     sorted_column = soft_sort(t[:, column].unsqueeze(0)).squeeze(0)
    #     permutation = torch.argsort(sorted_column)
    #     permutation = permutation.unsqueeze(-1).expand(h, w)
    #     sorted_t = torch.gather(t, 0, permutation)
    #     sorted_t[:, column] = sorted_column
    #     out.append(sorted_t)
    # return torch.stack(out)
    return tensor

########################################################################################################################################################################
########################################################################################################################################################################

import time
from ascii_graph import Pyasciigraph

class SoftSortByColumn4(torch.autograd.Function):
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

def soft_sort_by_column4(values, regularization="l2", regularization_strength=1.0, column=0):
    return SoftSortByColumn4.apply(values, regularization, regularization_strength, column)



soft_sort_by_column = soft_sort_by_column4
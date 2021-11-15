import torch
from differentiable_sorter_torchsort import soft_sort_by_column1, soft_sort_by_column2, soft_sort_by_column3
from itertools import product

def test_output_equal(tensor, **kwargs):
    out1 = soft_sort_by_column1(tensor, **kwargs)
    out2 = soft_sort_by_column2(tensor, **kwargs)
    out3 = soft_sort_by_column3(tensor, **kwargs)

    return (out1 == out2).all() and (out2 == out3).all()

def test_backward_equal(tensor, **kwargs):
    t1 = tensor.clone()
    t1.requires_grad = True
    out1 = soft_sort_by_column1(t1, **kwargs)
    l1 = out1.mean()
    l1.backward()
    grad1 = t1.grad

    t2 = tensor.clone()
    t2.requires_grad = True
    out2 = soft_sort_by_column2(t2, **kwargs)
    l2 = out2.mean()
    l2.backward()
    grad2 = t2.grad

    t3 = tensor.clone()
    t3.requires_grad = True
    out3 = soft_sort_by_column3(t3, **kwargs)
    l3 = out3.mean()
    l3.backward()
    grad3 = t3.grad

    return (grad1 == grad2).all() and (grad2 == grad3).all()

test_tensors = [
    torch.rand([16, 4, 10]),
    torch.rand([128, 9, 3]),
    torch.rand([64, 2, 10]),
    1e-4*torch.rand([32, 2, 10]),
    1e4*torch.rand([32, 2, 10]),
]

columns = [0, 1, -1]

test_cases = list(product(test_tensors, columns))

print("Outputs")
for test_case in test_cases:
    tensor, column = test_case

    result = test_output_equal(tensor, column=column)

    print(f"tensor shape {tensor.shape}, column {column}, result {result}")

print("Gradients")
for test_case in test_cases:
    tensor, column = test_case

    result = test_backward_equal(tensor, column=column)

    print(f"tensor shape {tensor.shape}, column {column}, result {result}")
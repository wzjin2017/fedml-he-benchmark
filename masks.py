# random mask

original_dy_dx = list((_.detach().clone() for _ in dy_dx))

flat_original_dy_dx = flat_grad(original_dy_dx) #flat gradients to a vector

mask = torch.FloatTensor(flat_original_dy_dx.shape).uniform_() > 0.3 #mask 30%

flat_original_dy_dx = flat_original_dy_dx * mask



# mask by gradient

def get_top_k_mask(vector, p = 0.1):

    top_k_indices = torch.topk(vector, int(len(vector) * p), largest = True).indices

    Mask = torch.zeros(vector.shape) == 0
    Mask[top_k_indices] = False
    return Mask

original_dy_dx = list((_.detach().clone() for _ in dy_dx))
flat_original_dy_dx = flat_grad(original_dy_dx) #flat gradients to a vector

mask = get_top_k_mask(flat_original_dy_dx.abs(), 0.1)

new_mask = torch.ones(flat_original_dy_dx.shape)
new_mask[:len(mask)] = mask
mask = new_mask

flat_original_dy_dx = flat_original_dy_dx * mask


# mask by sensitivity


def calculate_sensitivity(matrix):
    return torch.norm(matrix / (matrix.max() - matrix.min())) / len(matrix.view(-1))**0.5

def calculate_sensitivity_vector_range(matrix):
    matrix_shape = matrix.shape
    if len(matrix_shape) > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    if len(matrix_shape) == 1:
        matrix = matrix.reshape(1, matrix.shape[0])
    print(matrix.max(dim = 1).values.view(-1, 1).shape)
    return torch.norm(matrix / (matrix.max(dim = 1).values.view(-1, 1) - matrix.min(dim = 1).values.view(-1, 1))) / len(matrix.view(-1))**0.5



sensitivity_each_element = []
#sensitivity_each_layer = []
for layer_grad in dy_dx:

    sensitivity_each_element_current_layer = torch.zeros_like(layer_grad.view(-1))
    grad_of_layer_grad_to_y = []
    count = 0
    for each_element_grad in layer_grad.view(-1):
            each_element_grad_to_y = torch.autograd.grad(each_element_grad, gt_onehot_label, retain_graph=True)[0][:,gt_label]
            sensitivity_each_element_current_layer[count] = each_element_grad_to_y
            grad_of_layer_grad_to_y.append(each_element_grad_to_y.numpy())
            count += 1
            if count % 10000 == 0:
                print(count)
    sensitivity_each_element.append(sensitivity_each_element_current_layer)

flat_sensitivity_each_element = flat_grad(sensitivity_each_element)



original_dy_dx = list((_.detach().clone() for _ in dy_dx))
flat_original_dy_dx = flat_grad(original_dy_dx) #flat gradients to a vector

mask = get_top_k_mask(flat_sensitivity_each_element.abs(), 0.1)

new_mask = torch.ones(flat_original_dy_dx.shape)
new_mask[:len(mask)] = mask
mask = new_mask

flat_original_dy_dx = flat_original_dy_dx * mask
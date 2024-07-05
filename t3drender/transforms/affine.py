
def build_transform_matrix(R=None, T=None):
    if R is not None:
        batch_size = R.shape[0]
        device = R.device
    elif T is not None:
        batch_size = T.shape[0]
        device = T.device
    else:
        raise ValueError('Either R or T must be provided.')
    matrix = torch.eye(4)[None].repeat(batch_size, 1, 1).to(device)
    if R is not None:
        matrix[:, :3, :3] = R
    if T is not None:
        matrix[:, :3, 3] = T
    return matrix


def matmul(*mats):
    res = mats[0]
    for mat in mats[1:]:
        res = torch.matmul(res, mat)
    return res


def inverse_transform(T_batch):
    assert T_batch.ndim == 3 and T_batch.shape[1:] == (4, 4), 'T_batch must be of shape (B, 4, 4)'
    
    R_batch = T_batch[:, :3, :3]
    t_batch = T_batch[:, :3, 3]
    
    R_inv_batch = R_batch.transpose(1, 2)
    
    t_inv_batch = -torch.bmm(R_inv_batch, t_batch.unsqueeze(-1)).squeeze(-1)
    
    T_inv_batch = torch.zeros_like(T_batch)
    T_inv_batch[:, :3, :3] = R_inv_batch
    T_inv_batch[:, :3, 3] = t_inv_batch
    T_inv_batch[:, 3, 3] = 1
    
    return T_inv_batch


def transform_points(points, matrix):
    """
    Transform points using a homogeneous transformation matrix.
    
    Args:
        points (torch.Tensor): Tensor of shape (B, V, 3) representing the points.
        transform_matrix (torch.Tensor): Tensor of shape (B, 4, 4) representing the transformation matrices.
    
    Returns:
        torch.Tensor: Transformed points of shape (B, V, 3).
    """
    B, V, _ = points.shape
    if matrix.shape[1:] == (3, 3):
        matrix = build_transform_matrix(R=matrix)
    assert matrix.shape[1:] == (4, 4)
    
    homogeneous_points = torch.cat([points, torch.ones(B, V, 1, device=points.device)], dim=-1)  # (B, V, 4)
    homogeneous_points = homogeneous_points.transpose(1, 2)  # (B, 4, V)
    
    transformed_points = torch.matmul(matrix, homogeneous_points)  # (B, 4, V)
    
    transformed_points = transformed_points.transpose(1, 2)  # (B, V, 4)
    transformed_points = transformed_points[:, :, :3]  # Keep only the x, y, z coordinates
    
    return transformed_points

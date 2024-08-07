import torch
import numpy

from typing import Union

from pytorch3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)


class Compose:

    def __init__(self, transforms: list):
        """Composes several transforms together. This transform does not
        support torchscript.

        Args:
            transforms (list): (list of transform functions)
        """
        self.transforms = transforms

    def __call__(self,
                 rotation: Union[torch.Tensor, numpy.ndarray],
                 convention: str = 'xyz',
                 **kwargs):
        convention = convention.lower()
        if not (set(convention) == set('xyz') and len(convention) == 3):
            raise ValueError(f'Invalid convention {convention}.')
        if isinstance(rotation, numpy.ndarray):
            data_type = 'numpy'
            rotation = torch.FloatTensor(rotation)
        elif isinstance(rotation, torch.Tensor):
            data_type = 'tensor'
        else:
            raise TypeError(
                'Type of rotation should be torch.Tensor or numpy.ndarray')
        for t in self.transforms:
            if 'convention' in t.__code__.co_varnames:
                rotation = t(rotation, convention.upper(), **kwargs)
            else:
                rotation = t(rotation, **kwargs)
        if data_type == 'numpy':
            rotation = rotation.detach().cpu().numpy()
        return rotation


def aa_to_rotmat(
    axis_angle: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """
    Convert axis_angle to rotation matrixs.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f'Invalid input axis angles shape f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix])
    return t(axis_angle)


def aa_to_quat(
    axis_angle: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """
    Convert axis_angle to quaternions.
    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis angles f{axis_angle.shape}.')
    t = Compose([axis_angle_to_quaternion])
    return t(axis_angle)


def ee_to_rotmat(euler_angle: Union[torch.Tensor, numpy.ndarray],
                 convention='xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert euler angle to rotation matrixs.

    Args:
        euler_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if euler_angle.shape[-1] != 3:
        raise ValueError(
            f'Invalid input euler angles shape f{euler_angle.shape}.')
    t = Compose([euler_angles_to_matrix])
    return t(euler_angle, convention.upper())


def rotmat_to_ee(
        matrix: Union[torch.Tensor, numpy.ndarray],
        convention: str = 'xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation matrixs to euler angle.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix shape f{matrix.shape}.')
    t = Compose([matrix_to_euler_angles])
    return t(matrix, convention.upper())


def rotmat_to_quat(
    matrix: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation matrixs to quaternions.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_quaternion])
    return t(matrix)


def rotmat_to_rot6d(
    matrix: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation matrixs to rotation 6d representations.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_rotation_6d])
    return t(matrix)


def quat_to_aa(
    quaternions: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert quaternions to axis angles.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions f{quaternions.shape}.')
    t = Compose([quaternion_to_axis_angle])
    return t(quaternions)


def quat_to_rotmat(
    quaternions: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert quaternions to rotation matrixs.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(
            f'Invalid input quaternions shape f{quaternions.shape}.')
    t = Compose([quaternion_to_matrix])
    return t(quaternions)


def rot6d_to_rotmat(
    rotation_6d: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation 6d representations to rotation matrixs.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.
    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3, 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix])
    return t(rotation_6d)


def aa_to_ee(axis_angle: Union[torch.Tensor, numpy.ndarray],
             convention: str = 'xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert axis angles to euler angle.

    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f'Invalid input axis_angle shape f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix, matrix_to_euler_angles])
    return t(axis_angle, convention)


def aa_to_rot6d(
    axis_angle: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert axis angles to rotation 6d representations.

    Args:
        axis_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input axis_angle f{axis_angle.shape}.')
    t = Compose([axis_angle_to_matrix, matrix_to_rotation_6d])
    return t(axis_angle)


def ee_to_aa(euler_angle: Union[torch.Tensor, numpy.ndarray],
             convention: str = 'xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert euler angles to axis angles.

    Args:
        euler_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if euler_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input euler_angle f{euler_angle.shape}.')
    t = Compose([
        euler_angles_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle
    ])
    return t(euler_angle, convention)


def ee_to_quat(euler_angle: Union[torch.Tensor, numpy.ndarray],
               convention='xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert euler angles to quaternions.

    Args:
        euler_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).
    """
    if euler_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input euler_angle f{euler_angle.shape}.')
    t = Compose([euler_angles_to_matrix, matrix_to_quaternion])
    return t(euler_angle, convention)


def ee_to_rot6d(euler_angle: Union[torch.Tensor, numpy.ndarray],
                convention='xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert euler angles to rotation 6d representation.

    Args:
        euler_angle (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if euler_angle.shape[-1] != 3:
        raise ValueError(f'Invalid input euler_angle f{euler_angle.shape}.')
    t = Compose([euler_angles_to_matrix, matrix_to_rotation_6d])
    return t(euler_angle, convention)


def rotmat_to_aa(
    matrix: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation matrixs to axis angles.

    Args:
        matrix (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 3, 3). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    t = Compose([matrix_to_quaternion, quaternion_to_axis_angle])
    return t(matrix)


def quat_to_ee(quaternions: Union[torch.Tensor, numpy.ndarray],
               convention: str = 'xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert quaternions to euler angles.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 4). ndim of input is unlimited.
        convention (str, optional): Convention string of three letters
                from {“x”, “y”, and “z”}. Defaults to 'xyz'.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions f{quaternions.shape}.')
    t = Compose([quaternion_to_matrix, matrix_to_euler_angles])
    return t(quaternions, convention)


def quat_to_rot6d(
    quaternions: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert quaternions to rotation 6d representations.

    Args:
        quaternions (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 4). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 6).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(f'Invalid input quaternions f{quaternions.shape}.')
    t = Compose([quaternion_to_matrix, matrix_to_rotation_6d])
    return t(quaternions)


def rot6d_to_aa(
    rotation_6d: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation 6d representations to axis angles.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([
        rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_axis_angle
    ])
    return t(rotation_6d)


def rot6d_to_ee(rotation_6d: Union[torch.Tensor, numpy.ndarray],
                convention: str = 'xyz') -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation 6d representations to euler angles.

    Args:
        rotation_6d (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 3).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f'Invalid input rotation_6d f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix, matrix_to_euler_angles])
    return t(rotation_6d, convention)


def rot6d_to_quat(
    rotation_6d: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    """Convert rotation 6d representations to quaternions.

    Args:
        rotation (Union[torch.Tensor, numpy.ndarray]): input shape
                should be (..., 6). ndim of input is unlimited.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: shape would be (..., 4).

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(
            f'Invalid input rotation_6d shape f{rotation_6d.shape}.')
    t = Compose([rotation_6d_to_matrix, matrix_to_quaternion])
    return t(rotation_6d)


def make_homegeneous_rotmat_batch(input: torch.Tensor) -> torch.Tensor:
    """Appends a row of [0,0,0,1] to a batch size x 3 x 4 Tensor.

    Parameters
    ----------
    :param input: A tensor of dimensions batch size x 3 x 4
    :return: A tensor batch size x 4 x 4 (appended with 0,0,0,1)
    """
    batch_size = input.shape[0]
    row_append = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float)
    row_append.requires_grad = False
    padded_tensor = torch.cat(
        [input, row_append.view(1, 1, 4).repeat(batch_size, 1, 1)], dim=1)
    return padded_tensor


def make_homegeneous_rotmat(input: torch.Tensor) -> torch.Tensor:
    """Appends a row of [0,0,0,1] to a 3 x 4 Tensor.

    Parameters
    ----------
    :param input: A tensor of dimensions 3 x 4
    :return: A tensor batch size x 4 x 4 (appended with 0,0,0,1)
    """
    row_append = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float)
    row_append.requires_grad = False
    padded_tensor = torch.cat(input, row_append, dim=1)
    return padded_tensor


def flip_rotation(pose_vector: Union[torch.Tensor, numpy.ndarray],
                  pose_format: str = 'aa'):
    assert pose_format in ['aa', 'rotmat', 'quat', 'rot6d']
    assert isinstance(
        pose_vector,
        (torch.Tensor,
         numpy.ndarray)), 'pose_vector must be either torch.Tensor or numpy.ndarray'
    if isinstance(pose_vector, numpy.ndarray):
        rotation = pose_vector.copy()
    elif isinstance(pose_vector, torch.Tensor):
        rotation = pose_vector.clone()
    ori_shape = rotation.shape
    if pose_format == 'aa':
        rotation = rotation.reshape(-1, 3)
        pose_vector = pose_vector.reshape(rotation.shape)
        rotation[:, 1:] = pose_vector[:, 1:] * -1
        return rotation.reshape(ori_shape)

    elif pose_format == 'rotmat':
        rotation = rotation.reshape(-1, 9)
        pose_vector = pose_vector.reshape(rotation.shape)
        rotation[:, [1, 2, 3, 6]] = pose_vector[:, [1, 2, 3, 6]] * -1
        return rotation.reshape(ori_shape)

    elif pose_format == 'quat':
        rotation = rotation.reshape(-1, 4)
        pose_vector = pose_vector.reshape(rotation.shape)
        rotation[:, [2, 3]] = pose_vector[:, [2, 3]] * -1
        return rotation.reshape(ori_shape)

    elif pose_format == 'rot6d':
        rotation = rotation.reshape(-1, 6)
        pose_vector = pose_vector.reshape(rotation.shape)
        rotation[:, [1, 2, 3]] = pose_vector[:, [1, 2, 3]] * -1
        return rotation.reshape(ori_shape)
    else:
        raise ValueError(f'Unknown rotation format: {type(pose_format)}')

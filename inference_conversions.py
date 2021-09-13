import numpy as np

from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Transform, Pose

def transform_to_matrix4x4(tf:Transform) -> np.ndarray:
    m_ = np.empty((4,4)); m_[3,:] = ( 0, 0, 0, 1 )    
    m_[0,3], m_[1,3], m_[2,3] = tf.translation.x, tf.translation.y, tf.translation.z
    m_[:3,:3] = Rotation.from_quat(( tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w )).as_matrix()
    
    return m_

def translation_and_rotation_to_matrix4x4(translation:np.ndarray, rotation:Rotation) -> np.ndarray:
    m_ = np.empty([4,4]); m_[3,:] = ( 0, 0, 0, 1 )
    m_[:3,3], m_[:3,:3] = translation, rotation.as_matrix()

    return m_

def translation_and_rotation_to_transform_msg(translation:np.ndarray, rotation:Rotation) -> Transform:
    tf_msg_ = Transform()

    tf_msg_.translation.x = translation[0]
    tf_msg_.translation.y = translation[1]
    tf_msg_.translation.z = translation[2]

    quat_ = rotation.as_quat()
    tf_msg_.rotation.x = quat_[0]
    tf_msg_.rotation.y = quat_[1]
    tf_msg_.rotation.z = quat_[2]
    tf_msg_.rotation.w = quat_[3]

    return tf_msg_

def matrix4x4_to_pose_msg(m:np.ndarray) -> Pose:
    pose_msg_ = Pose()

    pose_msg_.position.x = m[0,3]
    pose_msg_.position.y = m[1,3]
    pose_msg_.position.z = m[2,3]
    
    quat_ = Rotation.from_matrix(m[:3,:3]).as_quat()
    pose_msg_.orientation.x = quat_[0]
    pose_msg_.orientation.y = quat_[1]
    pose_msg_.orientation.z = quat_[2]
    pose_msg_.orientation.w = quat_[3]
    
    return pose_msg_
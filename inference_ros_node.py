import sys, json
import numpy as np
from inference_helper_fns import *
from inference_webcam import build_model_and_load_weights, preprocess, postprocess
from scipy.spatial.transform import Rotation
from typing import Dict

import rclpy, tf2_ros
from std_msgs.msg import Header
from inference_conversions import *
from geometry_msgs.msg import PoseStamped, TransformStamped



class InferenceNode(rclpy.node.Node):
    def __init__(self, args:dict):
        super().__init__('efficientPose_'+args['camera_frame'])
        self._header = Header(frame_id=args['camera_frame'])
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        if 'publish' in args:
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
            self._pose_publisher = self.create_publisher(PoseStamped, args['publish']['topic'], 10)
            self._pose_frame_id = args['publish']['frame_id']
        else: self._pose_publisher = None
        
        with open(args['object_index_map'], 'r') as f:
            self._object_index_map = { int(k):v for k,v in json.load(f).items() }
            self._num_classes = len(self._object_index_map)

        self._camera_matrix, self._score_threshold, self._translation_scale_norm = args['camera_matrix'], args['score_threshold'], args['translation_scale_norm']
        self._model, self._image_size = build_model_and_load_weights(args['phi'], self._num_classes, self._score_threshold, args['weights'])

    def _predict(self, image:np.ndarray, zup:bool=True) -> Dict[str,tuple]:
        input_list, scale = preprocess(image, self._image_size, self._camera_matrix, self._translation_scale_norm)
        boxes, scores, labels, rotations, translations = self._model.predict_on_batch(input_list)
        boxes, scores, labels, rotations, translations = postprocess(boxes, scores, labels, rotations, translations, scale, self._score_threshold)

        out_ = {}
        for box, score, label, rotation, translation in zip(boxes, scores, labels, rotations, translations):
            if zup:
                r_ = Rotation.from_rotvec([ rotation[2], -rotation[0], -rotation[1] ])
                t_ = np.array([ translation[2], -translation[0], -translation[1] ])
            else: r_, t_ = Rotation.from_rotvec(rotation), translation

            out_[self._object_index_map[label]] = ( r_, t_, score, box )

        return out_

    def _publish(self, predictions:dict, clock_offset:float=0.0) -> None:
        clock_s_, clock_ns_ = self.get_clock().now().seconds_nanoseconds()
        clock_fs_ = (clock_s_ + (clock_ns_ / 1e9)) + clock_offset
        self._header.stamp.sec = int(clock_fs_)
        self._header.stamp.nanosec = int((clock_fs_ - self._header.stamp.sec) * 1e9)

        self._broadcast(predictions)
        
        if self._pose_publisher is not None:
            camera_stf_ = self._tf_buffer.lookup_transform(self._pose_frame_id, self._header.frame_id, self._header.stamp)
            camera_ext_mat_ = transform_to_matrix4x4(camera_stf_.transform)
            for label, pose_data in predictions.items():
                object_local_pose_mat_ = translation_and_rotation_to_matrix4x4(pose_data[1], pose_data[0])
                pose_msg_ = matrix4x4_to_pose_msg(np.matmul(camera_ext_mat_, object_local_pose_mat_))
                self._pose_publisher.publish(PoseStamped(header=Header(stamp=self._header.stamp, frame_id=label), pose=pose_msg_))

    def _broadcast(self, predictions:dict) -> None:
        stfs_ = []
        for label, pose_data in predictions.items():
            transform_msg_ = translation_and_rotation_to_transform_msg(pose_data[1], pose_data[0])
            stf_ = TransformStamped(header=self._stf_header, child_frame_id=label, transform=transform_msg_)
            stfs_.append(stf_)

        if len(stfs_): self._tf_broadcaster.sendTransform(stfs_)
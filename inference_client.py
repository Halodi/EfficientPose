import json, tempfile, os
import numpy as np
from time import perf_counter, sleep
from inference_helper_fns import load_object_index_map
from inference_conversions import *
from inference_filter import FilterMultiLabels

import rclpy, tf2_ros
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseArray



class InferenceClient:
    def __init__(self, args_fp:str, width:int, height:int, camera_matrix:np.ndarray):
        with open(args_fp, 'r') as f:
            args_ = json.load(f)
            args_['_width'], args_['_height'], args_['_camera_matrix'] = width, height, camera_matrix.ravel().tolist()

        self._object_index_map = load_object_index_map(args_['object_index_map'])
        self._timeout = args_['recv_timeout']
        self._translation_scale_norm = args_['translation_scale_norm']

        pid_ = str(os.getpid())
        args_['_image_file'] = os.path.join(tempfile.gettempdir(), pid_+'_input_buffer.dat')
        args_['_detections_file'] = os.path.join(tempfile.gettempdir(), pid_+'_output_buffer.dat')
        self._input_buffer = np.memmap(args_['_image_file'], mode='w+', dtype=np.uint8, shape=( 1, 1 + (height * width * 3) ))
        self._input_buffer_counter = np.memmap(args_['_image_file'], mode='r+', dtype=np.uint8, shape=( 1, 1 ))
        self._detections_buffer = np.memmap(args_['_detections_file'], mode='w+', dtype=float, shape=( 1, 1 + (args_['max_detections'] * 8) ))
        self._last_recv_value, self._det_shape, self._detections = self._detections_buffer[0,0], ( args_['max_detections'], 8 ), []

        args_fp_out_ = args_fp[:args_fp.rfind('.')] + '_extended.json'
        with open(args_fp_out_, 'w') as f: json.dump(args_, f, indent=2)

        self._filters = FilterMultiLabels(args_['filter'], self._object_index_map.values()) if 'filter' in args_ else None
        self._init_rclpy_comms(args_['ros'])        

    def _init_rclpy_comms(self, args:dict):
        self._node = rclpy.create_node("efficientpose_" + str(os.getpid()), namespace=args.get('namespace', None))
        self._camera_frame, self._publish_frame, self._camera_ext_mat = args['camera_frame'], args['publish_frame'], np.eye(4)
        if self._camera_frame != self._publish_frame:
            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)
        else: self._tf_buffer = None

        topic_base_str_ = "efficientpose_"+args['camera_frame']
        self._pub_raw    = self._node.create_publisher(PoseArray, topic_base_str_+'_raw', 10)
        self._pub_filtered = self._node.create_publisher(PoseArray, topic_base_str_+'_filtered', 10) if self._filters is not None else None

    def _send(self, img:np.ndarray) -> None:
        self._input_buffer[0,1:] = img.ravel()
        self._input_buffer.flush()
        self._input_buffer_counter[0,0] += 1
        self._input_buffer_counter.flush()

    def _wait_for_server(self):
        while True:
            if self._detections_buffer[0,0] != self._last_recv_value:
                self._last_recv_value = self._detections_buffer[0,0]
                return
            else: sleep(1.0)

    def _recv(self) -> bool:
        t0_ = perf_counter()

        while True:
            if self._timeout > 0.0 and (perf_counter() - t0_) >= self._timeout: return False

            if self._detections_buffer[0,0] != self._last_recv_value:
                self._last_recv_value = self._detections_buffer[0,0]
                detections_array_ = np.reshape(self._detections_buffer[0,1:], self._det_shape)

                self._detections = []
                for detection in detections_array_:
                    label_ = self._object_index_map.get(int(detection[0]))
                    if label_ is None: continue
                    r_ = Rotation.from_rotvec([ detection[4], -detection[2], -detection[3] ])
                    t_ = np.asarray([ detection[7], -detection[5], -detection[6] ]) / self._translation_scale_norm
                    self._detections.append(( label_, detection[1], r_, t_ ))

                return True

    def _publish(self, clock_offset:float):
        s_, ns_ = self._node.get_clock().now().seconds_nanoseconds()
        fs_ = (s_ + (ns_ / 1e9)) + clock_offset
        s_adj_ = int(fs_); ns_adj_ = int((fs_ - s_adj_) * 1e9)
        stamp_ = Time(sec=s_adj_, nanosec=ns_adj_)

        if self._tf_buffer is not None:
            camera_stf_ = self._tf_buffer.lookup_transform(self._publish_frame, self._camera_frame, stamp_)
            self._camera_ext_mat = transform_to_matrix4x4(camera_stf_.transform)

        pose_arrays_ = { k:[] for k in self._object_index_map.values() }
        filter_dict_ = { k:[] for k in self._object_index_map.values() }
        for label, score, rotation, translation in self._detections:
            m_local_ = translation_and_rotation_to_matrix4x4(translation, rotation)
            m_wrt_output_frame_ = np.matmul(self._camera_ext_mat, m_local_)
            pose_arrays_[label].append(matrix4x4_to_pose_msg(m_wrt_output_frame_))
            filter_dict_[label].append(m_wrt_output_frame_)

        for label, pose_array in pose_arrays_.items():
            self._pub_raw.publish(PoseArray(header=Header(frame_id=label, stamp=stamp_), poses=pose_array))

        if self._filters is not None:
            self._filters.step(filter_dict_)
            for label, M in self._filters.mean().items():
                pose_array_ = [ matrix4x4_to_pose_msg(m) for m in M ]
                self._pub_filtered.publish(PoseArray(header=Header(frame_id=label, stamp=stamp_), poses=pose_array_))



if __name__ == '__main__':

    import sys
    from time import sleep

    w_, h_ = 3, 3

    ic_ = InferenceClient(sys.argv[1], w_, h_, np.eye(3))

    while True:
        input_ = (np.random.random_sample([h_,w_,3]) * 255).astype(np.uint8)
        ic_._send(input_)

        print(input_)
        sleep(1.0)
from inference_client import InferenceClient
from inference_helper_fns import generate_camera_matrix

from os import getpid
import rclpy

import pyzed.sl as sl
import numpy as np
from time import sleep

from tf2_ros.transform_broadcaster import TransformBroadcaster



def open_zed(stream_address:str=None):
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    if stream_address is not None:
        colon_idx_ = stream_address.rfind(':')
        init_params.set_from_stream(stream_address[:colon_idx_], int(stream_address[colon_idx_+1:]))
    else:
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.camera_resolution = sl.RESOLUTION.HD1080

    # Open the camera
    zed = sl.Camera()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS: return None, None

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD

    return zed, runtime_parameters

def get_zed_fps(zed:sl.Camera, runtime_parameters:sl.RuntimeParameters, n=10) -> float:
    fps_array_ = np.empty(n, dtype=float)
    for i in range(n):
        zed.grab(runtime_parameters)
        fps_array_[i] = zed.get_current_fps()

    return np.mean(fps_array_)



class InferenceClientZed(InferenceClient):
    def __init__(self, args_fp:str, zed_stream_address:str=None):
        self._node = rclpy.create_node("efficientPoseZed_" + str(getpid()))
        self._zed, self._zed_runtime_parameters = open_zed(zed_stream_address)
        if self._zed is None:
            self._node.get_logger().error("Unable to connect to ZED")
            return
        else: self._node.get_logger().info("Connected to ZED")

        calib_ = self._zed.get_camera_information().calibration_parameters.left_cam
        w_, h_ = calib_.image_size.width, calib_.image_size.height
        camera_matrix_ = generate_camera_matrix(calib_.fx, calib_.fy, calib_.cx, calib_.cy)
        super().__init__(args_fp, w_, h_, camera_matrix_)

        self._nominal_frame_period = 1.0 / get_zed_fps(self._zed, self._zed_runtime_parameters)
        self._latest_frame_processed = True
        self._zed_left_buffer = sl.Mat()

    @property
    def zed_ok(self): return self._zed is not None

    def _zed_cb(self):
        if not self._latest_frame_processed:
            #self._node.get_logger().info("Waiting on inference results from previous ZED capture ...")
            sleep(self._nominal_frame_period)
            return

        if self._zed.grab(self._zed_runtime_parameters) != sl.ERROR_CODE.SUCCESS:
            self._node.get_logger().error("Failed to get ZED capture")
            return

        self._zed.retrieve_image(self._zed_left_buffer, sl.VIEW.LEFT)
        l_bgr_ = self._zed_left_buffer.get_data()[:,:,:3]
        self._send(l_bgr_)

        self._latest_frame_processed = False

    def spin(self):
        self._node.get_logger().info("Waiting for EfficientPose server ...")
        self._wait_for_server()
        self._node.create_timer(self._nominal_frame_period, self._zed_cb)
        self._node.get_logger().info("ZED callback started")

        while rclpy.ok():
            rclpy.spin_once(self._node)

            if self._recv():
                print(len(self._detections))

                '''image_ns_ = self._zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                current_ns_ = self._zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds()        
                time_offset_fs_ = (image_ns_ - current_ns_) / 1e9'''

                self._latest_frame_processed = True

        self._node.destroy_node()



def main(rclpy_args=None):
    import sys

    rclpy.init(args=rclpy_args)

    z_addr_ = sys.argv[2] if ':' in sys.argv[2] else None
    z_ = InferenceClientZed(sys.argv[1], z_addr_)
    if z_.zed_ok: z_.spin()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
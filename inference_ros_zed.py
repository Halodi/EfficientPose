import rclpy
from inference_ros_node import load_args_dict, generate_camera_matrix, InferenceNode
import pyzed.sl as sl



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

class DetectionNode(InferenceNode):
    def __init__(self, args:dict):
        self._zed, self._zed_runtime_parameters = open_zed(args.get('zed_stream_address'))
        if self._zed is None: return

        calib_ = self._zed.get_camera_information().calibration_parameters.left_cam
        args['camera_matrix'] = generate_camera_matrix(calib_.fx, calib_.fy, calib_.cx, calib_.cy)
        super().__init__(args)

        self._left = sl.Mat()
        self._timer = self.create_timer(0.02, self.predict)

    def predict(self):
        if self._zed.grab(self._zed_runtime_parameters) != sl.ERROR_CODE.SUCCESS:
            self.get_logger().error("Failed to get ZED capture")
            return

        image_ns_ = self._zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        current_ns_ = self._zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_nanoseconds()        
        time_offset_fs_ = (image_ns_ - current_ns_) / 1e9

        self._zed.retrieve_image(self._left, sl.VIEW.LEFT)
        predict_l_ = self._predict(self._left.get_data()[:,:,:3])
        self._publish(predict_l_, time_offset_fs_)



def main(rclpy_args=None):
    rclpy.init(args=rclpy_args)
    rclpy.spin(DetectionNode(load_args_dict()))
    rclpy.shutdown()

if __name__ == '__main__':
    main()

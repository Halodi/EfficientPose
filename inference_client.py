import json, tempfile, os
import numpy as np
from time import perf_counter

class InferenceClient:
    def __init__(self, args_fp:str, width:int, height:int, camera_matrix:np.ndarray) -> None:
        with open(args_fp, 'r') as f:
            args_ = json.load(f)
            args_['_width'], args_['_height'], args_['_camera_matrix'] = width, height, camera_matrix.ravel().tolist()

        pid_ = str(os.getpid())
        args_['_image_file'] = os.path.join(tempfile.gettempdir(), pid_+'_input_buffer.dat')
        args_['_detections_file'] = os.path.join(tempfile.gettempdir(), pid_+'_output_buffer.dat')

        self._input_buffer = np.memmap(args_['_image_file'], mode='w+', dtype=np.uint8, shape=( 1, 1 + (height * width * 3) ))
        self._detections_buffer = np.memmap(args_['_detections_file'], mode='w+', dtype=float, shape=( 1, 1 + (args_['max_detections'] * 8) ))
        self._last_recv_value, self._det_shape, self._detections = self._detections_buffer[0,0], ( args_['max_detections'], 8 ), None

        args_fp_out_ = args_fp[:args_fp.rfind('.')] + '_extended.json'
        with open(args_fp_out_, 'w') as f: json.dump(args_, f)

    def _send(self, img:np.ndarray) -> None:
        self._input_buffer[0,-1] += 1
        self._input_buffer[0,1:] = img.ravel()
        self._input_buffer.flush()

    def _recv(self, timeout=0.0) -> bool:
        t0_ = perf_counter()

        while True:
            if (perf_counter() - t0_) >= timeout and timeout > 0.0: return False

            if self._detections_buffer[0,0] != self._last_recv_value:
                self._last_recv_value = self._detections_buffer[0,0]
                self._detections = np.reshape(self._detections_buffer[0,1:], self._det_shape)
                return True
import json, tempfile, os
import numpy as np
from time import perf_counter, sleep
from inference_helper_fns import load_object_index_map

class InferenceClient:
    def __init__(self, args_fp:str, width:int, height:int, camera_matrix:np.ndarray):
        with open(args_fp, 'r') as f:
            args_ = json.load(f)
            args_['_width'], args_['_height'], args_['_camera_matrix'] = width, height, camera_matrix.ravel().tolist()

        self._object_index_map = load_object_index_map(args_['object_index_map'])
        self._timeout = args_['recv_timeout']

        pid_ = str(os.getpid())
        args_['_image_file'] = os.path.join(tempfile.gettempdir(), pid_+'_input_buffer.dat')
        args_['_detections_file'] = os.path.join(tempfile.gettempdir(), pid_+'_output_buffer.dat')
        self._input_buffer = np.memmap(args_['_image_file'], mode='w+', dtype=np.uint8, shape=( 1, 1 + (height * width * 3) ))
        self._input_buffer_counter = np.memmap(args_['_image_file'], mode='r+', dtype=np.uint8, shape=( 1, 1 ))
        self._detections_buffer = np.memmap(args_['_detections_file'], mode='w+', dtype=float, shape=( 1, 1 + (args_['max_detections'] * 8) ))
        self._last_recv_value, self._det_shape, self._detections = self._detections_buffer[0,0], ( args_['max_detections'], 8 ), {}

        args_fp_out_ = args_fp[:args_fp.rfind('.')] + '_extended.json'
        with open(args_fp_out_, 'w') as f: json.dump(args_, f, indent=2)

    def _send(self, img:np.ndarray) -> None:
        self._input_buffer[0,1:] = img.ravel()
        self._input_buffer.flush()
        self._input_buffer_counter[0,0] += 1
        self._input_buffer_counter.flush()

    def _recv(self) -> bool:
        t0_ = perf_counter()

        while True:
            if self._timeout > 0.0 and (perf_counter() - t0_) >= self._timeout: return False

            if self._detections_buffer[0,0] != self._last_recv_value:
                self._last_recv_value = self._detections_buffer[0,0]
                detections_array_ = np.reshape(self._detections_buffer[0,1:], self._det_shape)

                self._detections.clear()
                for detection in detections_array_:
                    label_ = self._object_index_map.get(int(detection[0]))
                    if label_ is None: continue
                    score_, r_, t_ = detection[1], np.array(detection[2:5]), np.array(detection[5:8])
                    self._detections[label_] = ( score_, r_, t_ )

                return True

    def _wait_for_server(self):
        while True:
            if self._detections_buffer[0,0] != self._last_recv_value:
                self._last_recv_value = self._detections_buffer[0,0]
                return
            else: sleep(1.0)

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
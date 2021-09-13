import json
import numpy as np
from inference_webcam import build_model_and_load_weights, preprocess, postprocess
from inference_helper_fns import load_object_index_map
from time import perf_counter

class InferenceServer:
    def __init__(self, args:dict, load_model:bool=True) -> None:
        if load_model:
            oim_ = load_object_index_map(args['object_index_map'])
            self._camera_matrix, self._score_threshold, self._translation_scale_norm = args['_camera_matrix'], args['score_threshold'], args['translation_scale_norm']
            self._model, self._image_size = build_model_and_load_weights(args['phi'], len(oim_), self._score_threshold, args['weights'])
        else: self._model = None

        self._input_shape, self._input, self._timeout = ( args['_height'], args['_width'], 3 ), None, args['recv_timeout']
        self._input_buffer = np.memmap(args['_image_file'], mode='r', dtype=np.uint8, shape=( 1, 1 + (args['_height'] * args['_width'] * 3) ))
        self._detections_buffer = np.memmap(args['_detections_file'], mode='r+', dtype=float, shape=( 1, 1 + (args['max_detections'] * 8) ))
        self._detections_buffer_counter = np.memmap(args['_detections_file'], mode='r+', dtype=float, shape=( 1, 1 ))
        self._last_recv_value, self._detections = self._input_buffer[0,0], np.zeros(( args['max_detections'], 8 ), float)

    def loop(self) -> None:
        while True:
            if not self._recv(): break

            self._detections[:,0] = -1

            if self._model is not None:
                input_list, scale = preprocess(self._input, self._image_size, self._camera_matrix, self._translation_scale_norm)
                boxes, scores, labels, rotations, translations = self._model.predict_on_batch(input_list)
                boxes, scores, labels, rotations, translations = postprocess(boxes, scores, labels, rotations, translations, scale, self._score_threshold)

                wI_ = 0
                for i in np.argsort(scores)[::-1][:self._detections.shape[0]]:
                    self._detections[wI_,0], self._detections[wI_,1] = labels[i], scores[i]
                    self._detections[wI_,2:5], self._detections[wI_,5:8] = rotations[i], translations[i]                
                    wI_ += 1
            else: print("EfficientPose inference server: Received data, but model was not loaded")

            self._detections_buffer[0,1:] = self._detections.ravel()
            self._detections_buffer.flush()
            self._detections_buffer_counter[0,0] += 1.0
            self._detections_buffer_counter.flush()

    def _recv(self) -> bool:
        t0_ = perf_counter()

        while True:
            if self._timeout > 0.0 and (perf_counter() - t0_) >= self._timeout: return False

            if self._input_buffer[0,0] != self._last_recv_value:
                self._last_recv_value = self._input_buffer[0,0]
                self._input = np.reshape(self._input_buffer[0,1:], self._input_shape)
                return True



if __name__ == '__main__':

    import sys

    with open(sys.argv[1], 'r') as f:
        args_ = json.load(f)

    load_model_ = ('t' in sys.argv[2] or 'T' in sys.argv[2]) if len(sys.argv) > 2 else True

    is_ = InferenceServer(args_, load_model_)
    is_.loop()
import time
import numpy as np
import cv2
from pose_engine import PoseEngine
import json
import zmq



EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)


def main(engine, cap, mq):
    input_shape = engine.get_input_tensor_shape()
    flags = 0

    while True:
        # Read image from camera
        _, img = cap.read()
        img = np.flipud(img)
        cam_width = np.shape(img)[1]
        cam_height = np.shape(img)[0]
        resized_img = cv2.resize(img, (input_shape[2], input_shape[1]), interpolation = cv2.INTER_CUBIC)
        input_tensor = np.asarray(resized_img).flatten()

        # Run inference
        output = engine.run_inference(input_tensor)
        persons, inference_time = engine.ParseOutput(output)
        for pose in persons:
            if pose.score < 0.4: continue # Skip person
            ps = []
            for label, keypoint in pose.keypoints.items():
                p = {"l": label, "x": int(keypoint.yx[1]), "y": int(keypoint.yx[0])}
                ps.append(p)
            rc = mq.send_json(ps, flags|zmq.SNDMORE)
            print(rc, pose.score)
            break # Just first person


if __name__ == '__main__':

    # Initialize ZeroMQ
    uri = "tcp://192.168.1.134:5555"
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    print("Connecting to {}".format(uri))
    while True:
        rc = socket.connect(uri)
        if rc != None:
            break
        else:
            time.sleep(5.0)

    # Initialize engine.
    model = 'models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'
    print('Loading model: ', model)
    engine = PoseEngine(model)

    # Initialize camera
    cam = cv2.VideoCapture(0)


    try:
        main(engine, cam, socket)
    except KeyboardInterrupt:
        print("Terminating on request. Closing module")
        cam.release
    except Exception as ex:
        print("Closing module")
        cam.release
        raise ex

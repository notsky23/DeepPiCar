"""A demo to classify Raspberry Pi camera stream."""
import argparse
import time
import datetime
import traceback

import numpy as np
import os

import cv2
from PIL import Image

# Import pycoral instead of edgetpu
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter

def main():
    os.chdir('/home/notsky/DeepPiCar/models/object_detection')
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=False)
    parser.add_argument(
      '--label', help='File path of label file.', required=False)
    args = parser.parse_args()
    
    args.model = 'data/model_result/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    args.label = 'data/model_result/coco_labels.txt'
        
    with open(args.label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    # initialize open cv
    IM_WIDTH = 640
    IM_HEIGHT = 480
    camera = cv2.VideoCapture(0)
    ret = camera.set(3,IM_WIDTH)
    ret = camera.set(4,IM_HEIGHT)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,IM_HEIGHT-10)
    fontScale = 1
    fontColor = (255,255,255)  # white
    boxColor = (0,0,255)   # RED?
    boxLineWidth = 1
    lineType = 2
    
    annotate_text = ""
    annotate_text_time = time.time()
    time_to_show_prediction = 1.0 # ms
    min_confidence = 0.20

    # Create pycoral interpreter instance
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    elapsed_ms = 0
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (IM_WIDTH,IM_HEIGHT))

    try:
        while camera.isOpened():
            try:
                start_ms = time.time()
                ret, frame = camera.read() # grab a frame from camera

                if ret == False :
                    print('can NOT read from camera')
                    break
                
                frame_expanded = np.expand_dims(frame, axis=0)
                
                ret, img = camera.read()
                input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB color space
                img_pil = Image.fromarray(input).resize((width, height), Image.ANTIALIAS)

                # input_size = common.input_size(interpreter)
                # img_pil = img_pil.resize(input_size, Image.ANTIALIAS)

                # Run inference using pycoral
                start_tf_ms = time.time()
                common.set_input(interpreter, img_pil)
                interpreter.invoke()
                results = detect.get_objects(interpreter, score_threshold=min_confidence)
                results = results[:5]
                end_tf_ms = time.time()
                elapsed_tf_ms = end_tf_ms - start_ms

                # Calculate scaling factors
                scale_x = IM_WIDTH / width
                scale_y = IM_HEIGHT / height

                if results :
                    for obj in results:

                        print("%s, %.0f%% %s %.2fms" % (labels[obj.id], obj.score * 100, obj.bbox, elapsed_tf_ms * 1000))
                        box = obj.bbox
                        coord_top_left = (int(box[0] * scale_x), int(box[1] * scale_y))
                        coord_bottom_right = (int(box[2] * scale_x), int(box[3] * scale_y))
                        cv2.rectangle(img, coord_top_left, coord_bottom_right, boxColor, boxLineWidth)
                        annotate_text = "%s, %.0f%%" % (labels[obj.id], obj.score * 100)
                        coord_top_left = (coord_top_left[0], coord_top_left[1] + 15)
                        cv2.putText(img, annotate_text, coord_top_left, font, fontScale, boxColor, lineType)
                    print('------')
                else:
                    print('No object detected')

                # Print Frame rate info
                elapsed_ms = time.time() - start_ms
                annotate_text = "%.2f FPS, %.2fms total, %.2fms in tf " % (1.0 / elapsed_ms, elapsed_ms * 1000, elapsed_tf_ms * 1000)
                print('%s: %s' % (datetime.datetime.now(), annotate_text))
                cv2.putText(img, annotate_text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                
                out.write(img)
                    
                cv2.imshow('Detected Objects', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                # catch it and don't exit the while loop
                print('In except')
                traceback.print_exc()

    finally:
        print('In Finally')
        camera.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

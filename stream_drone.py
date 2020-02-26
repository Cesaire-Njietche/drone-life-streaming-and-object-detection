import argparse
import cv2
import numpy as np
from random import randint
from Network import Network
#from imutils.video.webcamvideostream import WebcamVideoStream as vc
#import paho.mqtt.client as mqtt
import sys
import os

cv2.namedWindow("Kea AI", cv2.WINDOW_NORMAL)

INPUT_STREAM = "rtspsrc location=rtsp://192.168.53.1/live latency=10 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink max-buffers=1 drop=true"

Detection_MODEL = "/home/katelynn/Desktop/Kea_Live/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml"
CPU_EXTENSION = None


# MQTT server environment variables
'''
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001 ### TODO: Set the Port for MQTT
MQTT_KEEPALIVE_INTERVAL = 60
'''

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    c_desc = "The location of the CPU extension file"
    m_desc = "The location of the model xml file"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    parser.add_argument("-d", help=d_desc, default='CPU')
    parser.add_argument("-c", help=c_desc, default=CPU_EXTENSION)
    parser.add_argument("-m", help=m_desc, default=Detection_MODEL)
    args = parser.parse_args()

    return args



def infer_on_video(args):
    # Initialize the Inference Engine
    plugin = Network()

    # Load the network model into the IE
    plugin.load_model(args.m, args.d, args.c)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    #print(cv2. getBuildInformation()) cv2.CAP_GSTREAMER
    cap = cv2.VideoCapture(args.i)

    #cap = vc(args.i)

    print("Good...")
    cap.open(args.i) #args.i was here

    # Grab the shape of the input s
    width = int(cap.get(3))
    height = int(cap.get(4))
    #out = cv2.VideoWriter('out1.mp4', 0x00000021, 30, (width,height))
    # Process frames until the video ends, or process is exited


    nber_frame = 0
    while cap.isOpened():
        # Read the next frame
        ret, frame = cap.read()
        #height, width = frame[:2]
        nber_frame += 1
        #if not flag:
            #break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # Perform inference on the frame
        
        plugin.async_inference(p_frame)

        # Get the output of inference
        if plugin.wait() == 0:
            output = plugin.extract_output()
           
            output = output.squeeze(0).squeeze(0)
            nber_person = 0
            for o in output:
                conf = o[2]
                if conf >= .5 and o[1] == 1 :
                    bx = o[3]
                    by = o[4]
                    bw = o[5]
                    bh = o[6]
                    bx = int(bx*width)
                    by = int(by*height)
                    bw = int(bw*width)
                    bh = int(bh*height)
                    nber_person +=1
                    cv2.rectangle(frame, (bx, by), (bw, bh), (243, 69, 18), thickness=2)
   
            

        ### TODO: Send frame to the ffmpeg server
            frame = cv2.putText(frame,'Number of people is '+str(nber_person),(400,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(241, 214, 18),lineType=cv2.LINE_AA)
            #out.write(frame)
            
        cv2.imshow('Kea AI', frame)
            #cv2.imwrite("outputs/{}-output.png".format(args.t), output_image)
            #sys.stdout.buffer.write(frame)
            #sys.stdout.flush()
            

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    #out.release()
    cap.release()
    cv2.destroyAllWindows()
    

def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()

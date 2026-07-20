'''
    recorder.py =>  program utk record data dalam video. tidak ada proses apapun.
                    hanya record saja.
    Run: python3 recorder.py
'''

import numpy as np
import cv2
import random as rnd
import time
from datetime import datetime
import os

def FName():
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H%M%S")
    return date_time_str 

def hitungbatas(a, max):
    s = (max-a)/2
    return int(s)

def main(source, name):
    specialname = ''
    DirName = ''
    i = 1
    # Open a connection to the webcam
    cap = cv2.VideoCapture(source)  # 0 is the default ID for the primary webcam

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    _, frame = cap.read()

    height, width, _= frame.shape
    # print(frame.shape)

    maxlebar, maxtinggi = width, height
    
    # print("maxlebar, maxtinggi:",maxlebar, maxtinggi)
    maxlebara, maxtinggia = int(maxlebar/2), int(maxtinggi)
    # print("maxlebara, maxtinggia:", maxlebara, maxtinggia)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec
    out = None  # Initialize VideoWriter object
    recording = False  # Flag to indicate whether recording is active

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # print("Repeat the Video")
            # continue
            print("Video is done playing or failed to open camera")
            break

        # Check for keypress
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit the loop if 'q' is pressed
            print("Q pressed")
            cap.release()
            break
        
        elif key == ord('d'): # Quit the loop if 'd' is pressed
            print("D pressed")
            specialname = FName()
            DirName = "./data-extraction/recorded-video/"+ name + '-hand-' + specialname + '/'
            os.mkdir(DirName)  
            print("Directory:", DirName)

        elif key == ord('r'):  # Start/stop recording if 'r' is pressed
            if not recording:
                print("R pressed-start")
                # out = cv2.VideoWriter(DirName+'output.avi', fourcc, 20.0, (640, 360))  # Output filename, codec, FPS, frame size
                filename = DirName + name + '-vid-' + str(i) + '-' + specialname +'.avi'
                print("Creating file:", filename)
                out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
                recording = True
                i = i + 1
                
            else:
                print("R pressed-stop")
                out.release()
                recording = False

        # If recording, write the frame into the video file
        if recording:
            out.write(image)

        height, width, _= image.shape
        kiri = image[:, :width // 2]
        kanan = image[:, width //2:]
        # print(height, width)
        # draw rectangle
        # Window name in which image is displayed 
        window_name = 'Image'
        
        # Start coordinate, here (5, 5) 
        # represents the top left corner of rectangle 
        ukuran = int(maxlebara/2)
        ax, ay = int(maxlebara/2), int(maxtinggia/2)
        axx = int(ukuran/2)
        bx, by = int(maxlebara/2), int(maxtinggia/2)

        start_point = (ax-axx, ay-axx) 
        end_point = (bx+axx, by+axx)
        # print("start", start_point)
        # print("end", end_point)

        # Blue color in BGR 
        color = (0, 0, 255) 
        
        # Line thickness of 2 px 
        thickness = 2
        
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        kiri = cv2.rectangle(kiri, start_point, end_point, color, thickness) 

        # Frame by frame visualization
        cv2.imshow("Sighting Frame", kiri)
        # cv2.imshow("Frame full", image)
        

    # Release the VideoCapture and VideoWriter objects, and close the OpenCV windows
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    source = 0
    subject_name = "ema"
    main(0, subject_name)
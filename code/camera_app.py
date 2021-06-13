"""Demo for use yolo v3
"""
import os
import sys
import time
import cv2
import argparse
import numpy as np
import pandas as pd
from yolo_model import YOLO
import pantilthat as pth


# Input Args/Switches
ap = argparse.ArgumentParser()
ap.add_argument("-bbb", "--show_blue_border_box", action="store_true", default=0, help="Draws a (blue) border boundary box which can dictate camera movement. Default: Show box")
ap.add_argument("-p", "--show_person_box", action="store_true", default=0, help="Draws a (green) box around the person. Default: Show box")
ap.add_argument("-m", "--still_camera", action="store_true", default=0, help="Keep camera Still when enabled (1); otherwise the camera moves (0). Default: Camera moves")
ap.add_argument("-c", "--is_cascade", action="store_true", default=0, help="Whether to use Haar Cascade (1) or YOLO (0). Default: Yolo (0)")
ap.add_argument("-o", "--output_data", action="store_true", default=0, help="Whether to output FPS data to a .CSV file")
ap.add_argument("-f", "--is_wear_fun_hat", action="store_true", default=0, help="Puts a fun hat on you. Works only with Haar Cascade ATM. Default: No fun hat")
ap.add_argument("-w", "--is_windows", action="store_true", default=0, help="Runs on Windows (for testing, camera movement is disabled).")
args = ap.parse_args()


# GLOBALS
key_letter = 27
show_text = False
is_yolo_face = False
show_boundary_box = args.show_blue_border_box
show_person_box = args.show_person_box
is_camera_still = args.still_camera
is_cascade = args.is_cascade
output_data = args.output_data
is_windows = args.is_windows
is_wear_fun_hat = args.is_wear_fun_hat
shrink_person_box = False
show_onscreen_help = False
hat_path = '../assets/img/Propeller_hat.svg.med.png'
hat_img = cv2.imread(hat_path, -1)
# create Data Frame
df = pd.DataFrame({'Model':['Test'], 'FPS': np.nan, 'time': [0]})
df.dropna(inplace=True) # To retain column order
# Frame dimensions vars
FRAME_W = FRAME_H = 0
x = y = w = h = 0
# Boundary Box vars
w_min = w_max = h_min = h_max = 0


def man_move_camera(key_press):
    """Take keystrokes to dictate camera movemement.

    # Argument:
        key_press: takes in one letter for movement
                    (same as gaming controls, no inversion):
                        w: Up
                        a: Left
                        s: Down
                        d: Right

    # Returns
        None
    """
    cam_pan = pth.get_pan()
    cam_tilt = pth.get_tilt()
    move_x = 0
    move_y = 0

    if(key_press.lower() == 'a'):
        move_x = -2
    elif(key_press.lower() == 'd'):
        move_x = 2
    elif(key_press.lower() == 's'):
        move_y = -1
    elif(key_press.lower() == 'w'):
        move_y = 1

    if((cam_pan + move_x < 90) & (cam_pan - move_x > -90)):
        cam_pan += move_x
        pth.pan(int(cam_pan))
        time.sleep(0.005)
    else:
        print(f'MAX PAN - cannot move:  {cam_pan + move_x}')

    if((cam_tilt + move_y < 90) & (cam_tilt - move_y > -90)):
        cam_tilt -= move_y
        pth.tilt(int(cam_tilt))
        time.sleep(0.005)
    else:
        print(f'MAX TILT - cannot move:  {cam_tilt + move_y}')
    return


def move_camera(x, y, w, h):
    """Takes in object tracking coordinates and
        moves camera to try to "center" the subject.

    # Argument:
        x: coordinate on the x axis where subject is detected
        y: coordinate on the y axis where subject is detected
        w: width of object detected on screen
        h: height of object detected on screen

    # Returns
        None
    """
    if(is_camera_still):
        return
    if(shrink_person_box):
        #for camera tracking only
        # shrink height and width by half
        w = w//2
        h = h//2
        # center box by adding a quarter of w/h to x/y
        x += w//2
        y += h//2

    cam_pan = pth.get_pan()
    cam_tilt = pth.get_tilt()
    move_x = 2
    move_y = 1
    yolo_offset = 0 if is_cascade else (h_min * -0.75)

    if(((x + w)*0.95 > w_max) & (x*0.95 < w_min)):
        # If both subject borders take up 95% or
        # more of the boundary box, do nothing
        pass
    elif(w > (w_max - w_min)*0.95):
        # If subject border-length take up 95% (not centered)
        # or more of the boundary box, correct movement by aligning centers
        if(x + w/2 > (FRAME_W + w_min)/2):
            cam_pan += move_x
            pth.pan(int(cam_pan))
        elif(x - w/2 < (FRAME_W - w_min)/2):
            cam_pan -= move_x
            pth.pan(int(cam_pan))
    elif((cam_pan + move_x < 90) & (cam_pan - move_x > -90)):
        if(x + w > w_max):
            cam_pan += move_x
            pth.pan(int(cam_pan))
        elif(x < w_min):
            cam_pan -= move_x
            pth.pan(int(cam_pan))
    else:
        print(f'MAX PAN - cannot move:  {cam_pan + move_x}')

    if(((y + h)*0.95 > h_max) & (y*0.95 < h_min)):
        # If both subject borders take up 95% or
        # more of the boundary box, do nothing
        pass
    elif(h > (h_max - h_min)*0.95):
        # If subject border-length take up 95% (not centered)
        # or more of the boundary box, correct movement by aligning centers
        if(y + h/2 > (FRAME_H + h_min)/2):
            cam_tilt += move_y
            pth.tilt(int(cam_tilt))
        elif(y - h/2 < (FRAME_H - h_min)/2):
            cam_tilt -= move_y
            pth.tilt(int(cam_tilt))
    elif((cam_tilt + move_y < 90) & (cam_tilt - move_y > -90)):
        if(y + h > h_max):
            cam_tilt += move_y
            pth.tilt(int(cam_tilt))
        elif(y < h_min + yolo_offset):
            cam_tilt -= move_y
            pth.tilt(int(cam_tilt))
    else:
        print(f'MAX TILT - cannot move:  {cam_tilt + move_y}')
    return


def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.
    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    # Returns
        None
    """
    a = b = c = d = 0
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        if(not is_camera_still):
            move_camera(x, y, w, h)

        if(show_person_box):
            cv2.rectangle(image, (top, left), (right, bottom), (0, 0, 255), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 1,
                        cv2.LINE_AA)

        a, b, c, d = left, top, int(right - left), int(bottom - top)
    if(show_text):
        show_stats(a, b, c, d)
    return


def detect_image(image, yolo, all_classes, w_img=0, h_img=0):
    """Use yolo v4 tiny to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.
        w_img: width of subject detected on screen
        h_img: height of subject detected on screen

    # Returns:
        image: processed image.
    """
    start = time.time()
    if((w_img == 0) or (h_img == 0)):
        w_img = image.shape[0]
        h_img = image.shape[1]
    boxes, classes, scores = yolo.predict(image, (w_img, h_img))
    end = time.time()
    t = end - start
    global df
    df = df.append({'Model': yolo.name, 'FPS': (1/t), 'time': t}, ignore_index=True)


    if(show_text):
        show_time_and_fps(t)
    print('time: {0:.2f}s   - FPS: {1:.2f}'.format((end - start), (1/(end - start))), end='\r')
    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image


def show_help():
    """Draw help commands at the bottom of the image.

    # Argument:
    None

    # Returns
        None
    """
    help_text1 = f'q: Quit; SHOW - h: help; b: bound box; p: person box; f: fun;'
    cv2.putText(frame,
                help_text1,
                (10, FRAME_H - 80),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(frame,
                help_text1,
                (17, FRAME_H - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (240, 240, 240), 1,
                cv2.LINE_AA)
    help_text2 = f'k: shrink bound box; m: man/auto cam (wasd); i: MS Win;'
    cv2.putText(frame,
                help_text2,
                (10, FRAME_H - 60),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(frame,
                help_text2,
                (17, FRAME_H - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (240, 240, 240), 1,
                cv2.LINE_AA)
    help_text3 = f'c: Harr/YOLO; y: YOLO body/face; t: stats; r: reset cam pos;'
    cv2.putText(frame,
                help_text3,
                (10, FRAME_H - 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(frame,
                help_text3,
                (17, FRAME_H - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (240, 240, 240), 1,
                cv2.LINE_AA)
    return


def show_stats(x, y, w, h):
    """Draw interesting information at the top of the image.

    # Argument:
        x: original image.
        y: ndarray, boxes of objects.
        w: ndarray, classes of objects.
        h: ndarray, scores of objects.

    # Returns
        None
    """
    nerd_text1 = f'Nerd Stats: x: {x}, y: {y}, w: {w}, h: {h}'
    cv2.putText(frame,
                nerd_text1,
                (10, 20),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(frame,
                nerd_text1,
                (15, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (240, 240, 240), 1,
                cv2.LINE_AA)
    casc_txt = 'Method: Haar Cascade, ' if is_cascade else ('Method: YOLO Face, ' if is_yolo_face else 'Method: YOLO, ')
    cam_mov_txt = 'Camera Move: Manual Only' if is_camera_still else 'Camera Move: Auto+'
    nerd_text2 = casc_txt + cam_mov_txt
    cv2.putText(frame,
                nerd_text2,
                (10, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(frame,
                nerd_text2,
                (17, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (240, 240, 240), 1,
                cv2.LINE_AA)
    return


def show_time_and_fps(t):
    """Draw interesting time information at the top of the image.

    # Argument:
        t: time it takes to draw frame.

    # Returns
        None
    """
    f = 1/t
    od = 'Y' if(output_data) else 'N'
    fps_text1 = f'FPS: {f:.2f},    render time: {t:.2f},  Output: {od}'
    cv2.putText(frame,
                fps_text1,
                (10, 60),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(frame,
                fps_text1,
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (240, 240, 240), 1,
                cv2.LINE_AA)
    return


def draw_fun_hat():
    """Draw a fun hat on the subject in frame.
        Note: this is currently only implemented
                for the Haar Cascade model

    # Argument:
        None

    # Returns
        None
    """
    # Transparancy and image overlay
    # https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
    try:
        resize_x = int(w*1.1)
        resize_y = int(w*2/3)
        overlay = cv2.resize(hat_img, (resize_x, resize_y), interpolation = cv2.INTER_AREA)
        # overlay = cv2.resize(overlay, (170, 100),interpolation = cv2.INTER_AREA) # Fixed-size hat
        x_offset = x - 10
        y_offset = y - (h//2)
        y1, y2 = y_offset, y_offset + overlay.shape[0]
        x1, x2 = x_offset, x_offset + overlay.shape[1]
        alpha_s = overlay[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * overlay[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
        # Draw hat without transparency
        # x_offset = y_offset = 50
        # frame[y_offset:y_offset+overlay.shape[0], x_offset:x_offset+overlay.shape[1]] = overlay
    except:
        print('Cannot draw hat; face moved out of canvas-drawing area.')
    return


def reset_camera_position():
    """Resets Camera position.

    # Argument:
        None

    # Returns:
        None
    """
    pth.pan(0)
    pth.tilt(-20)
    time.sleep(2)


def get_date_time(suffix=''):
    year = time.localtime().tm_year
    month = time.localtime().tm_mon
    mday = time.localtime().tm_mday
    hour = time.localtime().tm_hour
    min = time.localtime().tm_min
    sec = time.localtime().tm_sec
    fill_mo = '0' if(month < 10) else ''
    fill_day = '0' if(mday < 10) else ''
    fill_hr = '0' if(hour < 10) else ''
    fill_min = '0' if(min < 10) else ''
    fill_sec = '0' if(sec < 10) else ''
    output_str = str(year) + fill_mo + str(month) + fill_day + str(mday)
    output_str += '_' + fill_hr + str(hour) + fill_min + str(min) + fill_sec + str(sec)
    return output_str + suffix


if __name__ == '__main__':
    """Main Function.

    # Argument:
        None

    # Returns:
        None
    """
    # YOLOv4tiny defined with full body
    cfg_file = '../cfg/custom-yolov4-tiny-detector.cfg'
    weights_file = '../models/custom-yolov4-tiny-detector_best.weights'
    file = '../names/custom-yolov4-tiny-detector.names'
    all_classes = get_classes(file)
    yolo = YOLO(cfg_file, weights_file, 'YOLO-Body')

    # YOLOv4tiny focused on faces
    cfg_face_file = '../cfg/custom-yolov4-tiny-detector_face.cfg'
    weights_face_file = '../models/custom-yolov4-tiny-detector_face_best.weights'
    face_file = '../names/custom-yolov4-tiny-detector_face.names'
    all_classes_face = get_classes(face_file)
    yolo_face = YOLO(cfg_face_file, weights_face_file, 'YOLO-Face')

    # Turn the camera to the default position
    if(not is_camera_still):
        reset_camera_position()

    if(is_windows):
        # Secondary, Monitor Camera (for my windows computer)
        cap = cv2.VideoCapture(1)
    else:
        # Primary, Laptop Camera or rpi Camera
        cap = cv2.VideoCapture(0)

    # Set placement vars
    FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # width
    FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # height
    # For BOUNDARY BOX
    w_min = int((FRAME_W)/6)
    w_max = int((FRAME_W) - w_min)
    h_min = int((FRAME_H)/5)
    h_max = int((FRAME_H) - h_min)

    cascPath = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
    if(is_windows):
        cascPath = 'C:\ProgramData\Anaconda3\envs\pantilthat\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if(not ret):
            print("Error getting image")
            continue

        # Vertical flip for camera orientation (Ribbon on top of camera)
        if(is_windows):
            # Mirror flip (since not rotated 180 deg)
            frame = cv2.flip(frame, 1)
        else:
            # Vertical flip (camera is rotated 180 deg because of cabling)
            frame = cv2.flip(frame, 0)

        if(is_cascade):  # | args.is_cascade):
            start = time.time()
            faces = faceCascade.detectMultiScale(frame, 1.1, 3)
            for (x, y, w, h) in faces:
                if(is_wear_fun_hat):
                    draw_fun_hat()
                elif(show_person_box):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame,
                                'Haar Cascade',
                                (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 1,
                                cv2.LINE_AA)

                if(not is_camera_still):
                    move_camera(x, y, w, h)
                break
            end = time.time()
            print('time: {0:.2f}s   - FPS: {1:.2f}'.format((end - start), (1/(end - start))), end='\r')
            # Show stats, e.g. camera tracking on, positions, model, etc.
            t = end - start
            if(show_text):
                show_stats(x, y, w, h)
                show_time_and_fps(t)
            df = df.append({'Model': 'Haar Cascade', 'FPS': (1/t), 'time': t}, ignore_index=True)
        else:
            # IS YOLO
            if(is_yolo_face):
                frame = detect_image(frame, yolo_face, all_classes_face, FRAME_W, FRAME_H)
            else:
                frame = detect_image(frame, yolo, all_classes, FRAME_W, FRAME_H)

        if(show_boundary_box):
            cv2.rectangle(frame, (w_min, h_min), (w_max, h_max), (255, 0, 0), 2)
        if(show_onscreen_help):
            show_help()

        ################ SHOW FRAME ################
        cv2.imshow('Video', frame)
        ################ SHOW FRAME ################

        # Take input for features
        key_stroke = cv2.waitKey(1)
        key_letter = key_stroke
        if key_stroke & 0xFF == ord('q'):
            break
        elif key_stroke & 0xFF == 27:
            break
        elif key_stroke & 0xFF == ord('b'):
            show_boundary_box = not show_boundary_box
        elif key_stroke & 0xFF == ord('p'):
            show_person_box = not show_person_box
        elif key_stroke & 0xFF == ord('m'):
            is_camera_still = (not is_camera_still)
        elif key_stroke & 0xFF == ord('c'):
            is_cascade = not is_cascade
        elif key_stroke & 0xFF == ord('y'):
            is_yolo_face = not is_yolo_face
        elif key_stroke & 0xFF == ord('h'):
            show_onscreen_help = not show_onscreen_help
        elif key_stroke & 0xFF == ord('f'):
            is_wear_fun_hat = not is_wear_fun_hat
        elif key_stroke & 0xFF == ord('t'):
            show_text = not show_text
        elif key_stroke & 0xFF == ord('i'):
            is_windows = not is_windows
        elif key_stroke & 0xFF == ord('k'):
            shrink_person_box = not shrink_person_box
        elif key_stroke & 0xFF == ord('o'):
            output_data = not output_data
        
        if(not is_windows):
            if key_stroke & 0xFF == ord('r'):
                reset_camera_position()
            elif key_stroke & 0xFF == ord('w'):
                man_move_camera('w')
            elif key_stroke & 0xFF == ord('a'):
                man_move_camera('a')
            elif key_stroke & 0xFF == ord('s'):
                man_move_camera('s')
            elif key_stroke & 0xFF == ord('d'):
                man_move_camera('d')


        prev_time = time.time()
    print('\n')
    # output dated file
    if(output_data):
        file_out = get_date_time(suffix='_stats.csv')
        path_out = '../output/'
        if(not os.path.exists(path_out)):
            os.mkdir(path_out)
        path_and_file_out = path_out + file_out
        df.to_csv(path_and_file_out)
        print(f'Output file created: {path_and_file_out}')
    # Release/Destroy resources when finished
    cap.release()
    cv2.destroyAllWindows()

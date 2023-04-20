import time
from typing import Any

from flask import Flask, render_template, request, redirect, url_for, Response

import cv2
import mediapipe as mp
import requests
import numpy as np
import json

SERVER_HOST = 'localhost'
SERVER_PORT = 9000
API_PATH = '/prediction'

app = Flask(__name__)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

cam = Any


def start_video_capture():
    cap = cv2.VideoCapture(0)
    cv2.waitKey(2)
    return cap


image_roi = np.zeros((375, 280, 3), np.uint8)

response_sign = str()


def frames_generator(capture):
    if not capture.isOpened():
        print('Error opening video')
        exit(0)
    else:
        while True:
            success, frame = capture.read()
            if success:

                color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # find hands on the frame
                results = hands.process(color_image)
                h, w, c = frame.shape

                global image_roi

                x_max = 0
                y_max = 0
                x_min = w
                y_min = h

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            if x > x_max:
                                x_max = x
                            if x < x_min:
                                x_min = x
                            if y > y_max:
                                y_max = y
                            if y < y_min:
                                y_min = y

                        cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 30, y_max + 30), (0, 255, 0), 2)
                        image_roi = frame[y_min - 20: y_max + 30, x_min - 20:x_max + 30]
                        cv2.waitKey(10)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            else:
                break


def snap_generator():
    global response_sign
    _, buffer = cv2.imencode('.jpg', image_roi)
    frame_t = buffer.tobytes()
    url = "http://" + SERVER_HOST + ":" + str(SERVER_PORT) + API_PATH
    if not np.all(image_roi == 0):  # check if it not is 0 or None
        response_sign = requests.post(url, data=frame_t, headers={'Content-Type': 'application/octet-stream'})
    yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_t + b'\r\n'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/init', methods=['POST', 'GET'])
def init():
    global cam, image_roi, response_sign
    cam = start_video_capture()
    if request.method == 'POST':
        if request.form.get('stop') == 'Stop':
            cam.release()
            cv2.destroyAllWindows()
            image_roi = np.zeros((375, 280, 3), np.uint8)  # init image_roi
            response_sign = str()  # init response
        return redirect(url_for('index'))
    else:
        return redirect(url_for('index'))


@app.route('/video')
def video():
    global cam
    cam = start_video_capture()
    return Response(frames_generator(cam), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snap')
def snap():
    return Response(snap_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result')
def result():
    global response_sign

    frame = np.ones((340, 450, 3), np.uint8)

    if response_sign:
        image_height, image_width = frame.shape[:2]
        prediction_value = json.loads(response_sign.text)

        text = prediction_value['response']
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = cv2.getFontScaleFromHeight(font, 120, thickness=2)
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness=2)
        text_width, text_height = text_size

        x_coordinate = (image_width - text_width) // 2
        y_coordinate = image_height - baseline - 70

        # add text centered on image
        cv2.putText(frame, text, (x_coordinate, y_coordinate), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        response_sign = str()  # init response

    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    image = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
    return Response(image, mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

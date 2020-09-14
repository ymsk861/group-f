import cv2
import numpy as np
import boto3
import time
from flask import Flask, render_template, request
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import matplotlib.pyplot as plt
import joblib


app = Flask(__name__)


def detect_face():

    # setup 
    scale_factor = .15
    green = (0,255,0)
    red = (0,0,255)
    frame_thickness = 2
    cap = cv2.VideoCapture(0)
    rekognition = boto3.client('rekognition', region_name='us-west-2')
    fontscale = 1.0
    color = (0, 120, 238)
    fontface = cv2.FONT_HERSHEY_DUPLEX

    while True:
        # 正常に動作した場合は1回のみ認識されます

        # Capture frame-by-frame
        ret, frame = cap.read()
        height, width, channels = frame.shape

        # Convert frame to jpg
        small = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
        ret, buf = cv2.imencode('.jpg', small)

        # Detect faces in jpg
        faces = rekognition.detect_faces(Image={'Bytes':buf.tobytes()}, Attributes=['ALL'])

        # Recognition
        for face in faces['FaceDetails']:
            smile = face['Smile']['Value']
            cv2.rectangle(frame,
                        (int(face['BoundingBox']['Left']*width),
                        int(face['BoundingBox']['Top']*height)),
                        (int((face['BoundingBox']['Left']+face['BoundingBox']['Width'])*width),
                        int((face['BoundingBox']['Top']+face['BoundingBox']['Height'])*height)),
                        green if smile else red, frame_thickness)
            emothions = face['Emotions']
            i = 0
            for emothion in emothions:
                cv2.putText(frame,
                            str(emothion['Type']) + ": " + str(emothion['Confidence']),
                            (25, 40 + (i * 25)),
                            fontface,
                            fontscale,
                            color)
                i += 1

            dom_em_conf=emothions[0]['Confidence']
            dom_em_type=emothions[0]['Type']
            sec_em_conf=emothions[1]['Confidence']
            sec_em_type=emothions[1]['Type']
            print('\n')
            print(emothions, '\n')

            if dom_em_conf-sec_em_conf>30:
                print(dom_em_type,dom_em_conf,'\n')
            else:
                print(dom_em_type,dom_em_conf)
                print(sec_em_type,sec_em_conf,'\n')
            
            cap.release()
            cv2.destroyAllWindows()
            
            return [dom_em_type, dom_em_conf]

        print("顔が認識できません．カメラを確認して下さい")


#スタートボタンを押す（＝http://127.0.0.1:5000/が再読み込まれる）と93行目から再読み込みされる（この時request.method='POST'）
#自動リロードされる場合も同じ（この時request.method='GET'）
@app.route('/', methods = ['GET', 'POST'])
def aaa():
    print(request.method)
    if request.method == 'GET':
        return render_template('first.html')
    
    # firstのスタートボタンを押すと実行
    elif request.method == 'POST':
        dom_em_type = detect_face() # スタートボタン押してからカメラ起動
        print(dom_em_type, '\n')
        return render_template('second.html', dom_em_type=dom_em_type) # dom_em_type ['HAPPY', 99.76985168457031]

if __name__ == "__main__":
    app.run()
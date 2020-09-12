import cv2
import numpy as np
import boto3
import time
from flask import Flask, render_template, request
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import matplotlib.pyplot as plt
 

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


# flask
app = Flask(__name__)

# dom_em_type = 'ワハハハ'

@app.route('/', methods = ['GET', 'POST'])
def aaa():
    print(request.method)
    if request.method == 'GET':
        return render_template('first.html')
    
    # firstのスタートボタンを押すと実行
    elif request.method == 'POST':
        
        # カメラ起動 + 認識
        # emothions = dom_em_type
        # print(emothions)

        # 顔データ送信 emotions : [{'Type': 'CALM', 'Confidence': 92.8204574584961}, {'Type': 'SURPRISED', 'Confidence': 3.136558771133423}, ...]
        return render_template('second.html', dom_em_type=dom_em_type)

"""
@app.route('/camera')
    def bbb():
        return render_template('')
"""

def detect_face():
    while(True): # 下でreturnしているので表情を読み取るのは1回です

        # Capture frame-by-frame
        ret, frame = cap.read()
        height, width, channels = frame.shape

        # Convert frame to jpg
        small = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
        ret, buf = cv2.imencode('.jpg', small)

        # Detect faces in jpg
        faces = rekognition.detect_faces(Image={'Bytes':buf.tobytes()}, Attributes=['ALL'])

        

        # Draw rectangle around faces
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
            time.sleep(2)
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
            

            if dom_em_conf-sec_em_conf>30:
                print(dom_em_type,dom_em_conf,'\n')
            else:
                print(dom_em_type,dom_em_conf)
                print(sec_em_type,sec_em_conf,'\n')
        return emothions

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return 




if __name__ == "__main__":
    app.run()     

# カメラ画像を画面に表示したい場合は96行目に以下のコードを書く
"""
# Display the resulting frame
cv2.imshow('frame', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
"""
# coding: utf-8
from time import sleep

import boto3
import cv2

client = boto3.client('rekognition',region_name='us-west-2')
scale_factor = .15
fontscale = 1.0
color = (0, 120, 238) # フォントカラー(B, G, R)
fontface = cv2.FONT_HERSHEY_DUPLEX
green = (0, 255, 0)
red = (0, 0, 255)


def detect_label(cv2, buf):
    response = client.detect_labels(Image={'Bytes': buf.tobytes()})
    i = 0
    for label in response['Labels']:
        print(label['Name'] + ' : ' + str(label['Confidence']))
        # cv2.putText(描画先, 描画文字列, 描画座標[左下が基準], フォント, フォントカラー)
        cv2.putText(frame, label['Name'] + ' : ' + str(label['Confidence']), (25, 40 + (i * 25)), fontface,
                    fontscale, color)
        i += 1

def detect_faces(cv2, buf):
    frame_thickness = 2
    faces = client.detect_faces(Image={'Bytes': buf.tobytes()}, Attributes=['ALL'])

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
        for emothion in emothions:
            cv2.putText(frame,
                        str(emothion['Type']) + ": " + str(emothion['Confidence']),
                        (25, 40 + (i * 25)),
                        fontface,
                        fontscale,
                        color)
            i += 1

def detect_text(cv2, buf):
    response = client.detect_text(Image={'Bytes':buf.tobytes()})
    textDetections = response['TextDetections']
    for text in textDetections:
        print('Detected text:' + text['DetectedText'])
        print('Confidence: ' + "{:.2f}".format(text['Confidence']) + "%")
        print('Id: {}'.format(text['Id']))
        cv2.rectangle(frame,
                      (int(text['Geometry']['BoundingBox']['Left'] * width),
                       int(text['Geometry']['BoundingBox']['Top'] * height)),
                      (int((text['Geometry']['BoundingBox']['Left'] + text['Geometry']['BoundingBox']['Width']) * width),
                       int((text['Geometry']['BoundingBox']['Top'] + text['Geometry']['BoundingBox']['Height']) * height)),
                      green)
        if 'ParentId' in text:
            print('Parent Id: {}'.format(text['ParentId']))
            print('Type:' + text['Type'])


def detect_moderation_labels(cv2, buf):
    response = client.detect_moderation_labels(Image={'Bytes':buf.tobytes()})
    for label in response['ModerationLabels']:
        print (label['Name'] + ' : ' + str(label['Confidence']))
        print (label['ParentName'])
        cv2.putText(frame, label['Name'] + ' : ' + str(label['Confidence']), (25, 40), fontface, fontscale, color)


if __name__ == "__main__":

    # 内蔵カメラを起動
    cap = cv2.VideoCapture(0)

    while True:

        # 内蔵カメラから読み込んだキャプチャデータを取得
        ret, frame = cap.read()
        height, width, channels = frame.shape
        small = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

        ret, buf = cv2.imencode('.png', small)

        # detect_label(cv2, buf)
        # detect_faces(cv2, buf)
        # detect_text(cv2, buf)
        # detect_moderation_labels(cv2, buf)

        # 表示
        cv2.imshow("frame", frame)

        # qキーを押すとループ終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 内蔵カメラを終了
    cap.release()
    cv2.destroyAllWindows()

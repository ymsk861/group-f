# coding: utf-8
from time import sleep

import boto3
import cv2

client = boto3.client('rekognition',region_name='us-west-2')

if __name__ == "__main__":

    # 内蔵カメラを起動
    cap = cv2.VideoCapture(0)

    cascade_path = "./model/haarcascade_frontalface_alt.xml"
    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)

    # 顔に表示される枠の色を指定（白色）
    color = (255, 255, 255)

    while True:

        # 内蔵カメラから読み込んだキャプチャデータを取得
        ret, frame = cap.read()

        # モノクロで表示する
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔認識の実行
        facerect = cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

        # 顔が見つかったらcv2.rectangleで顔に白枠を表示する
        if len(facerect) > 0:
            for rect in facerect:
                cv2.rectangle(frame, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=2)

        # 表示
        cv2.imshow("frame", frame)

        # qキーを押すとループ終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 内蔵カメラを終了
    cap.release()
    cv2.destroyAllWindows()

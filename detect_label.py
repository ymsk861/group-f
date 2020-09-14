# coding: utf-8
from time import sleep

import boto3
import cv2

client = boto3.client('rekognition',region_name='us-west-2')
scale_factor = .15

if __name__ == "__main__":

    # 内蔵カメラを起動
    cap = cv2.VideoCapture(0)

    # フォントの大きさ
    fontscale = 1.0
    # フォントカラー(B, G, R)
    color = (0, 120, 238)
    # フォント
    """
    使用可能なフォント一覧
    FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_COMPLEX_SMALL
    FONT_HERSHEY_DUPLEX
    FONT_HERSHEY_PLAIN
    FONT_HERSHEY_SCRIPT_COMPLEX
    FONT_HERSHEY_SCRIPT_SIMPLEX
    FONT_HERSHEY_SIMPLEX
    FONT_HERSHEY_TRIPLEX
    FONT_ITALIC
    """
    fontface = cv2.FONT_HERSHEY_DUPLEX

    while True:

        # 内蔵カメラから読み込んだキャプチャデータを取得
        ret, frame = cap.read()
        height, width, channels = frame.shape
        small = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

        ret, fileImg = cv2.imencode('.png', small)
        response = client.detect_labels(Image={'Bytes': fileImg.tobytes()})

        # print('Detected labels for Camera Capture.\n')
        i = 0
        for label in response['Labels']:
            print(label['Name'] + ' : ' + str(label['Confidence']))
            # cv2.putText(描画先, 描画文字列, 描画座標[左下が基準], フォント, フォントカラー)
            cv2.putText(frame, label['Name'] + ' : ' + str(label['Confidence']), (25, 40 + (i * 25)), fontface,
                        fontscale, color)
            i += 1

        # print(response)
        print('\n')

        # 表示
        cv2.imshow("frame", frame)

        # qキーを押すとループ終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 内蔵カメラを終了
    cap.release()
    cv2.destroyAllWindows()

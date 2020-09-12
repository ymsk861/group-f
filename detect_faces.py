import cv2
import numpy as np
import boto3
import time
import matplotlib.pyplot as plt
 
# Setup
scale_factor = .15
green = (0,255,0)
red = (0,0,255)
frame_thickness = 2
cap = cv2.VideoCapture(0)
rekognition = boto3.client('rekognition',region_name='us-west-2')

# font-size
fontscale = 1.0
# font-color (B, G, R)
color = (0, 120, 238)
# font
fontface = cv2.FONT_HERSHEY_DUPLEX

@app.route('/')
def aaa():
    return render_template('index.html', form=form)
"""
    while(True):

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

            
    # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
"""
        
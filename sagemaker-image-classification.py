import json
import numpy as np
import cv2

import boto3
runtime = boto3.Session().client(service_name='runtime.sagemaker')
scale_factor = .15

if __name__ == "__main__":

    # 内蔵カメラを起動
    cap = cv2.VideoCapture(0)

    # フォントの大きさ
    fontscale = 1.0
    # フォントカラー(B, G, R)
    color = (0, 120, 238)
    # フォント
    fontface = cv2.FONT_HERSHEY_DUPLEX

    while True:

        # 内蔵カメラから読み込んだキャプチャデータを取得
        ret, frame = cap.read()
        height, width, channels = frame.shape

        small = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
        ret, fileImg = cv2.imencode('.png', small)

        endpoint_name = 'DEMO-imageclassification-ep--2018-09-16-07-40-25'
        response = runtime.invoke_endpoint(EndpointName=endpoint_name,
                                           ContentType='application/x-image',
                                           Body=fileImg.tobytes())
        result = response['Body'].read()
        # result will be in json format and convert it to ndarray
        result = json.loads(result)
        # the result will output the probabilities for all classes
        # find the class with maximum probability and print the class index
        # index = np.argmax(result)
        indexs = np.argsort(result)[::-1][:5]  # ここを修正
        object_categories = ['ak47', 'american-flag', 'backpack', 'baseball-bat', 'baseball-glove', 'basketball-hoop',
                             'bat', 'bathtub', 'bear', 'beer-mug', 'billiards', 'binoculars', 'birdbath', 'blimp',
                             'bonsai-101', 'boom-box', 'bowling-ball', 'bowling-pin', 'boxing-glove', 'brain-101',
                             'breadmaker', 'buddha-101', 'bulldozer', 'butterfly', 'cactus', 'cake', 'calculator',
                             'camel', 'cannon', 'canoe', 'car-tire', 'cartman', 'cd', 'centipede', 'cereal-box',
                             'chandelier-101', 'chess-board', 'chimp', 'chopsticks', 'cockroach', 'coffee-mug',
                             'coffin', 'coin', 'comet', 'computer-keyboard', 'computer-monitor', 'computer-mouse',
                             'conch', 'cormorant', 'covered-wagon', 'cowboy-hat', 'crab-101', 'desk-globe',
                             'diamond-ring', 'dice', 'dog', 'dolphin-101', 'doorknob', 'drinking-straw', 'duck',
                             'dumb-bell', 'eiffel-tower', 'electric-guitar-101', 'elephant-101', 'elk', 'ewer-101',
                             'eyeglasses', 'fern', 'fighter-jet', 'fire-extinguisher', 'fire-hydrant', 'fire-truck',
                             'fireworks', 'flashlight', 'floppy-disk', 'football-helmet', 'french-horn', 'fried-egg',
                             'frisbee', 'frog', 'frying-pan', 'galaxy', 'gas-pump', 'giraffe', 'goat',
                             'golden-gate-bridge', 'goldfish', 'golf-ball', 'goose', 'gorilla', 'grand-piano-101',
                             'grapes', 'grasshopper', 'guitar-pick', 'hamburger', 'hammock', 'harmonica', 'harp',
                             'harpsichord', 'hawksbill-101', 'head-phones', 'helicopter-101', 'hibiscus',
                             'homer-simpson', 'horse', 'horseshoe-crab', 'hot-air-balloon', 'hot-dog', 'hot-tub',
                             'hourglass', 'house-fly', 'human-skeleton', 'hummingbird', 'ibis-101', 'ice-cream-cone',
                             'iguana', 'ipod', 'iris', 'jesus-christ', 'joy-stick', 'kangaroo-101', 'kayak',
                             'ketch-101', 'killer-whale', 'knife', 'ladder', 'laptop-101', 'lathe', 'leopards-101',
                             'license-plate', 'lightbulb', 'light-house', 'lightning', 'llama-101', 'mailbox',
                             'mandolin', 'mars', 'mattress', 'megaphone', 'menorah-101', 'microscope', 'microwave',
                             'minaret', 'minotaur', 'motorbikes-101', 'mountain-bike', 'mushroom', 'mussels', 'necktie',
                             'octopus', 'ostrich', 'owl', 'palm-pilot', 'palm-tree', 'paperclip', 'paper-shredder',
                             'pci-card', 'penguin', 'people', 'pez-dispenser', 'photocopier', 'picnic-table',
                             'playing-card', 'porcupine', 'pram', 'praying-mantis', 'pyramid', 'raccoon',
                             'radio-telescope', 'rainbow', 'refrigerator', 'revolver-101', 'rifle', 'rotary-phone',
                             'roulette-wheel', 'saddle', 'saturn', 'school-bus', 'scorpion-101', 'screwdriver',
                             'segway', 'self-propelled-lawn-mower', 'sextant', 'sheet-music', 'skateboard', 'skunk',
                             'skyscraper', 'smokestack', 'snail', 'snake', 'sneaker', 'snowmobile', 'soccer-ball',
                             'socks', 'soda-can', 'spaghetti', 'speed-boat', 'spider', 'spoon', 'stained-glass',
                             'starfish-101', 'steering-wheel', 'stirrups', 'sunflower-101', 'superman', 'sushi', 'swan',
                             'swiss-army-knife', 'sword', 'syringe', 'tambourine', 'teapot', 'teddy-bear', 'teepee',
                             'telephone-box', 'tennis-ball', 'tennis-court', 'tennis-racket', 'theodolite', 'toaster',
                             'tomato', 'tombstone', 'top-hat', 'touring-bike', 'tower-pisa', 'traffic-light',
                             'treadmill', 'triceratops', 'tricycle', 'trilobite-101', 'tripod', 't-shirt',
                             'tuning-fork', 'tweezer', 'umbrella-101', 'unicorn', 'vcr', 'video-projector',
                             'washing-machine', 'watch-101', 'waterfall', 'watermelon', 'welding-mask', 'wheelbarrow',
                             'windmill', 'wine-bottle', 'xylophone', 'yarmulke', 'yo-yo', 'zebra', 'airplanes-101',
                             'car-side-101', 'faces-easy-101', 'greyhound', 'tennis-shoes', 'toad', 'clutter']
        i = 0
        for index in indexs:
            probability = result[index] * 100
            print("Result: label - " + object_categories[index] + ", probability - " + str(probability))
            cv2.putText(frame, object_categories[index] + ": " + str(probability), (25, 40 + (i*25)), fontface, fontscale,
                        color)
            i += 1

        # 表示
        cv2.imshow("frame", frame)

        # qキーを押すとループ終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 内蔵カメラを終了
    cap.release()
    cv2.destroyAllWindows()

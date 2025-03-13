import cv2
import torch
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class CentroidTracker:
    def __init__(self, maxDisappeared=20, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, position, bbox):
        self.objects[self.nextObjectID] = (position, bbox)
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputPositions = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int(y2)
            inputPositions[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputPositions)):
                self.register(inputPositions[i], rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectPositions = [self.objects[objectID][0] for objectID in objectIDs]

            D = dist.cdist(np.array(objectPositions), inputPositions)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = (inputPositions[col], rects[col])
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputPositions[col], rects[col])

        return self.objects


def load_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.to(device)


def detect_players_image(image_path, model_path):
    model = load_model(model_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Помилка: не вдалося завантажити {image_path}")
        return None

    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    detections = results.xyxy[0].cpu().numpy()

    boxes = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf < 0.5:
            continue
        if int(cls) != 0:
            continue
        boxes.append((int(x1), int(y1), int(x2), int(y2)))

    return boxes


def detect_players_video(video_path, model_path, output_path):
    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не вдалося відкрити відео {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    ct = CentroidTracker(maxDisappeared=20, maxDistance=50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        detections = results.xyxy[0].cpu().numpy()

        boxes = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf < 0.5:
                continue
            if int(cls) != 0:
                continue
            boxes.append((int(x1), int(y1), int(x2), int(y2)))

        objects = ct.update(boxes)

        for objectID, (position, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {objectID}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_video.write(frame)

    cap.release()
    output_video.release()
    print(f"Відео з результатами збережено в {output_path}")

import cv2
import torch
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import os


class CentroidTracker:
    def __init__(self, maxDisappeared=20, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.trajectories = OrderedDict()

    def register(self, position, bbox):
        self.objects[self.nextObjectID] = (position, bbox)
        self.disappeared[self.nextObjectID] = 0
        self.trajectories[self.nextObjectID] = [position]
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.trajectories[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, self.trajectories

        inputPositions = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int(y2)
            inputPositions[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputPositions)):
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
                self.trajectories[objectID].append(inputPositions[col])
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputPositions[col], rects[col])

        return self.objects, self.trajectories


def smooth_trajectory(pts, window=5):
    if len(pts) < window:
        return pts
    smoothed = []
    for i in range(len(pts)):
        start = max(0, i - window // 2)
        end = min(len(pts), i + window // 2 + 1)
        avg_x = int(np.mean([p[0] for p in pts[start:end]]))
        avg_y = int(np.mean([p[1] for p in pts[start:end]]))
        smoothed.append((avg_x, avg_y))
    return smoothed


model_path = 'weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Не вдалося відкрити відео.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter('output_tracked.mp4', fourcc, fps, (frame_width, frame_height))

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

    objects, trajectories = ct.update(boxes)

    for objectID, (position, bbox) in objects.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {objectID}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        pts = trajectories[objectID]
        smoothed_pts = smooth_trajectory(pts, window=5)
        for i in range(1, len(smoothed_pts)):
            cv2.line(frame, smoothed_pts[i - 1], smoothed_pts[i], (0, 0, 255), 2)

    output_video.write(frame)

    cv2.imshow("Відео - відстеження", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()

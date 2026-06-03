# importing the module
import math
import os
import pickle
import threading
import time
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from Detector import Detector
import pygame
from progress.bar import Bar

from BoardScene import BoardScene
from Board import Board
from Constants import POINT_BBOXS, POINT_CENTERS, NUM_POINTS, NUM_POINT_HOMOGRAPHY
from Point import Point
from Disk import Disk, Color
from Utils import pointInPoly
from Dice import Dice
from Class import Class
from BoardPosition import BoardPosition
from Snapshot import Snapshot
from Snapshots import Snapshots
from ultralytics import YOLO

class BackgammonCV:
    def __init__(self):

        self.video = 0
        self.total_frames = 0
        self.detection_every_n_frames = 5
        self.fps = 0
        self.duration = 0
        self.frame_index = 0
        self.template_aligned = False
        self.isPlaying = False
        self.isReplaying = False
        self.replay_frame_index = 0
        self.detect_thread = threading.Thread(target=self.detect, args=(0,))
        self.replay_thread = threading.Thread(target=self.replay)

        self.snapshots = Snapshots()

        self.template = []
        self.frame = []
        self.overlay_text = []
        self.overlay = []
        self.transparent_overlay = []

        self.template_width = 0
        self.template_height = 0
        self.image_width = 0
        self.image_height = 0
        self.alpha_overlay = 0.5

        self.points_template = []
        self.points_homography = []
        self.transformation_matrix = []

        self.point_centers = POINT_CENTERS
        self.point_bboxs = POINT_BBOXS

        self.board = Board()
        self.prev_board = Board()
        self.board_scene = 0

        self.p_min = 0.2
        self.threshold_nms = 0.3
        self.detector = Detector(self.p_min, self.threshold_nms)

        self.bar = Bar("Processing", max=20)
        self.replayBar = Bar("Replay", max=20)
        self.saveBar = Bar("Saving", max=20)

        self.initBoardUI()
        self.loadTemplate()

    def initBoardUI(self):
        pygame.init()
        # UI absolute window position
        os.environ["SDL_VIDEO_WINDOW_POS"] = "%d, %d" % (2, 100)
        self.board_scene = BoardScene()

    def loadTemplate(self):
        self.template = cv2.imread("../data/images/test/template.jpg")
        self.template_height, self.template_width, _ = self.template.shape
        self.points_template = [(0, 0), (self.template_width, 0), (self.template_width, self.template_height), (0, self.template_height)]
        print(f"loadTemplate")

    def alignTemplate(self, image):

        if isinstance(image, str):
            self.frame = cv2.imread(image)
        else:
            self.frame = image

        self.board_scene.alligning()
        self.overlay_text = self.frame.copy()
        self.image_height, self.image_width, _ = self.frame.shape

        self.points_homography = []

        # Add points to board
        for i in range(len(self.point_bboxs)):
            point = Point((i + 1), self.point_centers[i], bbox=self.point_bboxs[i])
            self.board.addPoint(point)

        # displaying the image
        cv2.imshow("Source", self.frame)
        # cv2.setMouseCallback("Source", self.mouseEvent)
        # detecting bounding box of frame
        model = YOLO("border_model.pt")

        # Read image
        # image = cv2.imread("train/images/train_01_jpg.rf.00ae96d4d1829ebffe554f04e505b60e.jpg")
        image = self.frame.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Run inference on GPU
        results = model.predict(
            source=enhanced,
            device=0,      # Use RTX 3060
            conf=0.25
        )
        
        # Get bounding boxes in xyxy format
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            xyxy = results[0].boxes.xyxy
            print("\nBounding boxes (xyxy):")
            print(xyxy)
            print(f"Number of detections: {len(xyxy)}")
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                print(f"Box {i}: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")
                
                # Extract 4 corner points
                # self.points_homography = [
                #     (x1, y1),      # top-left
                #     (x2, y1),      # top-right
                #     (x2, y2),      # bottom-right
                #     (x1, y2),      # bottom-left
                # ]
                self.points_homography.append((int(x1.item()), int(y1.item())))
                self.points_homography.append((int(x2.item()), int(y1.item())))
                self.points_homography.append((int(x2.item()), int(y2.item())))
                self.points_homography.append((int(x1.item()), int(y2.item())))

                print(f"  Corner points: {self.points_homography}")
                break
            
            for cords in self.points_homography:
                x, y = cords
                cv2.putText(self.overlay_text, str(cords[0]) + ", " + str(y), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(self.overlay_text, (x, y), 2, (0, 0, 255), 2)
            cv2.imshow("Source", self.overlay_text)

            # print(points_img)
            self.overlay_text = self.frame.copy()

            self.board.bbox = self.points_homography.copy()
            print(f"board.bbox: {self.board.bbox}")

            # Draw
            if( len(self.points_homography) == NUM_POINT_HOMOGRAPHY ):
                for cords in self.points_homography:
                    x, y = cords
                    print(f"cords: {cords}")
                    cv2.circle(self.overlay_text, (x, y), 2, (0, 0, 255), 2)

                # MATRIX TRANFORM --------------------------------------------------------------

                # Find matrix
                source = np.asarray(self.points_template, dtype=np.float32)
                # print(f"   ##### source: {source}")
                destination = np.asarray(self.points_homography, dtype=np.float32)
                # print(f"   ##### destination: {destination}")
                self.transformation_matrix = cv2.getPerspectiveTransform(source, destination)
                # print(f"self.transformation_matrix: {self.transformation_matrix}")

                image_out = cv2.warpPerspective(self.template, self.transformation_matrix, (self.image_width, self.image_height))
                # print(f"image_out: {image_out}")
                
                # Apply transofmration to all point's bbox
                for point in self.board.points:
                    point.bbox_warped = np.asarray(point.bbox_warped, dtype=np.float32)
                    point.bbox_warped = point.bbox_warped.reshape((-1, 1, 2))

                    point.bbox_warped = cv2.perspectiveTransform(point.bbox_warped, self.transformation_matrix)

                    point.bbox_warped = np.array(point.bbox_warped, dtype=np.int32)

                    # point.center = self.warp_point(point.center, self.transformation_matrix)

                    # Draw
                    overlay_copy = self.overlay_text.copy()
                    cv2.fillConvexPoly(overlay_copy, point.bbox_warped, (255, 255, 255))

                    self.overlay_text = cv2.addWeighted(overlay_copy, self.alpha_overlay, self.overlay_text, 1 - self.alpha_overlay, 0)

                self.overlay = cv2.add(self.overlay_text, image_out)

                # print(self.board)
                self.template_aligned = True
                self.board_scene.update()
                cv2.imshow("Source", self.overlay_text)
                # self.detect(self.image)

    def mouseEvent(self, event, x, y, flags, params):

        # Test pointInPoly
        if event == cv2.EVENT_MOUSEMOVE:
            if len(self.points_homography) == NUM_POINT_HOMOGRAPHY:
                for point in self.board.points:
                    if pointInPoly((x, y), point.bbox_warped):
                        # print("inside " + str(point.id))
                        overlay_copy = self.overlay.copy()
                        if point.id == 25:
                            cv2.fillConvexPoly(overlay_copy, point.bbox_warped, (0, 0, 255))
                        else:
                            cv2.fillConvexPoly(overlay_copy, point.bbox_warped, (127))
                        self.overlay_text = cv2.addWeighted(overlay_copy, self.alpha_overlay, self.overlay, 1 - self.alpha_overlay, 0)
                        cv2.imshow("Source", self.overlay_text)

        if event == cv2.EVENT_LBUTTONDOWN:

            self.points_homography.append((x, y))

            if len(self.points_homography) > NUM_POINT_HOMOGRAPHY:

                self.board.reset()
                self.board_scene.updateBoard(self.board)
                self.points_homography.clear()
                self.overlay_text = self.frame.copy()

            for cords in self.points_homography:
                x, y = cords
                cv2.putText(self.overlay_text, str(cords[0]) + ", " + str(y), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(self.overlay_text, (x, y), 2, (0, 0, 255), 2)
            cv2.imshow("Source", self.overlay_text)

            # print((x, y))

            # Align template----------------------------------------------------------------------------------

            if len(self.points_homography) == NUM_POINT_HOMOGRAPHY:
                # print(points_img)
                self.overlay_text = self.frame.copy()

                self.board.bbox = self.points_homography.copy()

                # Draw
                for cords in self.points_homography:
                    x, y = cords
                    cv2.circle(self.overlay_text, (x, y), 2, (0, 0, 255), 2)

                # MATRIX TRANFORM --------------------------------------------------------------

                # Find matrix
                source = np.asarray(self.points_template, dtype=np.float32)
                destination = np.asarray(self.points_homography, dtype=np.float32)
                print(f"##### source: {source}")
                print(f"##### destination: {destination}")
                self.transformation_matrix = cv2.getPerspectiveTransform(source, destination)
                print(f"self.transformation_matrix: {self.transformation_matrix}")

                image_out = cv2.warpPerspective(self.template, self.transformation_matrix, (self.image_width, self.image_height))
                print(f"image_out: {image_out}")

                # Apply transofmration to all point's bbox
                for point in self.board.points:
                    point.bbox_warped = np.asarray(point.bbox_warped, dtype=np.float32)
                    point.bbox_warped = point.bbox_warped.reshape((-1, 1, 2))

                    point.bbox_warped = cv2.perspectiveTransform(point.bbox_warped, self.transformation_matrix)

                    point.bbox_warped = np.array(point.bbox_warped, dtype=np.int32)

                    # point.center = self.warp_point(point.center, self.transformation_matrix)

                    # Draw
                    overlay_copy = self.overlay_text.copy()
                    cv2.fillConvexPoly(overlay_copy, point.bbox_warped, (255, 255, 255))

                    self.overlay_text = cv2.addWeighted(overlay_copy, self.alpha_overlay, self.overlay_text, 1 - self.alpha_overlay, 0)

                self.overlay = cv2.add(self.overlay_text, image_out)

                # print(self.board)
                self.template_aligned = True
                self.board_scene.update()
                cv2.imshow("Source", self.overlay)
                # self.detect(self.image)

        # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points_homography) > 0:
                self.points_homography.pop()
                self.board.reset()
            self.overlay_text = self.frame.copy()
            for cords in self.points_homography:
                x, y = cords
                cv2.putText(self.overlay_text, str(cords[0]) + ", " + str(y), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(self.overlay_text, (x, y), 2, (0, 0, 255), 2)
            cv2.imshow("Source", self.overlay_text)

    def detectThread(self, image):
        if self.detect_thread.is_alive():
            self.detect_thread.join()
        self.detect_thread = threading.Thread(target=self.detect, args=(image,))
        self.detect_thread.start()

    def detect(self, image):

        while True:
            if not self.template_aligned:
                print("\nFirst you need to align the template. Click clockwise on the 4 extreme points of the board starting from the top left one\n")
                return

            if self.isPlaying:
                self.bar.goto(self.frame_index + 1)
            else:
                self.board_scene.detecting()

            # if isinstance(image, str):
            #     self.frame = cv2.imread(image)
            # else:
            #     self.frame = image

            # cv2.imshow("Source", self.frame)

            self.board.clear()

            self.image_height, self.image_width, _ = self.frame.shape

            self.detector.detect(self.frame)

            self.overlay = self.detector.drawResult()
            self.transparent_overlay = self.detector.drawBboxs()

            self.board.dices = []  # self.board.clear()

            # Generate objects from YOLO detection ---------------------------------------------------------------
            for i in range(len(self.detector.centers)):

                # DISK WHITE
                if self.detector.class_numbers[i] == Class.DISK_WHITE:
                    newDisk = Disk(self.detector.centers[i], self.detector.confidences[i], Color.BLACK)
                    self.board.addDisk(newDisk)

                # DISK BLACK
                if self.detector.class_numbers[i] == Class.DISK_BLACK:
                    newDisk = Disk(self.detector.centers[i], self.detector.confidences[i], Color.WHITE)
                    self.board.addDisk(newDisk)

                # DICE
                if self.detector.class_numbers[i] < Class.DISKS:
                    newDice = Dice(self.detector.class_numbers[i], self.detector.centers[i], self.detector.confidences[i])

                    # Dice position binarization
                    if newDice.center[0] >= self.board.getBar().bbox_warped[0][0][0]:
                        newDice.board_position = BoardPosition.RIGHT
                    else:
                        newDice.board_position = BoardPosition.LEFT

                    self.board.addDice(newDice)

            # print("Dices: " + str(self.board.dices))

            self.board_scene.updateBoard(self.board)
            # save board state in snapshots
            if( self.prev_board != self.board ):
                print(f"Current: {str(self.board)}")
                text_file = open(f"frame_{self.frame_index}.txt", "w")
                s = str(self.board) + "\n"
                text_file.write(s)
                text_file.close()
                self.prev_board = self.board.copy()

            # Add snapshot
            snapshot = Snapshot(self.frame_index, self.board.copy(), self.frame.copy(), self.transparent_overlay.copy())
            self.snapshots.addSnapshot(snapshot)
            # print(self.snapshots)

            cv2.imshow("Source", self.overlay)

            if not self.isPlaying:
                break
            # time.sleep(0.1)
            self.nextFrame()
            # self.detect(self.frame)

    # def warp_point(self, center, M):
    #     x, y = center
    #     d = M[2, 0] * x + M[2, 1] * y + M[2, 2]

    #     return (
    #         int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d),  # x
    #         int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d),  # y
    #     )

    def saveReplay(self):
        # print("\nSaving replay")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("../data/videos/result/replay.mp4", fourcc, 20.0, (1280, 720))
        s = ""
        currentSnapshotIndex = 0
        # overlay = np.zeros((self.image_height, self.image_width, 3), np.uint8)
        # frame_index = self.snapshots.getLastSnapshot().frame_index
        # print(frame_index)
        self.saveBar.max = self.total_frames
        self.saveBar.fill = "\u258C"
        self.saveBar.suffix = "%(index)d/%(max)d (%(percent)d%%) [remaining %(eta_time)s]"
        self.saveBar.goto(1)
        for i in range(self.total_frames - 1):
            self.saveBar.next()
            self.video.set(1, i)
            _, frame = self.video.read()

            # if frame processed and saved in snapshot add overlay
            if i == self.snapshots.getSnapshot(currentSnapshotIndex).frame_index and currentSnapshotIndex < self.snapshots.getLastSnapshot().id:
                currentSnapshotIndex += 1
                # self.board_scene.updateBoard(self.snapshots.getSnapshot(currentSnapshotIndex - 1).board)

            overlay = self.snapshots.getSnapshot(currentSnapshotIndex - 1).overlay
            frame = cv2.add(overlay, frame)
            out.write(frame)

        # for snapshot in self.snapshots.snapshots:
        #     # cv2.imshow("Snaphsots", snapshot.overlay)
        #     # cv2.imshow("Snaphsots", snapshot.overlay)
        #     # time.sleep(0.2)
        #     out.write(snapshot.overlay)
        #     # self.board_scene.updateBoard(snapshot.board)
        #     s += "Frame " + str(snapshot.frame_index) + "\n"
        #     s += str(snapshot.board) + "\n\n"
        out.release()
        # text_file = open("snapshots.txt", "w")
        # text_file.write(s)
        # text_file.close()
        print("\nReplay saved in replay.mp4 and snapshots.txt ")

    def replayThread(self):
        if self.replay_thread.is_alive():
            self.replay_thread.join()
            # print("Stop replay")
        else:
            self.replay_thread = threading.Thread(target=self.replay)
            self.replay_thread.start()

    def replay(self):
        # print("\nReplay")
        # for snapshot in self.snapshots.snapshots:
        #     # cv2.imshow("Snaphsots", snapshot.overlay)
        #     cv2.imshow("Source", snapshot.overlay)
        #     # time.sleep(0.2)
        #     self.board_scene.updateBoard(snapshot.board)
        #     time.sleep(0.06)
        currentSnapshotIndex = 0
        # overlay = np.zeros((self.image_height, self.image_width, 3), np.uint8)
        max_frame_index = self.snapshots.getLastSnapshot().frame_index
        # print(max_frame_index)
        self.replayBar.max = max_frame_index
        self.replayBar.fill = "\u258C"
        self.replayBar.suffix = "%(index)d/%(max)d (%(percent)d%%) [remaining %(eta_time)s]"
        self.replayBar.goto(0)

        for i in range(max_frame_index):
            self.replayBar.next()
            self.video.set(1, i)
            _, frame = self.video.read()

            # if frame processed and saved in snapshot add overlay
            if i == self.snapshots.getSnapshot(currentSnapshotIndex).frame_index:
                currentSnapshotIndex += 1
                self.board_scene.updateBoard(self.snapshots.getSnapshot(currentSnapshotIndex - 1).board)

            overlay = self.snapshots.getSnapshot(currentSnapshotIndex - 1).overlay
            frame = cv2.add(overlay, frame)

            cv2.imshow("Source", frame)

            time.sleep(0.016)
        # print("\nEnd replay")

    def nextFrame(self):
        self.frame_index += self.detection_every_n_frames
        self.frame_index = min(self.frame_index, self.total_frames - 1)
        if self.frame_index == self.total_frames - 1:
            self.stop()
            self.video.set(1, self.frame_index)
            _, self.frame = self.video.read()
            if not self.isPlaying:
                cv2.imshow("Source", self.frame)
            # time.sleep(3)
            self.saveSnapshots()
            # self.bar.finish()
            # self.replay()
        else:
            self.video.set(1, self.frame_index)
            _, self.frame = self.video.read()
            if not self.isPlaying:
                cv2.imshow("Source", self.frame)
        print("Current frame: " + str(self.frame_index) + "/" + str(self.total_frames - 1))
        # self.detect(self.frame)
        # return self.frame_index

    def previousFrame(self):
        self.frame_index -= self.detection_every_n_frames
        self.frame_index = max(0, self.frame_index)
        self.video.set(1, self.frame_index)
        _, self.frame = self.video.read()
        if not self.isPlaying:
            cv2.imshow("Source", self.frame)
        print("Current frame: " + str(self.frame_index) + "/" + str(self.total_frames - 1))
        # print(type(self.frame))
        # self.detect(self.frame)
        # return self.frame_index

    def togglePlay(self):
        if self.isPlaying:
            self.isPlaying = False
            self.stop()
        else:
            self.isPlaying = True
            self.play()

    # def toggleReplay(self):
    #     if self.isReplaying:
    #         self.isReplaying = False
    #         self.stopReplay()
    #     else:
    #         self.isReplaying = True
    #         self.replay()

    def play(self):
        print("Play")
        self.isPlaying = True
        self.detectThread(self.frame)

    def stop(self):
        print("Stop")
        self.isPlaying = False

    def saveSnapshots(self):
        print("\nSaving snapshots")
        picklefile = open("../data/snapshots/snapshots.bcv", "wb")
        pickle.dump(self.snapshots, picklefile)
        picklefile.close()
        print("Snapshots saved")

    def loadSnapshots(self):
        print("\n\nLoading snapshots")
        picklefile = open("../data/snapshots/snapshots.bcv", "rb")
        # unpickle the dataframe
        snapshots = pickle.load(picklefile)
        picklefile.close()
        # close file
        self.snapshots = snapshots
        # print(self.snapshots)
        print(str(len(self.snapshots.snapshots)) + " snapshots loaded\n")

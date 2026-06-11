# importing the module
import math
import os
import pickle
import threading
import time
import cv2
from pathlib import Path
import numpy as np
from shapely.geometry import Point, Polygon
from Cube import Cube
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
from orientation_solver import OrientationSolver, BoardState, renumber_points_by_orientation
from ultralytics import YOLO
import mediapipe as mp

from checker_color_classifier import classify_checkers_by_brightness
import torch

class BackgammonCV:
    def __init__(self):

        self.turn = 0
        self.is_white_bearing = False
        self.is_black_bearing = False
        self.white_borne_count = 0
        self.black_borne_count = 0
        self.video = 0
        self.total_frames = 0
        self.detection_every_n_frames = 10
        self.fps = 0
        self.duration = 0
        self.frame_index = 0
        self.template_aligned = False
        self.isPlaying = False
        self.isReplaying = False
        self.replay_frame_index = 0
        self.detect_thread = threading.Thread(target=self.detect, args=(0,))
        self.replay_thread = threading.Thread(target=self.replay)

        self.mp_hands = mp.solutions.hands
        # self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7)

        # Load once and reuse to avoid repeated model allocations.
        self.border_model = YOLO("border_model.pt")


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
        # Stability filtering
        self.candidate_board = None
        self.candidate_count = 0

        # Require board to remain stable for N frames
        self.STABLE_FRAMES = 1

        self.movements = []
        self.board_scene = 0

        # Orientation inference state.
        self.orientation_solver = OrientationSolver()
        self.orientation_scores = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.board_orientation = None
        self.orientation_locked = False
        self.orientation_score_margin = 15.0

        self.p_min = 0.2
        self.threshold_nms = 0.3
        self.detector = Detector(self.p_min, self.threshold_nms)

        self.bar = Bar("Processing", max=20)
        self.replayBar = Bar("Replay", max=20)
        self.saveBar = Bar("Saving", max=20)

        self.initBoardUI()
        self.loadTemplate()

    def is_board_normal(self, board):
        """
        Check if the board status is normal.

        Rules:
        - If the total count of checkers is 30, the board is normal.
        - If all checkers are on the player's home board, the board is normal (bearing off phase).
        - If all checkers are NOT on the home board AND the count is not 30, the board is abnormal.

        Returns:
            True if the board status is normal, False otherwise.
        """
        # # Count total checkers on the board
        # points_by_id = {point.id: point for point in board.points}
        # total_checkers = 0

        # for point_id in range(1, 26):  # Include all points 1-24 and bar (25)
        #     point = points_by_id.get(point_id)
        #     if point is not None:
        #         total_checkers += len(point.disks)

        # # If all 30 checkers are present, board is normal
        # if total_checkers == 30:
        #     return True

        white_count_on_points = board.get_count_on_points(Color.WHITE)
        black_count_on_points = board.get_count_on_points(Color.BLACK)
        if white_count_on_points == 15 and black_count_on_points == 15:
            # If all checkers are on home boards, board is bearing off phase
            if board.are_all_white_on_home_board():
                print(f"\t[is_board_normal]: are_all_white_on_home_board()=True")
                self.is_white_bearing = True
            if board.are_all_black_on_home_board():
                print(f"\t[is_board_normal]: are_all_black_on_home_board()=True")
                self.is_black_bearing = True

            return True
        # prev_white_count_on_points = prev_board.get_count_on_points(Color.WHITE)
        # prev_black_count_on_points = prev_board.get_count_on_points(Color.BLACK)

        print(f"\t\tis_white_bearing: {self.is_white_bearing}, white_count_on_points: {white_count_on_points}, white_borne_count: {self.white_borne_count}")
        print(f"\t\tis_black_bearing: {self.is_black_bearing}, black_count_on_points: {black_count_on_points}, black_borne_count: {self.black_borne_count}")
        if self.is_white_bearing:
            if white_count_on_points + self.white_borne_count < 13:
                return False
            if self.is_black_bearing:
                if black_count_on_points + self.black_borne_count < 13:
                    return False
                else:
                    return True
            else:
                if black_count_on_points < 15:
                    return False
                else:
                    return True
        else:
            if white_count_on_points < 15:
                return False
            
            if self.is_black_bearing:
                if black_count_on_points + self.black_borne_count < 13:
                    return False
                else:
                    return True
            else:
                if black_count_on_points < 15:
                    return False
                else:
                    return True

        # Otherwise, board is abnormal (missing checkers or in invalid state)
        return False

    def _board_to_state(self, board):
        """Convert a `Board` to an orientation-solver `BoardState`.

        The first 24 points are treated as the raw point order. Point id 25 is
        treated as the bar. Borne-off counts come from the tracked counters on
        `BackgammonCV`.
        """
        white = [0] * 24
        black = [0] * 24
        bar_white = 0
        bar_black = 0

        for raw_idx, point in enumerate(board.points[:24]):
            for disk in point.disks:
                if disk.color == Color.WHITE:
                    white[raw_idx] += 1
                else:
                    black[raw_idx] += 1

        for point in board.points:
            if point.id == 25:
                for disk in point.disks:
                    if disk.color == Color.WHITE:
                        bar_white += 1
                    else:
                        bar_black += 1
                break

        return BoardState(
            white=white,
            black=black,
            bar_white=bar_white,
            bar_black=bar_black,
            borne_white=self.white_borne_count,
            borne_black=self.black_borne_count,
        )

    def _apply_orientation_to_board(self, board, orientation_index):
        """Return a copy of `board` renumbered to standard point order.

        The point list is reordered so that point 1..24 are in standard order
        and point IDs are rewritten to match the standard numbering.
        """
        if orientation_index is None:
            return board.copy()

        oriented = board.copy()
        raw_points = oriented.points[:24]
        if len(raw_points) == 24:
            oriented_points = renumber_points_by_orientation(raw_points, orientation_index)
        else:
            oriented_points = [p.copy() for p in raw_points]
            for idx, point in enumerate(oriented_points, start=1):
                point.id = idx

        bar_points = [p.copy() for p in oriented.points[24:] if p.id == 25]
        if bar_points:
            bar_points[0].id = 25

        oriented.points = oriented_points + bar_points
        return oriented

    def _lock_orientation_if_possible(self, prev_board, current_board):
        """Score the four orientations and lock/renumber when confidence is high."""
        prev_state = self._board_to_state(prev_board)
        curr_state = self._board_to_state(current_board)

        dice = None
        if len(current_board.dices) == 2:
            dice = (current_board.dices[0].value, current_board.dices[1].value)

        transition_scores, best_idx = self.orientation_solver.score_orientations(
            [prev_state, curr_state],
            dice_sequence=[dice],
        )

        for idx, value in transition_scores.items():
            self.orientation_scores[idx] += value

        best_total_idx = max(self.orientation_scores.items(), key=lambda kv: kv[1])[0]
        total_scores = sorted(self.orientation_scores.values(), reverse=True)

        if len(total_scores) >= 2 and (total_scores[0] - total_scores[1]) >= self.orientation_score_margin:
            self.board_orientation = best_total_idx
            self.orientation_locked = True
            print(
                f"[processStableBoard] Orientation locked to {best_total_idx} "
                f"with scores {self.orientation_scores}"
            )

            # Renumber history so future move inference uses standard point ids.
            self.prev_board = self._apply_orientation_to_board(self.prev_board, best_total_idx)
            current_board = self._apply_orientation_to_board(current_board, best_total_idx)
            self.movements = [self._apply_orientation_to_board(board, best_total_idx) for board in self.movements]
            return current_board

        # Keep the best total orientation so far for later transitions.
        self.board_orientation = best_total_idx
        return current_board

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

        # Prevent point duplication if template alignment is re-run.
        self.board.points = []

        # Add points to board
        for i in range(len(self.point_bboxs)):
            point = Point((i + 1), self.point_centers[i], bbox=self.point_bboxs[i])
            self.board.addPoint(point)

        # displaying the image
        cv2.imshow("Source", self.frame)
        cv2.setMouseCallback("Source", self.mouseEvent)
        # detecting bounding box of frame
        model = self.border_model

        # Read image
        # image = cv2.imread("train/images/train_01_jpg.rf.00ae96d4d1829ebffe554f04e505b60e.jpg")
        image = self.frame.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Run yolov11 inference the board on GPU
        if torch.cuda.is_available():
            results = model.predict(
                source=enhanced,
                device=0,
                conf=0.25
            )
        else:
            results = model.predict(
                source=enhanced,
                conf=0.25
            )

        # Get bounding boxes in xyxy format
        result = results[0]
        class_names = model.names
        detections  = []
        if result.keypoints is None or len(result.boxes) == 0:
            print("No detections found.")
            return
        
        boxes  = result.boxes.xywh.cpu().numpy()   # (N, 4) cx cy w h  – pixel coords
        kps    = result.keypoints.xy.cpu().numpy() # (N, K, 2)
        confs  = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        num_kps = kps.shape[1]
        if num_kps < 4:
            print(f"[WARN] Model has only {num_kps} keypoint(s); expected 4.")

        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            det = {
                "class_id":   cls_ids[i],
                "class_name": class_names[cls_ids[i]],
                "confidence": float(confs[i]),
                "cx": float(cx), "cy": float(cy),
                "w":  float(w),  "h":  float(h),
            }
            for k in range(min(4, num_kps)):
                det[f"x{k+1}"] = float(kps[i, k, 0])
                det[f"y{k+1}"] = float(kps[i, k, 1])
            detections.append(det)
        
        for idx, d in enumerate(detections):
            print(
                f"{idx:<4} {d['class_name']:<15} {d['confidence']:>6.3f}  "
                f"{d['cx']:>8.2f} {d['cy']:>8.2f} {d['w']:>8.2f} {d['h']:>8.2f}  "
                f"{d['x1']:>8.2f} {d['y1']:>8.2f}  "
                f"{d['x2']:>8.2f} {d['y2']:>8.2f}  "
                f"{d['x3']:>8.2f} {d['y3']:>8.2f}  "
                f"{d['x4']:>8.2f} {d['y4']:>8.2f}"
            )
            self.points_homography.append((int(d['x1']), int(d['y1'])))
            self.points_homography.append((int(d['x2']), int(d['y2'])))
            self.points_homography.append((int(d['x3']), int(d['y3'])))
            self.points_homography.append((int(d['x4']), int(d['y4'])))

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
                # print(f"cords: {cords}")
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

        print(f"Mouse x,y = ({x}, {y})\r", end="")
        if ( self.isPlaying == True ):
            return

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

    def getMovedCheckers(self, previous_points, current_points):
        """
        Compare two Point() arrays and infer moved checkers.

        Returns:
            {
                "white": [{"from": <point_id>, "to": <point_id>}],
                "black": [{"from": <point_id>, "to": <point_id>}],
                "unmatched": {
                    "white": {"from": [<point_id>], "to": [<point_id>]},
                    "black": {"from": [<point_id>], "to": [<point_id>]}
                }
            }

        Notes:
        - Point id 25 is the bar.
        - Movement is inferred from checker count differences per color, because
          individual checkers do not have stable IDs.
        """
        def _count_by_color(points):
            counts = {}
            for point in points:
                white_count = 0
                black_count = 0
                for disk in point.disks:
                    if disk.color == Color.WHITE:
                        white_count += 1
                    else:
                        black_count += 1
                counts[point.id] = {
                    "white": white_count,
                    "black": black_count,
                }
            return counts
        def _are_all_on_homeboard(points, color):
            if color == Color.WHITE:
                home_board_range = range(1, 7)  # points 1-6
            elif color == Color.BLACK:
                home_board_range = range(19, 25)  # points 19-24
            else:
                return False

            points_by_id = {point.id: point for point in points}

            # Check all points outside the home board
            for point_id in range(1, 26):  # Check points 1-24 and bar (25)
                if point_id in home_board_range:
                    continue  # Skip home board points
                
                point = points_by_id.get(point_id)
                if point is not None:
                    for disk in point.disks:
                        if disk.color == color:
                            return False  # Found a checker outside home board
            return True

        prev_counts = _count_by_color(previous_points)
        curr_counts = _count_by_color(current_points)
        all_point_ids = sorted(set(prev_counts.keys()) | set(curr_counts.keys()))

        if _are_all_on_homeboard(current_points, Color.WHITE):
            print("\t\t[getMovedCheckers] All white checkers are on home board, white is bearing off.")
            self.is_white_bearing = True
        if _are_all_on_homeboard(current_points, Color.BLACK):
            print("\t\t[getMovedCheckers] All black checkers are on home board, black is bearing off.")
            self.is_black_bearing = True

        result = {
            "white": [],
            "black": [],
            "unmatched": {
                "white": {"from": [], "to": []},
                "black": {"from": [], "to": []},
            },
        }

        for color_name in ("white", "black"):
            from_points = []
            to_points = []

            for point_id in all_point_ids:
                prev_value = prev_counts.get(point_id, {}).get(color_name, 0)
                curr_value = curr_counts.get(point_id, {}).get(color_name, 0)
                # print(f"point_id: {point_id} color: {color_name} prev_value: {prev_value} curr_value: {curr_value}")
                delta = curr_value - prev_value

                if delta < 0:
                    from_points.extend([point_id] * (-delta))
                elif delta > 0:
                    to_points.extend([point_id] * delta)

            paired_moves = min(len(from_points), len(to_points))
            for i in range(paired_moves):
                result[color_name].append({
                    "from": from_points[i],
                    "to": to_points[i],
                })

            if len(from_points) > paired_moves:
                # this means bearing off, only count as borne-off if all checkers are on the home board.
                result["unmatched"][color_name]["from"] = from_points[paired_moves:]
                if color_name == "white" and self.is_white_bearing == True and len(result['unmatched'][color_name]['from']) <= 2:
                    print(f"\t\t[getMovedCheckers] Bear-off white from points: {result['unmatched']['white']['from']}")
                    self.white_borne_count += len(result['unmatched'][color_name]['from'])
                if color_name == "black" and self.is_black_bearing == True and len(result['unmatched'][color_name]['from']) <= 2:
                    print(f"\t\t[getMovedCheckers] Bear-off black from points: {result['unmatched']['black']['from']}")
                    self.black_borne_count += len(result['unmatched'][color_name]['from'])

            if len(to_points) > paired_moves:
                # this means coming back from bar or hidden
                result["unmatched"][color_name]["to"] = to_points[paired_moves:]
                if color_name == "white" and self.is_white_bearing == True and len(result['unmatched'][color_name]['to']) <= 2:
                    print(f"\t\t[getMovedCheckers] Resume white from hidden: {result['unmatched']['white']['to']}")
                    self.white_borne_count -= len(result['unmatched'][color_name]['to'])
                if color_name == "black" and self.is_black_bearing == True and len(result['unmatched'][color_name]['to']) <= 2:
                    print(f"\t\t[getMovedCheckers] Resume black from hidden: {result['unmatched']['black']['to']}")
                    self.black_borne_count -= len(result['unmatched'][color_name]['to'])

        return result

    def processStableBoard(self, stable_board):

        # If the orientation has already been locked, normalize the incoming
        # board before any comparison so all later move detection works in the
        # standard point numbering.
        if self.orientation_locked and self.board_orientation is not None:
            stable_board = self._apply_orientation_to_board(stable_board, self.board_orientation)
        elif len(self.prev_board.points) > 0:
            stable_board = self._lock_orientation_if_possible(self.prev_board, stable_board)

        # Ignore identical board
        if stable_board.points == self.prev_board.points:
            # print(f"stable_board.dices={stable_board.dices}, prev_board.dices={self.prev_board.dices}")
            if stable_board.is_dice_changed(self.prev_board) == False:
                print("[processStableBoard] Stable board identical to previous board, ignoring.")
                return
            
            # if dice changed
            if len(self.prev_board.dices) == 0 and len(stable_board.dices) == 2:
                # player just rolled the dice, but no movement, turn over.
                print("[processStableBoard] player just rolled the dice, but no movement, turn over.")
                stable_board.turn = self.prev_board.turn + 1
                stable_board.player = Color.BLACK if self.prev_board.player == Color.WHITE else Color.WHITE
                self.movements.append(stable_board.copy())
                with open(f"frame_{self.frame_index}.txt", "w") as f:
                    f.write(stable_board.exportJSON(as_string=True))
                self.prev_board = stable_board.copy()
                return

        moved = self.getMovedCheckers(self.prev_board.points, stable_board.points)

        print(f"\n\t[processStableBoard]############# moved: {moved}")

        if len(moved['white']) > 0 and len(moved['black']) > 0:
            moved_white_to = set(move['to'] for move in moved['white'])
            moved_black_to = set(move['to'] for move in moved['black'])
            if 25 in moved_white_to:
                print("\t\t[processStableBoard] white moved to bar")
                print("\t\t[processStableBoard] black moved from " + str(moved['black'][0]['from']) + " to " + str(moved['black'][0]['to']))
                stable_board.player = Color.BLACK
            elif 25 in moved_black_to:
                print("\t\t[processStableBoard] black moved to bar")
                print("\t\t[processStableBoard] white moved from " + str(moved['white'][0]['from']) + " to " + str(moved['white'][0]['to']))
                stable_board.player = Color.WHITE
            else:
                print("\t\t[processStableBoard] both white and black moved, ignoring.")
                self.prev_board = stable_board.copy()
                return

            print(f"\t\t[processStableBoard] stable_board.dices: {stable_board.dices}, prev_board.dices: {self.prev_board.dices}")

            # if len(stable_board.dices) == 2 and stable_board.is_dice_changed(self.prev_board) and stable_board.player != self.prev_board.player:
            if stable_board.player != self.prev_board.player:
                stable_board.turn = self.prev_board.turn + 1
            else:
                stable_board.turn = self.prev_board.turn
                if len(stable_board.dices) < 2:
                    stable_board.dices = self.prev_board.dices.copy()
                self.movements.pop()

        elif len(moved['white']) > 0:
            print("\t\t[processStableBoard] white moved from " + str(moved['white'][0]['from']) + " to " + str(moved['white'][0]['to']))
            stable_board.player = Color.WHITE

            print(f"\t\t[processStableBoard] stable_board.dices: {stable_board.dices}, prev_board.dices: {self.prev_board.dices}")
            # if len(stable_board.dices) == 2 and stable_board.is_dice_changed(self.prev_board) and stable_board.player != self.prev_board.player:
            if stable_board.player != self.prev_board.player:
                stable_board.turn = self.prev_board.turn + 1
            else:
                stable_board.turn = self.prev_board.turn
                if len(stable_board.dices) < 2:
                    stable_board.dices = self.prev_board.dices.copy()
                self.movements.pop()

        elif len(moved['black']) > 0:
            print("\t\t[processStableBoard] black moved from " + str(moved['black'][0]['from']) + " to " + str(moved['black'][0]['to']))
            stable_board.player = Color.BLACK

            print(f"\t\t[processStableBoard] stable_board.dices: {stable_board.dices}, prev_board.dices: {self.prev_board.dices}")
            # if len(stable_board.dices) == 2 and stable_board.is_dice_changed(self.prev_board) and stable_board.player != self.prev_board.player:
            if stable_board.player != self.prev_board.player:
                stable_board.turn = self.prev_board.turn + 1
            else:
                stable_board.turn = self.prev_board.turn
                if len(stable_board.dices) < 2:
                    stable_board.dices = self.prev_board.dices.copy()
                self.movements.pop()

        elif len(moved['unmatched']['white']['from']) > 0 or len(moved['unmatched']['black']['from']) > 0:
            # unmatched movement from, could be due to bearing-off.
            print("\t\t[processStableBoard] unmatched movement, could be due to bearing-off.")
            if len(moved['unmatched']['white']['from']) > 0:
                stable_board.player = Color.WHITE
            elif len(moved['unmatched']['black']['from']) > 0:
                stable_board.player = Color.BLACK

            print(f"\t\t[processStableBoard] stable_board.dices: {stable_board.dices}, prev_board.dices: {self.prev_board.dices}")
            stable_board.turn = self.prev_board.turn + 1
            if len(stable_board.dices) < 2:
                stable_board.dices = self.prev_board.dices.copy()
        elif len(moved['unmatched']['white']['to']) > 0 or len(moved['unmatched']['black']['to']) > 0:
            # unmatched movement to, could be due to hidden checkers coming back or detection error, resuming.
            print("\t\t[processStableBoard] unmatched movement to, could be due to hidden checkers coming back or detection error, resuming.")
            stable_board.player = self.prev_board.player
            stable_board.turn = self.prev_board.turn - 1 if self.prev_board.turn > 0 else 0

            print(f"\t\t[processStableBoard] stable_board.dices: {stable_board.dices}, prev_board.dices: {self.prev_board.dices}")
            if len(stable_board.dices) < 2:
                stable_board.dices = self.prev_board.dices.copy()

            self.prev_board = stable_board.copy()
            return
        else:
            print("\t\t[processStableBoard] No movement detected, ignoring.")

            print(f"\t\t[processStableBoard] stable_board.dices: {stable_board.dices}, prev_board.dices: {self.prev_board.dices}")
            return
        
        self.movements.append(stable_board.copy())

        with open(f"frame_{self.frame_index}.txt", "w") as f:
            f.write(stable_board.exportJSON(as_string=True))

        print(
            f"\t[processStableBoard]Stable board accepted. "
            f"Turn={stable_board.turn}, "
            f"Player={stable_board.player}\n"
        )

        self.prev_board = stable_board.copy()
    
    def updateStableBoard(self):

        # if len(self.board.dices) < 2:
        #     return
        
        if not self.is_board_normal(self.board):
            print("[updateStableBoard] Board not normal, ignoring.")
            return

        # First candidate
        if self.candidate_board is None:
            print("[updateStableBoard] First candidate board")
            self.candidate_board = self.board.copy()
            self.candidate_count = 1
            # return

        # Same as candidate
        if self.board.points == self.candidate_board.points:
            print("[updateStableBoard] Candidate board stable for another frame")
            self.candidate_board = self.board.copy()
            self.candidate_count += 1
        else:
            # New candidate board
            print("[updateStableBoard] New candidate board")
            self.candidate_board = self.board.copy()
            self.candidate_count = 1

        # Candidate accepted
        if self.candidate_count >= self.STABLE_FRAMES:

            print("[updateStableBoard] Board stable for " + str(self.candidate_count) + " frames, accepting.")

            # upadte Template board UI
            self.board_scene.updateBoard(self.board)

            if len(self.prev_board.points) == 0:
                # First stable board, initialize prev_board
                print("[updateStableBoard] First stable board, initializing prev_board")
                self.prev_board = self.candidate_board.copy()
                self.movements.append(self.prev_board.copy())
                with open(f"frame_{self.frame_index}.txt", "w") as f:
                    f.write(self.prev_board.exportJSON(as_string=True))
            else:
                self.processStableBoard(self.candidate_board)

            # Prevent reprocessing same state
            self.candidate_count = 0
        else:
            print("[updateStableBoard] Board not stable yet: candidate_count = " + str(self.candidate_count))

    def detect(self, image):

        while True:
            if not self.template_aligned:
                print("\nFirst you need to align the template. Click clockwise on the 4 extreme points of the board starting from the top left one\n")
                return

            # if self.isPlaying:
            #     self.bar.goto(self.frame_index + 1)
            # else:
            #     self.board_scene.detecting()

            # if isinstance(image, str):
            #     self.frame = cv2.imread(image)
            # else:
            #     self.frame = image

            # cv2.imshow("Source", self.frame)

            self.board.clear()

            self.image_height, self.image_width, _ = self.frame.shape

            # detect hands
            if True:
                res = self.hands.process(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                if res.multi_hand_landmarks:
                    print(f"✅ Hand detected! ({len(res.multi_hand_landmarks)} hand(s))")
                    def is_hand_in_board(hand_landmarks, img_w, img_h, board):
                        """Check if ANY landmark of the hand is inside the board."""
                        for lm in hand_landmarks.landmark:
                            x = int(lm.x * img_w)
                            y = int(lm.y * img_h)
                            # print(f"board: {board}, hand landmark: ({x}, {y})")
                            if board[0][0] <= x <= board[2][0] and board[0][1] <= y <= board[2][1]:
                                return True
                        return False
                    in_board = False
                    for hand_landmarks in res.multi_hand_landmarks:
                        # self.mp_draw.draw_landmarks(self.frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        if is_hand_in_board(hand_landmarks, self.image_width, self.image_height, self.board.bbox):
                            print("   🖐️ Hand is inside the board area, skipping detection to avoid occlusion issues.")
                            in_board = True
                            break

                    if in_board:
                        time.sleep(0.01)
                        self.nextFrame()
                        continue

            self.detector.detect(self.frame)

            self.board.dices = []  # self.board.clear()

            checker_results = classify_checkers_by_brightness(self.frame, self.detector.bounding_boxes)
            # Generate objects from YOLO detection ---------------------------------------------------------------
            for i in range(len(self.detector.centers)):
                checker_result = checker_results[i]
                if self.detector.class_numbers[i] == Class.DISK_BLACK or self.detector.class_numbers[i] == Class.DISK_WHITE:

                    # print(f"Checker at {self.detector.centers[i]} classified as {checker_result.label} with brightness {checker_result.brightness:.2f} and average BGR {checker_result.average_bgr}")
                    if checker_result.label == "white":
                        self.detector.class_numbers[i] = Class.DISK_WHITE
                    else:
                        self.detector.class_numbers[i] = Class.DISK_BLACK

                # DISK WHITE
                if self.detector.class_numbers[i] == Class.DISK_WHITE:
                    newDisk = Disk(self.detector.centers[i], self.detector.confidences[i], Color.WHITE)
                    self.board.addDisk(newDisk)

                # DISK BLACK
                if self.detector.class_numbers[i] == Class.DISK_BLACK:
                    newDisk = Disk(self.detector.centers[i], self.detector.confidences[i], Color.BLACK)
                    self.board.addDisk(newDisk)

                # DICE
                if self.detector.class_numbers[i] <= Class.DICE_6:
                    newDice = Dice(self.detector.class_numbers[i], self.detector.centers[i], self.detector.confidences[i])

                    # Dice position binarization
                    if newDice.center[0] >= self.board.getBar().bbox_warped[0][0][0]:
                        newDice.board_position = BoardPosition.RIGHT
                    else:
                        newDice.board_position = BoardPosition.LEFT

                    self.board.addDice(newDice)

                # CUBE
                if self.detector.class_numbers[i] >= Class.CUBE_16 and self.detector.class_numbers[i] <= Class.CUBE_32:
                    newCube = Cube(self.detector.class_numbers[i] - Class.CUBE_16, self.detector.centers[i], self.detector.confidences[i])
                    if pointInPoly((newCube.center[0], newCube.center[1]), self.board.bbox):
                        self.board.setCube(newCube)
                    elif newCube.center[1] < self.board.getBar().bbox_warped[0][0][1] + (self.board.getBar().bbox_warped[2][0][1] - self.board.getBar().bbox_warped[0][0][1]) / 4:
                        newCube.owner = Color.WHITE
                        self.board.setCube(newCube)
                    elif newCube.center[1] > self.board.getBar().bbox_warped[0][0][1] + (self.board.getBar().bbox_warped[2][0][1] - self.board.getBar().bbox_warped[0][0][1]) * 3 / 4:
                        newCube.owner = Color.BLACK
                        self.board.setCube(newCube)

            self.overlay = self.detector.drawResult()
            self.transparent_overlay = self.detector.drawBboxs()

            self.board.calibratePoints()

            # save board state in snapshots
            self.updateStableBoard()

            # Add snapshot
            # snapshot = Snapshot(self.frame_index, self.board.copy(), self.frame.copy(), self.transparent_overlay.copy())
            # self.snapshots.addSnapshot(snapshot)
            # print(self.snapshots)

            cv2.imshow("Source", self.overlay)

            if not self.isPlaying:
                break
            time.sleep(0.01)
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
            ret, frame = self.video.read()
            if not ret:
                continue

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
            ret, frame = self.video.read()
            if not ret:
                continue

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
            ret, self.frame = self.video.read()
            if not ret:
                print("Error reading last frame")
                return
            if not self.isPlaying:
                cv2.imshow("Source", self.frame)

            self.saveMovements()
            # time.sleep(3)
            # self.saveSnapshots()
            # self.bar.finish()
            # self.replay()
        else:
            self.video.set(1, self.frame_index)
            ret, self.frame = self.video.read()
            if not ret:
                print("Error reading frame " + str(self.frame_index))
                return
            if not self.isPlaying:
                cv2.imshow("Source", self.frame)
        print("Current frame: " + str(self.frame_index) + "/" + str(self.total_frames - 1))
        # self.detect(self.frame)
        # return self.frame_index

    def previousFrame(self):
        self.frame_index -= self.detection_every_n_frames
        self.frame_index = max(0, self.frame_index)
        self.video.set(1, self.frame_index)
        ret, self.frame = self.video.read()
        if not ret:
            print("Error reading frame " + str(self.frame_index))
            return
        
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
        with open("../data/snapshots/snapshots.bcv", "wb") as picklefile:
            pickle.dump(self.snapshots, picklefile)
        print("Snapshots saved")

    def saveMovements(self):
        print("\nSaving movements")
        path = Path("../data/movements/movements.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(self.movements) == 0:
            print("No movements to save.")
            return
        with open(path, "w") as f:
            f.write("[\n")
            for movement in self.movements:
                f.write(str(movement.exportJSON(as_string=True)) + "\n\n")
            f.write("]\n")
        print("Movements saved")

    def loadSnapshots(self):
        print("\n\nLoading snapshots")
        with open("../data/snapshots/snapshots.bcv", "rb") as picklefile:
            # unpickle the dataframe
            snapshots = pickle.load(picklefile)
        # close file
        self.snapshots = snapshots
        # print(self.snapshots)
        print(str(len(self.snapshots.snapshots)) + " snapshots loaded\n")

    def close(self):
        self.isPlaying = False

        if self.detect_thread.is_alive():
            self.detect_thread.join(timeout=1)
        if self.replay_thread.is_alive():
            self.replay_thread.join(timeout=1)

        if self.video not in (0, None):
            try:
                self.video.release()
            except Exception:
                pass

        if hasattr(self, "hands") and self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            pygame.quit()
        except Exception:
            pass

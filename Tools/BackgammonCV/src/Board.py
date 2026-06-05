import random
from Color import Color
from Disk import Disk
from Utils import pointInPoly
from BoardPosition import BoardPosition
import copy
import json


class Board:
    def __init__(self):
        self.bbox = []
        self.points = []
        self.dices = []
        self.turn = 0
        self.player = Color.BLACK
        self.cube = 0

    def reset(self):
        self.clear()
        for point in self.points:
            point.reset()

    def clear(self):
        for point in self.points:
            point.clear()
        self.dices.clear()

    def addPoint(self, point):
        self.points.append(point)

    def addDice(self, dice):
        # check if the detected dice is on the board
        if pointInPoly(dice.center, self.bbox):
            self.dices.append(dice)

    def addDisk(self, disk):
        # self.disks.append(disk)
        # Determine correct point to add
        for point in self.points:
            if pointInPoly(disk.center, point.bbox_warped):
                # TODO: dtermine color clustering
                point.addDisk(disk)

    # def updateDicesBoardPosition(self):
    #     center_bar = self.getBar().center[0]
    #     for dice in self.dices:
    #         if dice.center[0] <= center_bar:
    #             dice.board_position = BoardPosition.LEFT
    #         else:
    #             dice.board_position = BoardPosition.RIGHT

    def getBar(self):
        return self.points[len(self.points) - 1]

    def __str__(self):
        res = ""
        res += "dices: "
        for dice in self.dices:
            res += str(dice) + " "
        res += "\n"
        for point in self.points:
            res += str(point) + "\n"
        return res

    def copy(self):
        board = Board()
        board.bbox = self.bbox.copy()
        for point in self.points:
            newPoint = point.copy()
            board.addPoint(newPoint)

        board.dices = self.dices.copy()
        return board
        # return copy.deepcopy(self)

    def calibratePoints(self):
        """
        Ensure all disks on each point have the same color.

        If a point contains mixed colors, choose the winning color using
        confidence-weighted voting:
          - Sum confidence for WHITE disks
          - Sum confidence for BLACK disks
        The larger total wins and all disks on that point are set to that color.
        """
        for point in self.points:
            if point.id == 25:  # Skip bar point
                continue
            
            if len(point.disks) <= 1:
                continue

            # If all disks are already equal in color, no calibration is needed.
            colors = [disk.color for disk in point.disks]
            if all(c == colors[0] for c in colors):
                continue

            white_conf = 0.0
            black_conf = 0.0
            best_white_conf = -1.0
            best_black_conf = -1.0

            for disk in point.disks:
                conf = float(disk.confidence)
                if disk.color == Color.WHITE:
                    white_conf += conf
                    if conf > best_white_conf:
                        best_white_conf = conf
                else:
                    black_conf += conf
                    if conf > best_black_conf:
                        best_black_conf = conf

            # Primary rule: confidence sum vote.
            # Tie-breaker: the side with the highest single-disk confidence wins.
            if white_conf > black_conf:
                target_color = Color.WHITE
            elif black_conf > white_conf:
                target_color = Color.BLACK
            else:
                target_color = Color.WHITE if best_white_conf >= best_black_conf else Color.BLACK

            for disk in point.disks:
                disk.color = target_color

    def exportJSON(self, total_checkers_per_player=15, as_string=False, indent=2):
        """
        Export board state using the requested schema.

        Example shape:
        {
          "turn": 25,
          "player": "white",
          "dice": [3, 5],
          "cube": 2,
          "points": {
            "1": {"white":0,"black":2},
            ...,
            "24": {"white":5,"black":0}
          },
          "bar": {"white":1,"black":0},
          "borne_off": {"white":3,"black":1}
        }
        """
        # Build lookup by point id for robust access to 1..24 and bar(25).
        points_by_id = {point.id: point for point in self.points}

        points_json = {}
        white_on_board = 0
        black_on_board = 0

        for point_id in range(1, 25):
            point = points_by_id.get(point_id)
            white_count = 0
            black_count = 0

            if point is not None:
                for disk in point.disks:
                    if disk.color == Color.WHITE:
                        white_count += 1
                    else:
                        black_count += 1

            points_json[str(point_id)] = {"white": white_count, "black": black_count}
            white_on_board += white_count
            black_on_board += black_count

        # Bar is point id 25 in this project.
        bar_point = points_by_id.get(25)
        bar_white = 0
        bar_black = 0
        if bar_point is not None:
            for disk in bar_point.disks:
                if disk.color == Color.WHITE:
                    bar_white += 1
                else:
                    bar_black += 1

        # Borne-off pieces are inferred from total pieces per player.
        borne_off_white = max(0, int(total_checkers_per_player) - white_on_board - bar_white)
        borne_off_black = max(0, int(total_checkers_per_player) - black_on_board - bar_black)

        state = {
            "turn": int(self.turn),
            "player": "white" if self.player == Color.WHITE else "black",
            "dice": [int(dice.value) for dice in self.dices],
            "cube": int(self.cube),
            "points": points_json,
            "bar": {
                "white": bar_white,
                "black": bar_black,
            },
            "borne_off": {
                "white": borne_off_white,
                "black": borne_off_black,
            },
        }

        if as_string:
            return json.dumps(state, indent=indent)
        return state

    def __eq__(self, other):
        return (isinstance(other, Board)
                and self.points == other.points 
                and self.dices == other.dices)

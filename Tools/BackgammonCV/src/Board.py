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

    def __eq__(self, other):
        return (isinstance(other, Board)
                and self.points == other.points 
                and self.dices == other.dices)

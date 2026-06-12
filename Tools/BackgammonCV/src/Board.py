import random
from Color import Color
from Cube import Cube
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
        self.cube = None
        self.player = None
        self.cube_owner = None

    def reset(self):
        self.clear()
        for point in self.points:
            point.reset()

    def clear(self):
        for point in self.points:
            point.clear()
        self.dices.clear()
        self.turn = 0
        self.cube = None
        self.player = None
        self.cube_owner = None

    def addPoint(self, point):
        self.points.append(point)

    def addDice(self, dice):
        # check if the detected dice is on the board
        if len(self.dices) >= 2:
            return
        
        if pointInPoly(dice.center, self.getBar().bbox_warped):
            return
        
        if pointInPoly(dice.center, self.bbox):
            self.dices.append(dice)

    def addDisk(self, disk):
        # self.disks.append(disk)
        # Determine correct point to add
        for point in self.points:
            if pointInPoly(disk.center, point.bbox_warped):
                # TODO: dtermine color clustering
                point.addDisk(disk)

    def setCube(self, cube):
        self.cube = cube

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
        res += "turn: " + str(self.turn) + "\n"
        res += "player: " + str(self.player) + "\n"
        res += "cube: " + str(self.cube) + "\n"
        res += "cube_owner: " + str(self.cube_owner) + "\n"
        for point in self.points:
            res += str(point) + "\n"
        return res
    
    def is_dice_changed(self, other):
        if len(self.dices) != len(other.dices):
            return True
        for dice in self.dices:
            if dice not in other.dices:
                return True
        return False

    def copy(self):
        board = Board()
        board.bbox = self.bbox.copy()
        for point in self.points:
            newPoint = point.copy()
            board.addPoint(newPoint)
        
        for dice in self.dices:
            newDice = dice.copy()
            board.addDice(newDice)
            
        board.turn = self.turn
        board.player = self.player
        board.cube = self.cube
        board.cube_owner = self.cube_owner
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

    def are_all_checkers_on_home_board(self, color):
        """
        Check if all checkers of the given color on the board are on their home board.

        White home board: points 1-6
        Black home board: points 19-24

        Args:
            color: Color.WHITE or Color.BLACK

        Returns:
            True if all checkers of this color on the board are in the home board.
            False if any checker is on the board but outside the home board or on the bar.
        """
        if color == Color.WHITE:
            home_board_range = range(1, 7)  # points 1-6
        elif color == Color.BLACK:
            home_board_range = range(19, 25)  # points 19-24
        else:
            return False

        points_by_id = {point.id: point for point in self.points}

        # Check all points outside the home board
        for point_id in range(1, 26):  # Check points 1-24 and bar (25)
            if point_id in home_board_range:
                continue  # Skip home board points
            
            point = points_by_id.get(point_id)
            if point is not None:
                for disk in point.disks:
                    if disk.color == color:
                        return False  # Found a checker outside home board

        # # Also check the bar (point 25)
        # bar_point = points_by_id.get(25)
        # if bar_point is not None:
        #     for disk in bar_point.disks:
        #         if disk.color == color:
        #             return False  # Found a checker on the bar

        return True
    
    def get_count_on_points(self, color):
        """
        Get the total count of checkers of the given color on the points (excluding borne-off).

        Args:
            color: Color.WHITE or Color.BLACK

        Returns:
            Total count of checkers of this color on the points.
        """
        count = 0
        for point in self.points:
            # if point.id == 25:  # Skip bar point
            #     continue
            
            for disk in point.disks:
                if disk.color == color:
                    count += 1
        return count

    def are_all_white_on_home_board(self):
        """Check if all white checkers are on their home board (points 1-6)."""
        return self.are_all_checkers_on_home_board(Color.WHITE)

    def are_all_black_on_home_board(self):
        """Check if all black checkers are on their home board (points 19-24)."""
        return self.are_all_checkers_on_home_board(Color.BLACK)

    def exportJSON(self, as_string=False):
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

        # Borne-off pieces: only count as borne-off if all checkers are on the home board.
        if self.are_all_white_on_home_board():
            borne_off_white = max(0, 15 - white_on_board - bar_white)
        else:
            borne_off_white = 0

        if self.are_all_black_on_home_board():
            borne_off_black = max(0, 15 - black_on_board - bar_black)
        else:
            borne_off_black = 0

        state = {
            "turn": int(self.turn),
            "player": "None" if self.player is None else "white" if self.player == Color.WHITE else "black",
            "dice": [int(dice.value) for dice in self.dices],
            "cube": int(self.cube.value) if self.cube is not None else 1,
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
            return json.dumps(state, indent=2)
        return state

    def __eq__(self, other):
        return (isinstance(other, Board)
                and self.points == other.points 
                and self.dices == other.dices
                and self.cube == other.cube)

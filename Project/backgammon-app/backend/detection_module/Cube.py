from Color import Color
from BoardPosition import BoardPosition
from Constants import CUBE_VALUE

class Cube:
    def __init__(
        self,
        id=0,
        center=(0, 0),
        confidence=0,
        owner=None,
    ):
        self.id = id
        self.value = CUBE_VALUE[id]
        self.confidence = confidence
        self.owner = owner
        self.center = center

    def __str__(self):
        res = str(self.value)
        return res

    def __repr__(self):
        return str(self.value)

    def __eq__(self, other):
        return (isinstance(other, Cube)
                and self.id == other.id 
                and self.owner == other.owner)
    
    def copy(self):
        cube = Cube(self.id, self.center, self.confidence, self.owner)
        return cube
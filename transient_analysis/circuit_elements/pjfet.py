from .njfet import NJFET


class PJFET(NJFET):
    """PJFET"""

    def __init__(self):
        super().__init__()
        self.JFET_type = -1  # 1 - n-type, -1 - p-type

    def __repr__(self):
        return 'p-JFET'

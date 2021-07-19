class GlobalConstant(dict):
    pass

Const = GlobalConstant(
    MAX_GB = 2.0,
    SHOW_PROGRESS = True,
    ID_AXIS = "p",
    FONT_SIZE_FACTOR = 1.0,
)

class SetConst:
    n_ongoing = 0
    def __init__(self, name, value):
        self.name = name
        self.value = value
    
    def __enter__(self):
        self.old_value = Const[self.name]
        Const[self.name] = self.value

    def __exit__(self, exc_type, exc_value, traceback):
        Const[self.name] = self.old_value
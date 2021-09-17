AI2THOR_TARGET_CLASSES = {
    19: [
        'Pillow', 'Television', 'GarbageCan', 'Box', 'RemoteControl',
        'Toaster', 'Microwave', 'Fridge', 'CoffeeMachine', 'Mug', 'Bowl',
        'DeskLamp', 'CellPhone', 'Book', 'AlarmClock',
        'Sink', 'ToiletPaper', 'SoapBottle', 'LightSwitch'
    ],
    22: [
        'AlarmClock', 'Book', 'Bowl', 'CellPhone', 'Chair', 'CoffeeMachine', 'DeskLamp', 'FloorLamp',
        'Fridge', 'GarbageCan', 'Kettle', 'Laptop', 'LightSwitch', 'Microwave', 'Pan', 'Plate', 'Pot',
        'RemoteControl', 'Sink', 'StoveBurner', 'Television', 'Toaster',
    ],
    60: [
        'AlarmClock', 'Bed', 'Blinds', 'Book', 'Bowl', 'Bread', 'ButterKnife', 'Cabinet', 'Candle', 'CD', 'CellPhone',
        'Chair', 'CoffeeMachine', 'WineBottle', 'DeskLamp', 'DishSponge', 'Drawer', 'Dresser', 'FloorLamp', 'Fridge',
        'GarbageCan', 'HandTowel', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LightSwitch', 'Microwave',
        'Mirror', 'Mug', 'Painting', 'Pan', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Pot', 'RemoteControl',
        'Safe', 'ScrubBrush', 'Sink', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'StoveBurner', 'TeddyBear',
        'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'ToiletPaper', 'Towel', 'Vase', 'Watch', 'WateringCan',
        'Window',
    ]
}

AI2THOR_TARGET_CLASSES_19_TYPES = [
    ['Toaster', 'Microwave', 'Fridge', 'CoffeeMachine', 'Mug', 'Bowl'],
    ['Pillow', 'Television', 'GarbageCan', 'Box', 'RemoteControl'],
    ['DeskLamp', 'CellPhone', 'Book', 'AlarmClock'],
    ['Sink', 'ToiletPaper', 'SoapBottle', 'LightSwitch'],
]

MOVE_AHEAD = 'MoveAhead'
ROTATE_LEFT = 'RotateLeft'
ROTATE_RIGHT = 'RotateRight'
LOOK_UP = 'LookUp'
LOOK_DOWN = 'LookDown'
DONE = 'Done'

# navigation rewards
DONE_ACTION_INT = 5
GOAL_SUCCESS_REWARD = 5
STEP_PENALTY = -0.01

# wandering rewards
DUPLICATE_STATE = -0.01
UNSEEN_STATE = 0.1

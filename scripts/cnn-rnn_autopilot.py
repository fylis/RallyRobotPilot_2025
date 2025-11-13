import collections
import torch
import numpy as np
from model import CNNGRU
from preprocess import preprocess_image

SEQ_LEN = 5
IMAGE_SIZE = (256,256)
CKPT_FILE = 'checkpoints/model_epoch_17.pt'


from PyQt6 import QtWidgets

from data_collector import DataCollectionUI

"""
This file is provided as an example of what a simplistic controller could be done.
It simply uses the DataCollectionUI interface zo receive sensing_messages and send controls.

/!\ Be warned that if the processing time of NNMsgProcessor.process_message is superior to the message reception period, a lag between the images processed and commands sent.
One might want to only process the last sensing_message received, etc. 
Be warned that this could also cause crash on the client side if socket sending buffer overflows

/!\ Do not work directly in this file (make a copy and rename it) to prevent future pull from erasing what you write here.
"""


class AutoPiloteNNMsgProcessor:
    def __init__(self):
        self.always_forward = True
        self.model = CNNGRU()
        states = torch.load(CKPT_FILE, map_location='cpu')
        self.model.load_state_dict(states)
        self.model.eval()  # IMPORTANT
        self.images = collections.deque(maxlen=SEQ_LEN)  # use deque for speed

    def nn_infer(self, message):
        commands = ["forward", "back", "left", "right"]
        img = preprocess_image(message.image)  # (H, W) float32
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        self.images.append(img)

        # Not enough frames?
        if len(self.images) < SEQ_LEN:
            print(f"Not yet sequence < {SEQ_LEN}")
            return None

        # Build sequence tensor
        seq = torch.stack(list(self.images), dim=0)  # (SEQ_LEN, 1, H, W)
        seq = seq.unsqueeze(0)  # (1, SEQ_LEN, 1, H, W)
        print(seq.shape)

        with torch.no_grad():
            logits = self.model(seq).squeeze(0)  # (4,)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)
        preds[0] = np.random.randint(0,2)
        print(preds)
        return list(zip(commands[:4], preds))

    def process_message(self, message, data_collector):
        commands = self.nn_infer(message)
        if commands is None:
            return
        for command, start in commands:
            data_collector.onCarControlled(command, bool(start))



if  __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = AutoPiloteNNMsgProcessor()
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()
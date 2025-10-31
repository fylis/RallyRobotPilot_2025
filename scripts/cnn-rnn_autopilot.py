import torch
import numpy as np
from model import RallyAutopilot
from preprocess import preprocess_image


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

CKPT_FILE = 'checkpoints/best_epoch_099.pt'

class AutoPiloteNNMsgProcessor:
    def __init__(self):
        self.always_forward = True
        self.model = RallyAutopilot()
        ckpt = torch.load(CKPT_FILE)
        self.model.load_state_dict(ckpt["model_state"])
        self.images = []
        self.frame_cnt = 0

    def nn_infer(self, message):
        #   Do smart NN inference here
        print("new frame ", message.image.shape)
        img = preprocess_image(message.image)
        print("processed image ", img.shape)
        self.images.append(img)

        if len(self.images) < 5:
            print("Not ready < 5")
            return [("forward", False)]
        if len(self.images) > 5:
            self.images.pop(0)
        
        features = np.stack(self.images[-5:])              # [5, 128, 128]
        features = features[:, np.newaxis, :, :]           # [5, 1, 128, 128]
        x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, 5, 1, 128, 128]
        with torch.no_grad():
            print("Start prediction ", x_tensor.shape)
            logits = self.model(x_tensor)[:4]
            print(logits)
            preds = (torch.sigmoid(logits) > 0.5).squeeze(0).numpy().astype(int)
        print(preds)
        commands = ["forward", "back", "left", "right"]
        return [(cmd, bool(pred)) for cmd, pred in zip(commands, preds)]



    def process_message(self, message, data_collector):
        if self.frame_cnt == 10:
            self.frame_cnt = 0
            commands = self.nn_infer(message)

            for command, start in commands:
                data_collector.onCarControlled(command, start)
        else:
            self.frame_cnt += 1

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
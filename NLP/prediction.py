from sklearn.feature_extraction.text import CountVectorizer
import joblib



NotNecesaryWord = [" and ", " are ", " or ", " the ", " is ", " this ", " of ", "  "]
punctuations = [",",".",";","!","?", "\"", "(", ")", "@", "'"]


def processText(txt):
    text = txt.lower()
    text = " " + text + " "
    for w in NotNecesaryWord: 
        text = text.replace(w, ' ')
    for p in punctuations:
        text = text.replace(p, '')
    return text

def predictFN(text):
    """
        take a text and tell if it is a fakenews or not.
        @param: string
        @return: BOOL
    """
    pipeline = joblib.load("pipeline.pkl")
    txt = processText(text)
    prediction = pipeline.predict([txt])
    return prediction[0]

def runPrediction(msg):
    prediction = predictFN(msg)
    if prediction == 1:
        return "This seems to be a real disaster"
    else:
        return "This is a false statement"


##### GUI #####

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QLineEdit

class MyGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prediction of fake desaster")
        self.resize(1000, 300)
        self.init_ui()

    def init_ui(self):
        self.text_zone = QTextEdit()
        self.text_zone.setReadOnly(True)  

        self.input_box = QLineEdit()
        self.input_box.setFixedHeight(50)

        self.button_add_text = QPushButton("PREDICT")
        self.button_add_text.clicked.connect(self.add_text_to_zone)

        layout = QVBoxLayout()
        layout.addWidget(self.input_box)
        layout.addWidget(self.button_add_text)
        layout.addWidget(self.text_zone)
        self.setLayout(layout)

    def add_text_to_zone(self):
        current_text = self.text_zone.toPlainText()
        inputTxt = self.input_box.text() 
        prediction = runPrediction(inputTxt)+ "\n"
        self.text_zone.setPlainText(inputTxt + "\n" + "Prediction: " + prediction)
        self.input_box.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MyGUI()
    gui.show()
    sys.exit(app.exec_())

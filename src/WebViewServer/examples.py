import sys
from PyQt4.QtCore import QObject, pyqtSlot
from PyQt4.QtGui import QApplication
from PyQt4.QtWebKit import QWebView

html = """
<html>
<body>
    <h1>Hello!</h1><br>
    <h2><a href="#" onclick="printer.text(printer.India)">QObject Test</a></h2>
    <h2><a href="#" onclick="alert('Javascript works!')">JS test</a></h2>
</body>
</html>
"""

class ConsolePrinter(QObject):
    def __init__(self, parent=None):
        super(ConsolePrinter, self).__init__(parent)
        self.India = "Love India"

    @pyqtSlot(str)
    def text(self, message):
        print "Message being printed",message

if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = QWebView()
    frame = view.page().mainFrame()
    printer = ConsolePrinter()
    view.setHtml(html)
    frame.addToJavaScriptWindowObject('printer', printer)
    frame.evaluateJavaScript("alert('Hello');")
    frame.evaluateJavaScript("printer.India;")
    view.show()
    app.exec_()
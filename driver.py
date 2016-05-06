from WebViewServer.CustomWebView import CustomWebView
from sys import exit
from PySide.QtGui import QApplication
from PySide.QtWebKit import QWebView, QWebPage

app = QApplication([])
view = CustomWebView()
view.show()

exit(app.exec_())




import os
import sys

from PySide import QtCore, QtGui, QtWebKit
from PySide.QtCore import QObject, Property
# from PySide.QtCore import *
# from PySide.QtGui import *
# from PySide.QtWebKit import *
from PySide import QtWebKit

execfile('index.py')

def readFile():
    #Should be swapped with a read of a template
    class JavaScriptObjectToSend(QtCore.QObject):  
        """Simple class with one slot and one read-only property."""  
     
        @QtCore.Slot(str)  
        def showMessage(self, msg):  
            """Open a message box and display the specified message."""  
            QtGui.QMessageBox.information(None, "Info", msg)  
      
        def _pyVersion(self):  
            """Return the Python version."""  
            return sys.version  
      
        """Python interpreter version property."""  
        pyVersion = Property(str, fget=_pyVersion) 

    class jsonObject(QObject):
        def __init__(self,startval=42):
            QObject.__init__(self)
            self.ppval="source"
            self.pp = Property(str,self.readPP,self.setPP)

        @QtCore.Slot(str)  
        def readPP(self,msg):
            return msg,self.ppval
        
        @QtCore.Slot(str)  
        def setPP(self,val):
            self.ppval = val
     

    obj = jsonObject()

    basepath = os.path.dirname(os.path.abspath(__file__))
    basepath = str(basepath)+'/'

    win = QtWebKit.QWebView()

    win.setWindowTitle('D3d visualization')
    layout = QtGui.QVBoxLayout()
    win.setLayout(layout)
    myObj = JavaScriptObjectToSend()
    
    view = QtWebKit.QWebView()
    view.settings().setAttribute(QtWebKit.QWebSettings.LocalContentCanAccessRemoteUrls, True)
    
    view.page().mainFrame().addToJavaScriptWindowObject("pyObj", myObj)
    view.page().mainFrame().addToJavaScriptWindowObject("jsonObj", obj)  

    view.settings().setAttribute(QtWebKit.QWebSettings.PluginsEnabled, True)
    view.settings().setAttribute(QtWebKit.QWebSettings.WebAttribute.DeveloperExtrasEnabled, True)
    view.settings().setAttribute(QtWebKit.QWebSettings.PrivateBrowsingEnabled, True)

    view.setHtml(html, baseUrl=QtCore.QUrl().fromLocalFile(basepath))
    return view

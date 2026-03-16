from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
from views.mainWidget import mainWidget


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = mainWidget()
    main.show()
    sys.exit(app.exec_())
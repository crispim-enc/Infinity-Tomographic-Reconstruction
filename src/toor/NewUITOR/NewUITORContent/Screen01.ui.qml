

/*
This is a UI file (.ui.qml) that is intended to be edited in Qt Design Studio only.
It is supposed to be strictly declarative and only uses a subset of QML. If you edit
this file manually, you might introduce QML code that is not supported by Qt Design Studio.
Check out https://doc.qt.io/qtcreator/creator-quick-ui-forms.html for details on .ui.qml files.
*/
import QtQuick
import QtQuick.Controls
import QtQuick.Shapes
import NewUITOR

Rectangle {
    id: mainRectangle
    width: Constants.width
    height: Constants.height
    gradient: RadialGradient {
        orientation: Gradient.Vertical
        GradientStop {
            position: 0
            color: "#011b31"
        }
        GradientStop {
            position: 1.0
            color: "#0a376e"
        }
    }
    Rectangle {
        id: pinksidebar
        width: 6
        height: parent.height
        anchors.left: parent.left
        // anchors.leftMargin: sideBar.width
        gradient: RadialGradient {
            GradientStop {
                position: 0
                color: "#E00048"
            }
            GradientStop {
                position: 0.8
                color: "transparent"
            }
        }
    }

    Rectangle {
        id: sideBar
        // implicitwidth: 250
        implicitWidth: 250
        height: parent.height
        gradient: RadialGradient {
            GradientStop {
                position: 0
                color: "#65e00048"
            }
            GradientStop {
                position: 0.1
                color: "transparent"
            }
        }

        Column {
            id: sidebarlayout
            width: parent.width
            height: parent.height
            anchors.top: parent.top
            spacing: 4
            Image {
                id: tOR_logo
                x: 8
                y: 8
                width: 210
                height: 100
                source: "images/TOR_logo.png"
                fillMode: Image.PreserveAspectFit
            }
            ToolButton {
                id: systemConfigurationsMenuButton
                width: parent.width
                height: 60
                text: qsTr("System \n Configurations")
                highlighted: false
                flat: true
                font.family: "Gill Sans MT"
                font.weight: Font.Bold
                font.pointSize: 12
                checkable: true
                background: Rectangle {
                    radius: 2
                    color: "white"
                    border.color: "green"
                    border.width: 3
                }
            }
        }
    }

    Rectangle {
        id: statusBar
        x: 242
        y: 0
        width: 1678
        height: 46
        opacity: 1
    }

    Item {
        id: name
        anchors.left: sideBar.right
        PageSetGeometryForm {}
    }
}

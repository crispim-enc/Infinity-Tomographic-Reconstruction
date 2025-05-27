

/*
This is a UI file (.ui.qml) that is intended to be edited in Qt Design Studio only.
It is supposed to be strictly declarative and only uses a subset of QML. If you edit
this file manually, you might introduce QML code that is not supported by Qt Design Studio.
Check out https://doc.qt.io/qtcreator/creator-quick-ui-forms.html for details on .ui.qml files.
*/
import QtQuick
import QtQuick.Controls

import QtQuick.Shapes

// Item {
//     width: 1920
//     height: 1080
//     opacity: 1
import QtQuick.Studio.DesignEffects
import QtQuick.Layouts

Rectangle {
    id: system_configurations_rect
    width: 1720
    height: 1080
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
    Grid {
        id: grid_system
        x: 0
        y: 0
        width: parent.width
        height: parent.height
        anchors.left: parent.left
        anchors.top: parent.top
        spacing: 5

        Rectangle {
            id: scannertype
            x: 20
            y: 20
            width: parent.width / 4
            height: parent.height / 5
            opacity: 1
            // color: "#140080ff"
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
            radius: 10
            border.color: "#23c7ba"
            border.width: 2
            Text {
                id: scannertype_str
                x: 8
                y: 8
                width: parent.width
                height: parent.height / 5
                color: "#e3e4e4"
                text: qsTr("SCANNER TYPE")
                font.pixelSize: 20
                font.bold: true
            }

            Grid {
                readonly property font toogleButtonFont: ({
                                                              "family": "Helvetica",
                                                              "pointSize": 14,
                                                              "bold": true
                                                          })

                id: grid
                anchors.top: scannertype_str.bottom
                anchors.topMargin: 15
                anchors.left: parent.left
                anchors.leftMargin: 4
                width: parent.width
                height: parent.height
                verticalItemAlignment: Grid.AlignVCenter
                horizontalItemAlignment: Grid.AlignHCenter
                rows: 1
                columns: 4
                spacing: 4
                ToolButton {
                    id: petbutton
                    width: parent.width / 4 - 5
                    height: parent.height - scannertype_str.height - 40

                    layer.enabled: false
                    layer.format: ShaderEffectSource.RGBA
                    icon.color: "#5f1b4f83"
                    autoExclusive: true
                    checked: true
                    checkable: true
                    highlighted: false
                    text: qsTr("PET")
                    flat: false
                    font: parent.toogleButtonFont
                }

                ToolButton {
                    id: spectbutton

                    width: parent.width / 4 - 5
                    height: parent.height - scannertype_str.height - 40
                    layer.enabled: false
                    layer.format: ShaderEffectSource.RGBA
                    icon.color: "#1b4f83"
                    autoExclusive: true
                    checked: false
                    checkable: true
                    highlighted: false
                    text: qsTr("SPECT")
                    font: parent.toogleButtonFont
                }

                ToolButton {
                    id: comptonbutton
                    width: parent.width / 4 - 5
                    height: parent.height - scannertype_str.height - 40
                    layer.enabled: false
                    layer.format: ShaderEffectSource.RGBA
                    icon.color: "#1b4f83"
                    autoExclusive: true
                    checked: false
                    checkable: true
                    highlighted: false
                    text: qsTr("Compton")
                    font: parent.toogleButtonFont
                }

                ToolButton {
                    id: ctbutton
                    width: parent.width / 4 - 5
                    height: parent.height - scannertype_str.height - 40

                    icon.color: "#1b4f83"
                    autoExclusive: true
                    checked: false
                    checkable: true
                    highlighted: false
                    text: qsTr("CT")
                    font: parent.toogleButtonFont
                }
            }
        }

        Rectangle {
            id: system_id_rectangle
            width: scannertype.width
            implicitHeight: 200
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
            radius: 10
            border.color: "#23c7ba"
            border.width: 2
            anchors.top: scannertype.bottom
            anchors.topMargin: 10
            anchors.left: scannertype.left
        }

        Rectangle {
            id: geometry_rectangle

            implicitWidth: scannertype.width
            implicitHeight: parent.height / 2
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
            radius: 10
            border.color: "#23c7ba"
            border.width: 2
            anchors.top: system_id_rectangle.bottom
            anchors.topMargin: 10
            anchors.left: scannertype.left

            GridLayout {
                id: gridLayout
                width: parent.width

                anchors.left: parent.left
                anchors.top: parent.top
                anchors.leftMargin: 5
                anchors.topMargin: 0
                Text {
                    id: geometry_type_str
                    width: parent.width
                    height: 33
                    color: "#e3e4e4"
                    text: qsTr("GEOMETRY TYPE")
                    font.pixelSize: 20
                    font.bold: true
                }
            }
        }

        Rectangle {
            id: geometry_cells
            x: 471
            y: 25
            width: 1218
            height: 1022
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
            radius: 10
            border.color: "#23c7ba"
            border.width: 3
        }
    }
}

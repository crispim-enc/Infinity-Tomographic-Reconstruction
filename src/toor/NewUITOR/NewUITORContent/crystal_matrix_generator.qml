import QtQuick 2.7
import QtQuick.Controls 2.2
import QtQuick.Layouts 1.1
import QtQuick.Window 2.2





Rectangle{
        id: fillRectangle
        objectName: 'fillRectangle'
        color: "#00000000"
        width: 640
        height: 480



    Text{
        y: 55
        id: sideA_text
        text: 'SIDE A'
        color: 'steelblue'
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        font.pointSize: 13
        font.bold: true
        font.family: "Verdana"
        anchors.left: crystalgridA.left
        anchors.leftMargin: crystalgridA.width/2-y/2
        anchors.bottom: crystalgridA.top
        anchors.bottomMargin: 15
    }

    Text{
        y: 55
        id: sideB_text
        text: 'SIDE B'
        color: '#004d4d'
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignHCenter
        font.pointSize: 13
        font.bold: true
        font.family: "Verdana"
        anchors.left: crystalgridB.left
        anchors.leftMargin: crystalgridB.width/2-y/2
        anchors.bottom: crystalgridB.top
        anchors.bottomMargin: 15
    }

    Grid {
            id: crystalgridA
            objectName: "crystalgridA"
            anchors.top: parent.top
            anchors.topMargin: 110
            anchors.left: parent.left
            anchors.leftMargin: 30
            width: 320
            height: 400
            x: 0

            Repeater{
                model: model_crystals_array_A
                id: repeater_crystals_arrayA
                objectName:"REPEATERA"
                delegate: Rectangle{

                    height:model.height
                    width:model.width

                    color: model.color

                    Text{
                     anchors.centerIn: parent
                     text: model.text
                     color: "#004d4d"

                    }
                }
            }
    }

    Slider {

        id: sliderHorizontal_motor_axis_x

        padding: 0

        objectName: "sliderHorizontal_motor_axis_x"
        anchors.top: crystalgridA.bottom
        anchors.left: crystalgridA.left
        anchors.leftMargin: 0
        anchors.topMargin: 0
        height: handle_image.height
        width: repeater_crystals_arrayA.width

        handle: BorderImage {
            id: handle_image
            source: "Resources/motors_icon.png"
            width: 40;
            height: 40;

            border.left: 0; border.top: 0
            border.right: 0; border.bottom: 0
            /*
            implicitWidth: 26
            implicitHeight: 26
            radius: 13
            border.width: 6*/
            x: sliderHorizontal_motor_axis_x.leftPadding + sliderHorizontal_motor_axis_x.visualPosition * (sliderHorizontal_motor_axis_x.availableWidth)-width/2
            y: sliderHorizontal_motor_axis_x.topPadding + sliderHorizontal_motor_axis_x.availableHeight / 2 - height / 2
        }
        onValueChanged: buttonText.text = '%1 mm'.arg(Math.round(value * 100) / 100)

        TextInput{
            id: buttonText
            objectName: 'buttonText'
            text: Math.round(parent.value * 100) / 100

            font.family: "Verdana"

            anchors.top: parent.bottom
            anchors.topMargin: 0
            anchors.left: parent.left
            anchors.leftMargin: parent.width/2-buttonText.width/2

            selectByMouse: true

            validator: DoubleValidator {
                id: validator_slider_text
                locale: "Portuguese" /*verificar se isto nao vai dar problemas em teclados estrangeiros*/
                bottom: sliderHorizontal_motor_axis_x.from
                top:  sliderHorizontal_motor_axis_x.to
                decimals: 2

            }

            onEditingFinished: {
                sliderHorizontal_motor_axis_x.value = parseFloat(buttonText.text)
                buttonText.text = '%1 mm'.arg(Math.round(parent.value * 100) / 100)

            }
            onSelectedTextChanged: buttonText.clear()


        }

    }

    Slider {

        id: sliderHorizontal_motor_axis_y
        value: 0.5
        padding: 0

        objectName: "sliderHorizontal_motor_axis_y"
        anchors.top: crystalgridA.bottom
        anchors.left: sliderHorizontal_motor_axis_x.right
        anchors.leftMargin: 0
        anchors.topMargin: 0
        width: fillRectangle.width-2*sliderHorizontal_motor_axis_x.width-2*crystalgridA.x;
        height: handle_image_slider_y.height

        handle: BorderImage {
            id: handle_image_slider_y
            source: "Resources/motors_icon.png"
            width: 40;
            height: 40;

            border.left: 0; border.top: 0
            border.right: 0; border.bottom: 0
            /*
            implicitWidth: 26
            implicitHeight: 26
            radius: 13
            border.width: 6*/
            x: sliderHorizontal_motor_axis_y.leftPadding + sliderHorizontal_motor_axis_y.visualPosition * (sliderHorizontal_motor_axis_y.availableWidth)-width/2
            y: sliderHorizontal_motor_axis_y.topPadding + sliderHorizontal_motor_axis_y.availableHeight / 2 - height / 2
        }
        Text {
            id: distancesideA
            objectName: "distancesideA"
            text: "%1 cm".arg(spinBox_distance_between_crystals.value*Math.round(parent.value * 100) / 1000)
            x:sliderHorizontal_motor_axis_y.leftPadding + (sliderHorizontal_motor_axis_y.visualPosition -sliderHorizontal_motor_axis_y.value/2)* (sliderHorizontal_motor_axis_y.availableWidth)
            y: parent.height
        }
        Text {
            id: distancesideB
            objectName: "distancesideB"
            text: "%1 cm".arg(spinBox_distance_between_crystals.value*Math.round((1-parent.value) * 100) / 1000)
            x:sliderHorizontal_motor_axis_y.leftPadding + (sliderHorizontal_motor_axis_y.visualPosition-sliderHorizontal_motor_axis_y.value/2)* (sliderHorizontal_motor_axis_y.availableWidth)+sliderHorizontal_motor_axis_y.width/2
            y: parent.height
        }
        Text {
            id: topmotor_identifier_slider
            text: qsTr("TOP MOTOR")
            font.bold: true
            font.family: "Verdana"
            y: parent.height/2 - topmotor_identifier_slider.height/2
            x: parent.width + 5
        }

    }


    Grid {
            id: crystalgridB
            objectName: "crystalgridB"
            anchors.top: parent.top
            anchors.topMargin: 110

            anchors.right: parent.right
            anchors.rightMargin: 30
            width: 320
            height: 400
            x: 0

            Repeater{
                model: model_crystals_array_B
                id: repeater_crystals_array_B
                objectName:"REPEATER_B"
                delegate: Rectangle{

                    height:model.height
                    width:model.width

                    color: model.color

                    Text{
                     anchors.centerIn: parent
                     text: model.text
                     color: "#004d4d"

                    }

                }


            }


    }
    Rectangle{
        id: rectangle_distance_between_crystals
        width: fillRectangle.width-2*sliderHorizontal_motor_axis_x.width-2*crystalgridA.x-2*sideA_arrow.width;
        height: 6;


        anchors.bottom: sliderHorizontal_motor_axis_y.top
        anchors.bottomMargin: 40
        x: sliderHorizontal_motor_axis_y.x+sideA_arrow.width/2+2

        color: '#292a2b'

        Rectangle{
            id: sideA_arrow
            x: -1
            y:-rectangle_distance_between_crystals.height/2
            width: rectangle_distance_between_crystals.height*2
            height: rectangle_distance_between_crystals.height*2
            radius: rectangle_distance_between_crystals.height*2
            color: '#292a2b'
        }
        Rectangle{
            id: sideB_arrow
            x: rectangle_distance_between_crystals.width-1
            y:-rectangle_distance_between_crystals.height/2
            width: rectangle_distance_between_crystals.height*2
            height: rectangle_distance_between_crystals.height*2
            radius: rectangle_distance_between_crystals.height*2
            color: '#292a2b'
        }


        SpinBox {
            id: spinBox_distance_between_crystals
            objectName: 'spinBox_distance_between_crystals'
            anchors.left: sliderHorizontal_motor_axis_x.right
            x: rectangle_distance_between_crystals.width/2-spinBox_distance_between_crystals.width/2 +sideA_arrow.width/2
            y: -spinBox_distance_between_crystals.height/2


            from: 1
            value: 60
            to: 1000* 10
            stepSize: 1
            editable: false
            property int decimals: 1
            property real realValue: value / 10

            validator: DoubleValidator {
                bottom: Math.min(spinBox_distance_between_crystals.from, spinBox_distance_between_crystals.to)
                top:  Math.max(spinBox_distance_between_crystals.from, spinBox_distance_between_crystals.to)
            }

            textFromValue: function(value, locale) {
                return "%1 cm".arg(Number(value/10).toLocaleString(locale, 'f', spinBox_distance_between_crystals.decimals))
            }

            valueFromText: function(text, locale) {
                if (text === "none") return Number.fromLocaleString(locale, text)*10; else return  text[0]*10;
                       }

        }

    }


    Rectangle {

        id: rectanglecrystal2
        x: fillRectangle.width/2-rectanglecrystal2.width/2
        y: 100
        gradient: Gradient {
                GradientStop { position: 0.0; color: "#ccffff" }
                GradientStop { position: 1.0; color: "#f2f2f2" }
            }
        border.color: "#8c8c8c"
        border.width: 2
        width: 40; height: 40





        Rectangle {
            id: rectanglecrystal3

            x: -rectanglecrystal2.width/2-rectanglecrystal3.width/2
            gradient: Gradient {
                    GradientStop { position: 0.0; color: "#e6e6e8" }
                    GradientStop { position: 1.0; color: "#afafb6" }
                }
            border.color: "#8c8c8c"
            border.width: 2
            width: rectanglecrystal2.width; height: rectanglecrystal2.height
            transform: Matrix4x4 {
                      property real a: Math.PI / 4
                      matrix: Qt.matrix4x4(1, 0, 0, 0,
                                           Math.tan(a),  1, 0, -rectanglecrystal2.height,
                                           0,           0,            1, 0,
                                           0,           0,            0, 1)
            }


        }
        Rectangle {
            id: rectanglecrystal4
            y: -rectanglecrystal2.height/2-rectanglecrystal3.height/2

            gradient: Gradient {
                    GradientStop { position: 0.0; color: "#e6e6e8" }
                    GradientStop { position: 1.0; color: "#afafb6" }
                }
            border.color: "#8c8c8c"
            border.width: 2
            width: rectanglecrystal2.width; height: rectanglecrystal2.height

            transform: Matrix4x4 {
                      property real a: Math.PI / 4
                      matrix: Qt.matrix4x4(1, Math.tan(a), 0, -rectanglecrystal2.width,
                                           0,  1, 0, 0,
                                           0,           0,            1, 0,
                                           0,           0,            0, 1)
            }
        }
    }





    TextInput{
        id: pitch_X_crystal
        objectName: 'pitch_X_crystal'
        text: '2 mm'
        font.bold: true
        font.pointSize: 11
        color: '#292a2b'
        selectByMouse: true
        x:rectanglecrystal2.x-pitch_X_crystal.width
        y:rectanglecrystal2.y-3*rectanglecrystal2.height/2

        validator: DoubleValidator {
            locale: "Portuguese" /*verificar se isto nao vai dar problemas em teclados estrangeiros*/
            bottom: 0.01
            top:  10
            decimals: 3

        }

        signal update_size_crystal()

        onEditingFinished: {
            pitch_X_crystal.update_size_crystal()
            pitch_X_crystal.text = '%3 mm'.arg(pitch_X_crystal.text)
        }
        onSelectedTextChanged: pitch_X_crystal.clear()

    }

    TextInput{
        id: pitch_Y_crystal
        objectName: 'pitch_Y_crystal'
        text: '1,5 mm'


        selectByMouse: true
        font.bold: true
        font.pointSize: 11
        color: '#292a2b'
        x:rectanglecrystal2.x - rectanglecrystal2.width -pitch_Y_crystal.width
        y:rectanglecrystal2.y - 3*pitch_Y_crystal.height/2

        validator: DoubleValidator {
            locale: "Portuguese" /*verificar se isto nao vai dar problemas em teclados estrangeiros*/
            bottom: 0.01
            top:  10
            decimals: 3

        }
        signal update_size_crystal()

        onEditingFinished: {
            pitch_Y_crystal.update_size_crystal()



            pitch_Y_crystal.text = '%3 mm'.arg(pitch_Y_crystal.text)


        }
        onSelectedTextChanged: pitch_Y_crystal.clear()


    }

    TextInput{
        id: lenght_crystal
        objectName: 'lenght_crystal'
        text: '20 mm'
        font.bold: true
        font.pointSize: 11
        selectByMouse: true
        color: '#292a2b'
        x:rectanglecrystal2.x+rectanglecrystal3.x-lenght_crystal.width/2 -4
        y:rectanglecrystal2.y+rectanglecrystal3.y/2+rectanglecrystal3.width/4
        validator: DoubleValidator {
            locale: "Portuguese" /*verificar se isto nao vai dar problemas em teclados estrangeiros*/
            bottom: 0.01
            top:  100
            decimals: 3

        }
        signal update_size_crystal()
        onEditingFinished: {
            lenght_crystal.update_size_crystal()
            lenght_crystal.text = '%3 mm'.arg(lenght_crystal.text)
        }
        onSelectedTextChanged: lenght_crystal.clear()

    }

    TextInput{
        id: reflector_crystal_exterior
        objectName: 'reflector_crystal_exterior'
        text: '0.1 mm'
        font.bold: true
        font.pointSize: 11
        selectByMouse: true
        color: '#292a2b'
        x:rectanglecrystal2.x - rectanglecrystal2.width/4
        y:rectanglecrystal2.y - pitch_Y_crystal.height

        validator: DoubleValidator {
            locale: "Portuguese" /*verificar se isto nao vai dar problemas em teclados estrangeiros*/
            bottom: 0.01
            top:  10
            decimals: 3

        }
        signal update_size_crystal()
        onEditingFinished: {
            reflector_crystal_exterior.update_size_crystal()
            reflector_crystal_exterior.text = '%3 mm'.arg(reflector_crystal_exterior.text)
        }
        onSelectedTextChanged: reflector_crystal_exterior.clear()
    }

    TextInput{
        id: reflector_crystal_interior_A_X
        objectName: 'reflector_crystal_interior_A_X'
        text: '0.1 mm'
        font.bold: true
        font.pointSize: 11
        selectByMouse: true
        color: sideA_text.color
        x:rectanglecrystal2.x
        y:rectanglecrystal2.y + pitch_Y_crystal.height+ reflector_crystal_interior_A_X.height + 4

        validator: DoubleValidator {
            locale: "Portuguese" /*verificar se isto nao vai dar problemas em teclados estrangeiros*/
            bottom: 0.01
            top:  10
            decimals: 3

        }
        signal update_size_crystal()

        onEditingFinished: {
            reflector_crystal_interior_A_X.update_size_crystal
            reflector_crystal_interior_A_X.text = '%3 mm'.arg(reflector_crystal_interior_A_X.text)
        }
        onSelectedTextChanged: reflector_crystal_interior_A_X.clear()
    }

    TextInput{
        id: reflector_crystal_interior_A_Y
        objectName: 'reflector_crystal_interior_A_Y'
        text: '0.1 mm'
        font.bold: true
        font.pointSize: 11
        selectByMouse: true
        color: sideA_text.color
        x:rectanglecrystal2.x + rectanglecrystal2.width
        y:rectanglecrystal2.y

        validator: DoubleValidator {
            locale: "Portuguese" /*verificar se isto nao vai dar problemas em teclados estrangeiros*/
            bottom: 0.01
            top:  10
            decimals: 3

        }
        signal update_size_crystal()
        onEditingFinished: {
            reflector_crystal_interior_A_Y.update_size_crystal()
            reflector_crystal_interior_A_Y.text = '%3 mm'.arg(reflector_crystal_interior_A_Y.text)
        }
        onSelectedTextChanged: reflector_crystal_interior_A_Y.clear()
    }

    TextInput{
        id: reflector_crystal_interior_B_X
        objectName: 'reflector_crystal_interior_B_X'
        text: '0.1 mm'
        font.bold: true
        font.pointSize: 11
        selectByMouse: true
        color: sideB_text.color
        x:rectanglecrystal2.x
        y:rectanglecrystal2.y + pitch_Y_crystal.height+ reflector_crystal_interior_B_X.height +reflector_crystal_interior_A_X.height+ 4

        validator: DoubleValidator {
            locale: "Portuguese" /*verificar se isto nao vai dar problemas em teclados estrangeiros*/
            bottom: 0.01
            top:  10
            decimals: 3

        }
        signal update_size_crystal()

        onEditingFinished: {
            reflector_crystal_interior_B_X.update_size_crystal()
            reflector_crystal_interior_B_X.text = '%3 mm'.arg(reflector_crystal_interior_B_X.text)
        }
        onSelectedTextChanged: reflector_crystal_interior_B_X.clear()
    }

    TextInput{
        id: reflector_crystal_interior_B_Y
        objectName: 'reflector_crystal_interior_B_Y'
        text: '2 mm'
        font.bold: true
        font.pointSize: 11
        selectByMouse: true
        color: sideB_text.color
        x:rectanglecrystal2.x + rectanglecrystal2.width
        y:rectanglecrystal2.y + pitch_Y_crystal.height

        validator: DoubleValidator {
            locale: "Portuguese" /*verificar se isto nao vai dar problemas em teclados estrangeiros*/
            bottom: 0.01
            top:  10
            decimals: 3

        }
        signal update_size_crystal()

        onEditingFinished: {
            reflector_crystal_interior_B_Y.update_size_crystal()
            reflector_crystal_interior_B_Y.text = '%3 mm'.arg(reflector_crystal_interior_B_Y.text)
        }
        onSelectedTextChanged: reflector_crystal_interior_B_Y.clear()
    }

    BorderImage {
        id: bot_motor
        source: "Resources/motors_icon.png"
        anchors.top : sliderHorizontal_motor_axis_y.bottom
        anchors.topMargin: 20
        anchors.left: parent.left
        anchors.leftMargin: fillRectangle.width/4-bot_motor.width/2




        width: 70; height: 70
        Text {
            id: text_botmotor
            text: qsTr("BOT MOTOR")
            font.bold: true
            font.family: "Verdana"
            y: bot_motor.height
            x: bot_motor.width/2-text_botmotor.width/2
        }

        Rectangle{
            id: rectangle_distance_between_motors
            height: 6;
            width: top_motor.x-bot_motor.width+30
            y: bot_motor.height/2
            x: bot_motor.width - 18

            color: '#292a2b'
        }

        SpinBox {
            id: spinBox_distance_between_motors
            objectName: 'spinBox_distance_between_motors'
            anchors.left: sliderHorizontal_motor_axis_x.right
            x: rectangle_distance_between_motors.width/2-spinBox_distance_between_motors.width/2+rectangle_distance_between_motors.x
            y:spinBox_distance_between_motors.height/2

            width: top_motor.x-bot_motor.x
            from: 1
            value: 90
            to: 1000 * 10
            stepSize: 1
            editable: false

            property int decimals: 1
            property real realValue: value / 10

            validator: DoubleValidator {
                locale: "Portuguese"
                bottom: Math.min(spinBox_distance_between_motors.from, spinBox_distance_between_motors.to)
                top:  Math.max(spinBox_distance_between_motors.from, spinBox_distance_between_motors.to)
            }

            textFromValue: function(value, locale) {
                return "%1 cm".arg(Number(value/10).toLocaleString(locale, 'f', spinBox_distance_between_crystals.decimals))
            }

            valueFromText: function(text, locale) {
                if (text === "none") return Number.fromLocaleString(locale, text)*10; else return  text[0]*10;
                       }




        }
        BorderImage {
            id: top_motor
            source: "Resources/motors_icon.png"
            width: 50; height: 50

            x : fillRectangle.width/2
            y: bot_motor.width/2-top_motor.width/2

            Text {
                id: text_topmotor
                text: qsTr("TOP MOTOR")
                font.bold: true
                font.family: "Verdana"
                y: top_motor.height
                x: top_motor.width/2-text_topmotor.width/2
            }

        }

    }

 }

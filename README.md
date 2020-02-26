## Project Name

KEA

## How to run the project

The simplest command on the terminal is `python stream_drone.py`. Additional options can be added to change the model name, the input stream.

Before hitting the command, follow the steps below:
* Power up your drone(I am using parrot anafi)
* Connect your skycontroller3 to your computer. It serves as a bridge between your code and the drone

## Various Options

* `-i` To change the input stream. By default, it is an rstp stream from the drone
* `-m` To change the model. By default, it is a ssd mobilenet v2 trained on coco dataset
* `-c` To add cpu extension if needed. By default cpu extension is not needed

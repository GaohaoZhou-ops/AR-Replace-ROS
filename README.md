AR Replace ROS

This repository uses AR to implement the function of replacing template images.

The non-ROS implementation with the same functionality as this repository is linked to:

* https://github.com/GaohaoZhou-ops/AR-Replace


----
# How to Use

## Step1. Pull code

Assume your ros workspace name is `ar_ws`:

```bash
$ cd ar_ws/src
$ git clone https://github.com/GaohaoZhou-ops/AR-Replace.git
```

## Step2. Build 

```bash
$ cd ar_ws
$ catkin_make
$ source devel/setup.bash
```

## Step3. Run

You can subscribe to different topics by modifying the topic name in the source script.


Use the following launch file to detect only one object:
```bash
$ roslaunch ar_replace_pkg ar_single.launch
```

Use the following launch file to replace multiple objects
```bash
$ roslaunch ar_replace_pkg ar_multi.launch
```

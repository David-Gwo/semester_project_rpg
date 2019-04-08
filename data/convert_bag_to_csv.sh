#!/bin/sh

usage()
{
    echo "usage: convert_bag_to_csv <bag file> <topic name 1> <topic_name_2> ...   | -h for help (this message)"
}

. /opt/ros/kinetic/setup.sh
export ROS_ROOT=/opt/ros/kinetic/ros
export PATH=$ROS_ROOT/bin:$PATH
export PYTHONPATH=$ROS_ROOT/core/roslib/src:$PYTHONPATH
export ROS_PACKAGE_PATH=catkin_ws:/opt/ros/kinetic/stacks

while [ "$1" != "" ]; do
    case $1 in
        -h | --help )           usage
                                exit
                                ;;
        * )                     . catkin_ws/devel/setup.sh & python catkin_ws/src/bag2csv/main.py "$@"
                                exit 1
    esac
done



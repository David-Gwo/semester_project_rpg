"""
This script saves each topic in a bagfile as a csv.

Accepts a filename as an optional argument. Operates on all bagfiles in current directory if no argument provided

Usage 1 (for one bag file):
    python bag2csv.py bag_directory/filename.bag topic_1 topic_2 ... topic_n
Usage 2 (for all bag files in current directory):
    python bag2csv.py bag_directory/filename.bag

Script adapted by Guillem Torrente from original script by Nick Speal in May 2013 at McGill University's Aerospace
Mechatronics Laboratory. Bugfixed by Marc Hanheide. June 2016.
www.speal.ca

Supervised by Professor Inna Sharf, Professor Meyer Nahon
"""

import rosbag
import sys
import csv
import string
import os  # for file management make directory

# verify correct input arguments: 1 or 2
if len(sys.argv) < 2:
    print("invalid number of arguments:   " + str(len(sys.argv)))
    print("should be 3 or more: 'bag2csv.py' 'bagDir/Name' 'topicName_1' 'topicName_2' ...")
    print("or just 2  : 'bag2csv.py' 'bagDir/Name")
    sys.exit(1)
else:
    listOfBagFiles = [sys.argv[1]]
    numberOfFiles = 1
    print("reading bag file: " + str(listOfBagFiles[0]))

    topic_names = None
    if len(sys.argv) > 2:
        topic_names = sys.argv[2:]
        topic_name_list = ""
        for topic_name in topic_names:
            topic_name = string.lower(topic_name)
            topic_name_list += topic_name + ' '
        print("reading topics: " + topic_name_list)

for bagFile in listOfBagFiles:
    # access bag
    bag = rosbag.Bag(bagFile)
    bagContents = bag.read_messages()
    bagName = bag.filename

    # create a new directory
    folder = string.split(bagName, ".bag")[0]
    if not os.path.exists(folder):
        os.makedirs(folder)

    # get list of topics from the bag
    listOfTopics = []
    for topic, msg, t in bagContents:
        if topic not in listOfTopics:
            listOfTopics.append(topic)

    for topicName in listOfTopics:
        if topic_names is not None and string.lower(topicName) not in topic_names:
            continue

        # Create a new CSV file for each topic
        filename = folder + '/' + string.replace(topicName, '/', '_slash_') + '.csv'
        with open(filename, 'w+') as csvfile:
            file_writer = csv.writer(csvfile, delimiter=',')
            firstIteration = True  # allows header row
            for subtopic, msg, t in bag.read_messages(topicName):
                # for each instant in time that has data for topicName parse data from this instant, which is of the
                # form of multiple lines of "Name: value\n", then put it in the form of a list of 2-element lists

                msgString = str(msg)
                msgList = string.split(msgString, '\n')
                instantaneousListOfData = []
                for nameValuePair in msgList:
                    splitPair = string.split(nameValuePair, ':')
                    for i in range(len(splitPair)):  # should be 0 to 1
                        splitPair[i] = string.strip(splitPair[i])
                    instantaneousListOfData.append(splitPair)

                # write the first row from the first element of each pair
                if firstIteration:  # header
                    headers = ["rosbagTimestamp"]  # first column header
                    for pair in instantaneousListOfData:
                        headers.append(pair[0])
                    file_writer.writerow(headers)
                    firstIteration = False
                    
                # write the value from each pair to the file
                values = [str(t)]  # first column will have rosbag timestamp
                for pair in instantaneousListOfData:
                    if len(pair) > 1:
                        values.append(pair[1])
                file_writer.writerow(values)
    bag.close()
print("Done reading all " + str(numberOfFiles) + " bag files.")

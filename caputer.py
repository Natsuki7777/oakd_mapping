#!/usr/bin/env python3

from time import sleep
import cv2
import depthai as dai
import numpy as np
import time

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

ctrl = dai.CameraControl()
ctrl.setAutoExposureLock(True)
ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
colorCam = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)
color = pipeline.create(dai.node.XLinkOut)
recLeft = pipeline.create(dai.node.XLinkOut)
recRight = pipeline.create(dai.node.XLinkOut)
disparity = pipeline.create(dai.node.XLinkOut)
conf = pipeline.create(dai.node.XLinkOut)

color.setStreamName("color")
recLeft.setStreamName("recLeft")
recRight.setStreamName("recRight")
disparity.setStreamName("disparity")
conf.setStreamName("conf")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
colorCam.setPreviewSize(1920, 1080)
colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

# Linking
colorCam.preview.link(color.input)
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)

depth.disparity.link(disparity.input)
depth.rectifiedLeft.link(recLeft.input)
depth.rectifiedRight.link(recRight.input)
depth.confidenceMap.link(conf.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the disparity frames from the outputs defined above
    qdis = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
    qconf = device.getOutputQueue(name="conf", maxSize=1, blocking=False)
    qrecLeft = device.getOutputQueue(name="recLeft", maxSize=1, blocking=False)
    qrecRight = device.getOutputQueue(name="recRight", maxSize=1, blocking=False)
    qcolor = device.getOutputQueue(name="color", maxSize=1, blocking=False)
    direc = "./images"
    count = 0

    while True:
        time.sleep(1)
        inDisparity = qdis.get()  # blocking call, will wait until a new data has arrived
        disframe = inDisparity.getFrame()
        cv2.imwrite(direc + "/" + str(count).zfill(3)+"disparity" + ".png", disframe)

        inConf = qconf.get()  # blocking call, will wait until a new data has arrived
        confframe = inConf.getFrame()
        cv2.imwrite(direc + "/" + str(count).zfill(3)+"conf" + ".png", confframe)

        inRecLeft = qrecLeft.get()  # blocking call, will wait until a new data has arrived
        recleftframe = inRecLeft.getFrame()
        cv2.imwrite(direc + "/" + str(count).zfill(3)+"recLeft" + ".png", recleftframe)

        inRecRight = qrecRight.get()  # blocking call, will wait until a new data has arrived
        recrightframe = inRecRight.getFrame()
        cv2.imwrite(direc + "/" + str(count).zfill(3)+"recRight" + ".png", recrightframe)

        inColor = qcolor.get()  # blocking call, will wait until a new data has arrived
        colorframe = inColor.getFrame()
        cv2.imwrite(direc + "/" + str(count).zfill(3)+"color" + ".png", colorframe)

        count += 1


        if cv2.waitKey(1) == ord('q'):
            break

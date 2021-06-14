# Face-Tracker Project


## Contents
 - [Problem Statement](#Problem-Statement)
 - [Executive Summary](#Executive-Summary)
 - [File Directory](#File-Directory)
 - [Data Description](#Data-Description)
 - [Conclusion](#Conclusion)
 - [Areas for Further Research/Study](#Areas-for-Further-Research/Study)
 - [Sources](#Sources)
 - [Visualizations](#Visualizations)


## Problem Statement
[(back to top)](#Face-Tracker-Project)

Is it possible to track a person and their movement with a moving camera? I plan on exploring this with a raspberry pi and moving camera, and possibly exploring the idea of tracking a toddler (my son).


## Executive Summary

[(back to top)](#Face-Tracker-Project)

While the YOLO algorithm performs better, a Neural Net does not perform on restricted resources, such as the Raspberry Pi - not even YOLOv4-tiny! The Haar Cascade seems to perform better in this capacity since it is a quick and light weight algorithm, even after 20 years it is still useful. 

However, not to take away from the Haar Cascade algorithm, if resources were not an issue the YOLOv4-tiny algorithm has little to no latency and out-performs the Haar Cascade algorithm.


## File Directory
[(back to top)](#Face-Tracker-Project)

Face-Tracker<br />
|<br />
|__ assets<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ img <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ balena-etcher.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ balena_etcher_logo.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ haar_cascade_1.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ haar_cascade_2.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ haar_wavelet_1.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ hardware_pic.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ movidius.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ ncs2-lid-box.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ opencv_logo.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ openvino-logo.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ pimoroni_logo.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ Propeller_hat.svg.med.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ pth-assembled.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ Raspberry_Pi_OS_Logo.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ raspi-config.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ raspi.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ yolov4_stats.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ yolo_01.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ yolo_02.png <br />
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|__ yolo_03.png <br />
|<br />
|__ cfg<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ custom-yolov4-tiny-detector_face.cfg <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ custom-yolov4-tiny-detector.cfg <br />
|<br />
|__ code<br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ camera_app.py <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ yolo_model.py <br />
|<br />
|__ models <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ custom-yolov4-tiny-detector_best.weights <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ custom-yolov4-tiny-detector_face_best.weights <br />
|<br />
|__ names <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ custom-yolov4-tiny-detector_face.names <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ custom-yolov4-tiny-detector.names <br />
|<br />
|__ notebooks <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ 01_install_opencv.ipynb <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ 02_install_pantilthat.ipynb <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ 03_toddler-tracker_YOLOv4-tiny-Darknet-Roboflow.ipynb <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ 04_toddler-tracker-Face_YOLOv4-tiny-Darknet-Roboflow.ipynb <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ 05_comparing_models_and_hardware.ipynb <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ Face-Tracker_Report.ipynb <br />
|<br />
|__ presentation <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ Face-Tracker_presentation.pdf <br />
|&nbsp;&nbsp;&nbsp;&nbsp;|__ Face-Tracker_presentation.pptx <br />
|<br />
|__ README.md <br />


## Data Description
[(back to top)](#Face-Tracker-Project)

The Data for this project were images of my son and my family. For privacy, I am not including this as part of the repository. Please contact me if you would like to see the photos I used for this project.



## Conclusion
[(back to top)](#Face-Tracker-Project)

Looking at the comparison of render-times for each device and model, it is easy to see that my laptop was able to process facial recognition and image rendering better. This makes sense since my laptop has 4 times the RAM And 50% more CPU cores. But until I can get a pan-tilt mechanism for my laptop, I will likely stick with Haar Cascades on my Raspberry Pi.

| Device | Model        | FPS       | Render-Time |
| ------ | ------------ | --------- | ----------- |
| Laptop | Haar Cascade | 43.831705 | 0.027543    |
|        | YOLO-Body    | 23.610659 | 0.047073    |
|        | YOLO-Face    | 26.130075 | 0.041805    |
| Ras-Pi | Haar Cascade | 7.896255  | 0.128951    |
|        | YOLO-Body    | 1.530603  | 0.654779    |
|        | YOLO-Face    | 1.539430  | 0.651276    |




## Future Considerations and Recommendations
[(back to top)](#Face-Tracker-Project)

 - Integration with Flask for local video-streaming.
 - Utilizing a 64-bit OS on the Raspberry Pi, like Ubuntu, to improve performance.
 - Incorporate the Intel Neural Compute stick 2 (or the Google COral Compute stick) to improve Neural Networks.
 - Implement Models on other hardware


## Sources
[(back to top)](#Face-Tracker-Project)

##### Installing opencv on raspberry pi:
https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/


##### OpenCV
https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html


##### Camera Pan-Tilt-Hat
https://pantilt-hat.readthedocs.io/en/latest/
https://learn.pimoroni.com/tutorial/sandyj/assembling-pan-tilt-hat
https://learn.pimoroni.com/tutorial/electromechanical/building-a-pan-tilt-face-tracker
https://github.com/pimoroni/pantilt-hat
https://github.com/pimoroni/PanTiltFacetracker/blob/master/facetracker_lbp.py


##### Intel Neural Compute Stick 2:
https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_raspbian.html
https://www.hackster.io/news/getting-started-with-the-intel-neural-compute-stick-2-and-the-raspberry-pi-6904ccfe963
https://www.youtube.com/watch?v=LmtHEBuJfII
https://www.youtube.com/watch?v=joElT3UfspA


##### Haar Casacade
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
https://ieeexplore.ieee.org/document/710772
https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework
https://en.wikipedia.org/wiki/Cascading_classifiers
https://en.wikipedia.org/wiki/AdaBoost


##### YOLO
https://pjreddie.com/media/files/papers/yolo_1.pdf
https://pjreddie.com/media/files/papers/YOLOv3.pdf
https://stackoverflow.com/questions/57706412/what-is-the-working-and-output-of-getlayernames-and-getunconnecteddoutlayers
https://arxiv.org/abs/2004.10934
https://arxiv.org/abs/2011.08036
https://datascience.stackexchange.com/questions/65945/what-is-darknet-and-why-is-it-needed-for-yolo-object-detection


##### Roboflow.com (and CVAT)
https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208
https://blog.roboflow.com/cvat/
https://blog.roboflow.com/train-yolov4-tiny-on-custom-data-lighting-fast-detection/
https://www.youtube.com/watch?v=NTnZgLsk_DA&t=212s



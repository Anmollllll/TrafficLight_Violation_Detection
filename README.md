# TrafficLight_Violation_Detection

This is a custom project that uses computer vision and YOLOv8 model to detect whether vehicles or people are violating traffic light signals in a live video stream for a specific location based on traffic light simulation. It identifies violators in real-time, captures and saves their image, and labels them accordingly.
---
## How it works?
It captures live video and object detection is done using YOLOv8.A SORT algorithm is used to track the people and vehicles.It is actually based on traffic light simulation,(green and yellow) and red light describe a specific red region basically to find out if people cross the road when the light is green or yellow and if vehicle pass by the road lane when the light is red.It identifies if a detected object enters a defined red zone area or not.If an object i.e. people or vehicle do so,it saves a snapshot of the violators (with ID and label)
---
## Requirements

 Install all dependencies from requirements.txt
 Download SORT file(Comment the  import matplotlib and next below part if running on cloud notebook env)
 Download the yolov8x model
 GPU installation to run locally
---
# Project Dependencies

Camera angle of the camera
Locating positions of the specific bound areas which can be case of trial and error 
Integrating vehicle traffic light control
---
# Important Note
This project is designed as a custom solution for specific surveillance setups. The red zone area must be manually defined based on your camera angle and environment. As a result, it is not plug-and-play for general use without customization.

#  Traffic Light Violation Detection

This is a **custom computer vision project** that uses the **YOLOv8 object detection model** and **SORT tracking** to detect whether vehicles or pedestrians violate traffic light rules based on traffic light simulation in a live video stream. The system simulates traffic light changes and identifies violators based on predefined zones, capturing and saving images of violations in real-time.

---

##  How It Works

- Captures a live video stream (e.g., from webcam or CCTV).
- Uses **YOLOv8** for detecting people and vehicles.
- Uses **SORT** for object tracking across frames.
- Simulates traffic light signals (`green`, `yellow`, and `red`) based on a timer.
- Defines forbidden zones (as polygons):
  - Pedestrian zone (active during green/yellow light).
  - Vehicle zone (active during red light).
- If an object enters a forbidden zone during the wrong signal:
  - It is labeled as a **violator**.
  - A snapshot of the violator is saved with ID and timestamp.

---

##  Requirements

- Install dependencies listed in `requirements.txt`
- Download the `yolov8x` model from the [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- Add `SORT.py` tracker (remove or comment `matplotlib` imports if using on cloud notebooks)
- A system with GPU is recommended for real-time performance


Important Note:
This project is designed as a custom solution for specific surveillance setups. The red zone area must be manually defined based on your camera angle and environment. As a result, it is not plug-and-play for general use without customization.


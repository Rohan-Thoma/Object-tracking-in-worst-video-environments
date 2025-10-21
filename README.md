# 🚦 A Practical Approach to Reliable Tracking on Low-FPS videos (ft. Road Footage)

This project demonstrates a **robust end-to-end pipeline** for detecting and tracking road signs, even in **extremely low-FPS, low-quality video footage**.  
Unlike conventional methods that rely solely on per-frame detection, this solution combines **a YOLOv11-based detector** with a **finely tuned Norfair tracking algorithm** to achieve **temporal consistency, ID stability, and smooth tracking** — even when objects briefly disappear or move abruptly between frames.

---

![output2](https://github.com/user-attachments/assets/1d59368e-6d6a-40a1-beba-41ce5b807126)

### As you can see from the GIF above, the tracking-id stays consistent even though the video sample is playing at 2 FPS.

---
## 🧭 Table of Contents
1. [Overview](#overview)
2. [Dataset & Preprocessing](#dataset--preprocessing)
3. [Model Training](#model-training)
4. [Detection on Unseen Data](#detection-on-unseen-data)
5. [Tracking Algorithm](#tracking-algorithm)
6. [Results](#results)
7. [How to Run](#how-to-run)
8. [Future Work](#future-work)
9. [Key takeaway](#key-takeaways)

---

## 🧩 Overview

The project focuses on:
- Developing a **custom tracking pipeline** that works even on **1–2 FPS video footage**.  
- Enhancing detection stability through **relabeling and data cleaning**.  
- Demonstrating **how to pair deep learning with lightweight geometric tracking logic** for resource-constrained environments.

---

## 🗂️ Dataset & Preprocessing

### Dataset Overview
- **Total images:** 6000  
- **Usable after cleaning:** 1474  
- **Classes:**  
  - `advisory speed mph`  
  - `directional arrows`  
  - `do not enter`  
  - `stop`  
  - `wrong way`  

### Data Cleaning & Labeling
- Many labels were incomplete or missing; re-annotated using **labelImg**.
- Added two new classes: `stop` and `wrong way` for completeness.
- Ensured that **each image** had **consistent and accurate annotations**.

### Augmentation
- Applied **rotation, brightness, and scale augmentations** to simulate real-world variations.
- Final dataset of **1474 images / 5 classes** used for training and evaluation.

---

## 🧠 Model Training

Two YOLOv11 models were trained and compared for performance:

| Model | mAP@0.5 | Training Speed | Notes |
|--------|----------|----------------|-------|
| **YOLOv11n (Nano)** | 0.57 | ⚡ Fast | Lightweight but limited accuracy on new classes |
| **YOLOv11m (Medium)** | 0.71 | 🧠 Moderate | Significantly better generalization & stability |

The **YOLOv11m model** achieved a **14% higher mAP** and was selected for inference and tracking tasks.

---

## 🎥 Detection on Unseen Video

### Video Characteristics
- **Low frame rate:** ~1–2 FPS  
- **Perspective:** From a moving vehicle  
- **Challenges:** Motion blur, sudden scale changes, abrupt frame jumps  

### Detection Observations
- Despite the low FPS, the **YOLOv11m model** effectively detected road signs, including partially occluded or distant ones.  
- Occasional missed detections occurred due to frame skips, but overall detection was **stable and accurate**.

---

## 🔍 Tracking Algorithm

The **core highlight** of this project lies in the **custom tracking setup** built using the **Norfair library** — chosen over default YOLO trackers for its flexibility and resilience to missing detections.

### Why Norfair?
- Designed for **non-continuous frame streams** (like low-FPS CCTV footage).  
- Allows **custom distance functions and persistence tuning** for object identity stability.

### Tracker Configuration
- **Distance function:** `mean_manhattan`  
- **Key parameters:**
```python
  distance_threshold = tuned_value
  hit_counter_max = higher_value_for_stability
````

* A **higher `hit_counter_max`** ensured that object IDs persisted even when detections were temporarily missed.
* **Distance threshold** was carefully adjusted to prevent ID switches on abrupt motion.

### Tracking Logic

```python
for each frame in video:
    detections = yolov11m.predict(frame)
    tracker.update(detections)
    for each active track:
        draw bounding box + track ID
```

### Tracking Performance

* **Stable IDs** maintained for most objects throughout the video.
* Successfully handled objects **disappearing and reappearing**.
* Minor ID resets occurred only on **severe frame skips**, expected at ~1 FPS input.
* Overall, the **YOLOv11m + Norfair** combination proved **robust and reliable** under challenging conditions.

---

## 📊 Results

| Metric             | Result              |
| ------------------ | ------------------- |
| Detection mAP@0.5  | **0.71 (YOLOv11m)** |
| Tracking Stability | **High**            |
| FPS Tested         | **1.0 – 2.0**       |
| ID Switch Rate     | **Low (<10%)**      |

**Qualitative Outcome:**

* Smooth, stable tracking visualization with consistent track IDs.
* Effective in motion-blur and lighting variation scenarios.

**Example Console Output:**

```
[Frame 12] STOP_SIGN_3 → position updated
[Frame 13] STOP_SIGN_3 → re-identified after 2 skipped frames
[Frame 14] DIRECTIONAL_ARROW_1 → continuous tracking
```

---

## ⚙️ How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/Rohan-Thoma/Object-tracking-in-worst-video-environments.git
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**

   ```bash
   jupyter notebook Sign_board.ipynb
   ```

4. **(Optional)** Replace the demo video with your own under `/input/`.

---

## 🚀 Future Work

* Integrate **Kalman Filters** for motion prediction and smoother trajectories.
* Implement **confidence-weighted IoU matching** to refine association logic.
* Extend the system to **multi-camera sign tracking** for traffic analysis.

---

## 💡 Key Takeaway

> Even with **poor video quality** and **low frame rate**, intelligent combination of
> detection and geometric tracking can yield **stable, production-ready results** —
> proving that **smart engineering often outperforms brute-force deep learning**.


# Number Plate Recognition & Helmet Detection

Two interconnected computer vision projects demonstrating end-to-end ML pipeline development: custom CRNN architecture for OCR and multi-model helmet violation detection system.

---

## Project Overview

1. **CRNN Number Plate Recognition** - Custom OCR pipeline (pretrained ResNet-18 + Bi-LSTM/CTC) for number plate recognition
2. **Helmet Detection System** - Real-time violation detection using multi-model pipeline

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     HELMET DETECTION SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Video Input    │
                    └──────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   YOLO Object Detection       │
              │   (4 Classes)                 │
              │   • Bike                      │
              │   • Person with helmet        │
              │   • Person without helmet     │
              │   • Number plate             │
              └───────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
        ┌──────────────────┐  ┌──────────────────┐
        │  DeepSort        │  │  Violation       │
        │  Bike Tracking   │  │  Detection       │
        │  (Unique IDs)    │  │  (5-frame rule)  │
        └──────────────────┘  └──────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    ┌──────────────────┐
                    │  Number Plate    │
                    │  Detected?       │
                    └──────────────────┘
                              │ Yes
                              ▼
              ┌───────────────────────────────┐
              │   CRNN OCR Model              │
              │   (pretrained ResNet-18 + Bi-LSTM/CTC)       │
              └───────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Output          │
                    │  • Annotated     │
                    │    Video         │
                    │  • Evidence      │
                    │    Images        │
                    │  • Plate Text    │
                    └──────────────────┘
```


## Projects

### 1. CRNN Number Plate Recognition (Experimental)

Custom CRNN pipeline for number plate OCR, using a pretrained ResNet-18 backbone (through layer3) with a custom Bi-LSTM + CTC head.

**Features:**
- ResNet-18 backbone (pretrained, fine-tuned) + custom Bi- LSTM/CTC head
- CTC loss for variable-length sequences
- Targeted blur/motion/compression data augmentation
- ONNX export for deployment

**Details:**
- Dataset: 456 number plate images
- Architecture:  ResNet-18(conv1 -> layer3) + Bi-LSTM + CTC 
- Input: 100x200 RGB images
- Output: Alphanumeric text (a-z, A-Z, 0-9)

**Results:**
- Identified a domain-shift problem: model trained only on clean images performed poorly on real-world blurred plates (56% character accuracy on blur)
- Retrained with targeted blur/motion/compression augmentation, a pretrained ResNet-18 backbone, and extended training (100+ epochs with early stopping)
- **Improved character accuracy on blurred test images from 56% → 68%**

**Prediction Results:**


*Sample predictions on test images*
<img width="1726" height="596" alt="Image" src="https://github.com/user-attachments/assets/ac4ea936-f0b5-4a91-893c-fddf1ecc514d" />

**Files:**
```
CRNN/
├── model.py         
├── train.py         
├── test.py          
├── export_onnx.py   
└── requirements.txt
```

**Usage:**
```bash
python train.py    # Train model
python test.py     # Test model
python export_onnx.py  # Export to ONNX
```

---

### 2. Helmet Detection System

Multi-model pipeline for automated helmet violation detection.

**Features:**
- Real-time object detection (4 classes)
- Multi-object tracking
- Violation confirmation (5-frame threshold)
- Evidence collection
- Number plate OCR

**Details:**
- Detection: YOLO26
- Tracking: DeepSort
- OCR: Custom CRNN
- Classes: Bike, person with/without helmet, number plate

**Demo Videos:**

**Multiple Vehicle Tracking:**

*DeepSort tracking multiple bikes with unique IDs*

https://github.com/user-attachments/assets/7b947ea9-9aab-4f8e-bb79-762b9b4e6e3d

https://github.com/user-attachments/assets/2aa6d687-b74e-4ff5-8e17-5e39de326599


**Violation Detection:**

*Person without helmet detection and tracking*

https://github.com/user-attachments/assets/db516259-7eb5-4f41-aa45-266745295474

**Number Plate OCR:**

*Real-time number plate recognition*

https://github.com/user-attachments/assets/cbb0f611-3eeb-4699-9056-3b41027f4f34

**System Output:**

*Evidence collection: bike, person, and number plate crops*

https://github.com/user-attachments/assets/4b7eb1f3-a312-4947-afcb-591380df35b8

**Files:**
```
Helmet-Detection/
├── main.py
├── requirements.txt
└── USAGE.md
```

**Usage:**
```bash
python main.py
```

**Output:**
```
track_output/HH-MM-SS/
├── bike_1_bike.jpg               
├── bike_1_person_no_helmet.jpg   
└── bike_1_number_plate.jpg       

output_video.mp4  # Annotated video with detections
```

---

## Results & Learnings

### What Worked:
- Domain shift (clean to blur) - addressed via targeted blur/motion augmentation (Albumentations: MotionBlur, GaussianBlur,
ImageCompression) , a pretrained ResNet-18 backbone , and extended training (100+ epochs with early stopping), improving character accuracy on blurred plates from 56% to 68%
- Multi-model pipeline integration
- DeepSort maintains bike IDs across frames
- ONNX deployment ready

### Challenges:

- Small dataset (456 images) limits generalization
- Person-to-bike association needs spatial logic

### Learnings:
- CTC loss for variable-length sequences
- Data augmentation strategies
- Multi-model integration
- Domain adaptation importance
- ONNX

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, ONNX Runtime, Ultralytics YOLO |
| **Computer Vision** | OpenCV, PIL, torchvision , Albumentations|
| **Tracking** | DeepSort |
| **ML Tools** | scikit-learn, NumPy, matplotlib |

---

## Future Improvements

**CRNN:**
- Larger dataset (1000+ images)
- Attention mechanisms
- Confidence scores

**Helmet Detection:**
- Spatial association (distance-based)
- Real-time optimization
- Multi-bike improvements

---

## Installation

```bash
# Clone repository
git clone https://github.com/Pooja-Vachhad/Helmet_Detection.git

# Install CRNN dependencies
cd CRNN
pip install -r requirements.txt

# Install Helmet Detection dependencies
cd Helmet-Detection
pip install -r requirements.txt
```

---

## License

MIT License

---

**Note:** Experimental project demonstrating CV concepts. Helmet detection uses simplified person-to-bike association for proof-of-concept.






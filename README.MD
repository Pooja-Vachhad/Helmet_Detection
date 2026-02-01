# Number Plate Recognition & Helmet Detection

Two interconnected computer vision projects demonstrating end-to-end ML pipeline development: custom CRNN architecture for OCR and multi-model helmet violation detection system.

---

## Project Overview

1. **CRNN Number Plate Recognition** - Custom OCR model built from scratch
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
              │   (Custom Architecture)       │
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

---

## CRNN Architecture

```
Input Image (3, 100, 200)
         │
         ▼
┌─────────────────┐
│   CNN Blocks    │  ← Feature Extraction
│   3 → 64 → 128  │     • 3 Conv blocks
│   → 256         │     • BatchNorm + ReLU
└─────────────────┘     • MaxPooling
         │
         ▼
┌─────────────────┐
│ Reshape to      │  ← Prepare for sequence
│ (B, T, C*H)     │     
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  FC Layer       │  ← Dimension reduction
│  256*H → 128    │     + Dropout (0.5)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Bidirectional   │  ← Sequence modeling
│ LSTM (2 layers) │     • Hidden: 128
│                 │     • Dropout: 0.3
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Classifier     │  ← Character prediction
│  256 → 63 chars │     (a-z, A-Z, 0-9 + blank)
└─────────────────┘
         │
         ▼
   CTC Decoding
         │
         ▼
   Output Text
```

---

## Projects

### 1. CRNN Number Plate Recognition (Experimental)

Custom CRNN architecture for number plate OCR.

**Features:**
- Custom CNN-RNN architecture 
- CTC loss for variable-length sequences
- Data augmentation
- ONNX export for deployment

**Details:**
- Dataset: 456 clean number plate images
- Architecture: 3 CNN blocks + Bi-LSTM + CTC
- Input: 100x200 RGB images
- Output: Alphanumeric text (a-z, A-Z, 0-9)

**Results:**
- Tested on blurred images
- 56% character accuracy on blur (trained on clean data)
- Demonstrates domain adaptation challenge

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
- Detection: YOLOv8 (custom trained)
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
- Custom CRNN learns clean data effectively
- Multi-model pipeline integration
- DeepSort maintains bike IDs across frames
- ONNX deployment ready

### Challenges:
- Domain shift (clean to blur) reduces accuracy
- Small dataset (456 images) limits generalization
- Person-to-bike association needs spatial logic

### Learnings:
- CTC loss for variable-length sequences
- Data augmentation strategies
- Multi-model integration
- Domain adaptation importance
- Model deployment (ONNX)

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, ONNX Runtime, Ultralytics YOLO |
| **Computer Vision** | OpenCV, PIL, torchvision |
| **Tracking** | DeepSort |
| **ML Tools** | scikit-learn, NumPy, matplotlib |

---

## Future Improvements

**CRNN:**
- Train on blurred data for robustness
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
git clone <repo-url>

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

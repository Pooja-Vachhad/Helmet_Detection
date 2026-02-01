# Helmet Detection System - Usage

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit these paths in `main.py`:

```python
CRNN_MODEL_PATH = "path/to/crnn_model.onnx"
YOLO_MODEL_PATH = "path/to/best.onnx" 
VIDEO_PATH = "path/to/video.mp4"
```

## Run

```bash
python main.py
```

## Output

Creates folder `track_output/HH-MM-SS/` with:
- `output_video.mp4` - annotated video with detections and OCR
- `bike_X_bike.jpg` - bike images (for violators)
- `bike_X_person_no_helmet.jpg` - person without helmet images
- `bike_X_number_plate.jpg` - number plate crops

## How It Works

1. **YOLO Detection**: Detects 4 classes in each frame
   - bike
   - person with helmet
   - person without helmet
   - number plate

2. **DeepSort Tracking**: Tracks bikes across frames with unique IDs

3. **Violation Detection**: 
   - Checks if "person without helmet" is detected
   - Needs 5 consecutive frames to confirm violation
   - Prevents false positives

4. **Evidence Collection**: Saves images only for confirmed violations
   - Bike crop
   - Person without helmet crop
   - Number plate crop

5. **OCR**: Runs CRNN model on detected number plates to read text

6. **Output Video**: Annotated with bounding boxes, labels, and OCR results

## Models Required

- **YOLO model**: Custom trained on 4 classes (bike, person with/without helmet, number plate)
- **CRNN model**: For number plate OCR (reads alphanumeric characters)

## Notes

- Violation threshold is set to 5 frames (adjustable in code)
- OCR runs on plates with confidence >= 0.5
- Each bike gets unique tracking ID
- Images saved only once per violation per bike
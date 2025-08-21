import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# ---------------------------
# 1. Load the image
# ---------------------------
image_path = "src/images/bedroom.jpg"  # replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------------------
# 2. Configure Detectron2
# ---------------------------
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # confidence threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

predictor = DefaultPredictor(cfg)

# ---------------------------
# 3. Run segmentation
# ---------------------------
outputs = predictor(image_rgb)
masks = outputs["instances"].pred_masks.cpu().numpy()
classes = outputs["instances"].pred_classes.cpu().numpy()

# COCO class IDs for furniture objects
# chair=56, couch=57, potted plant=58, bed=59, dining table=60, toilet=61, tv=62
furniture_classes = [56, 57, 58, 59, 60, 61, 62]

# ---------------------------
# 4. Combine furniture masks
# ---------------------------
combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
for mask, cls in zip(masks, classes):
    if cls in furniture_classes:
        combined_mask = np.logical_or(combined_mask, mask)

# Convert boolean mask to 0-255 image
combined_mask = (combined_mask * 255).astype(np.uint8)

# ---------------------------
# 5. Save and visualize
# ---------------------------
cv2.imwrite("furniture_mask.png", combined_mask)

# Optional: visualize mask
cv2.imshow("Furniture Mask", combined_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
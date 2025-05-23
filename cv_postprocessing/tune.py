# tuning
from ultralytics import YOLO
def tunit():
    # Initialize the YOLO model
    model = YOLO("yolov8s-seg.pt")
    
    # Define search space
    search_space = {
        "lr0": (1e-5,1e-3, 1e-1),
        "degrees": (0.0, 15.0,45.0),
        "batch":(1, 2, 4, 8, 16)
    }
    
    # Tune hyperparameters on COCO8 for 30 epochs
    model.tune(
        data='yolo.yaml',
        iterations=300,
        optimizer="AdamW",
        epochs=10,
        space=search_space,
        plots=True,
        save=True,
        val=True,
        verbose=True, 
         device=[0],
         imgsz=640,
         augment=True,
    )
if __name__ == '__main__':
   tunit() 
# model.train(data='yolo.yaml', epochs=500, device=[0],imgsz=640, augment=True, save_period=5, lr0=0.0001,mask_ratio=4, batch=16)
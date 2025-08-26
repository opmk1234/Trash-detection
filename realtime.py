from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model = YOLO("best.pt")

results = model.predict(source="datasets/images/val", save=False, conf=0.05)

#results = model("dataset/images/val/400.jpg", conf=0.01)
#results[0].show() 
for r in results[:5]:  # show first 5 images
    im = r.plot()  # annotated image (numpy array BGR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.axis("off")
    plt.show()

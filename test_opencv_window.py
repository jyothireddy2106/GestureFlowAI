
import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)

image = np.zeros((400, 600, 3), dtype=np.uint8)
image[:] = (0, 255, 0)

cv2.putText(
    image,
    "OPENCV WINDOW TEST",
    (35, 200),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 0),
    3,
)

cv2.namedWindow("GestureFlow AI Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("GestureFlow AI Test", 600, 400)
cv2.imshow("GestureFlow AI Test", image)

print("OpenCV window displayed. Press Q to close.")

while True:
    key = cv2.waitKey(100)

    if key != -1 and (key & 0xFF) == ord("q"):
        break

cv2.destroyAllWindows()
print("Test finished.")

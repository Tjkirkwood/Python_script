import cv2
import numpy as np

blank = np.zeros((300, 600, 3), dtype=np.uint8)
cv2.putText(blank, "Test Window", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
cv2.imshow("Window Test", blank)
cv2.waitKey(0)
cv2.destroyAllWindows()

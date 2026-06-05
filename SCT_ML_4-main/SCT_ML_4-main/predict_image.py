import cv2
import joblib
model = joblib.load("hand_gesture_model.pkl")
image_path = input("Enter image path: ").strip()
img = cv2.imread(image_path)
if img is None:
    print("Image not found!")
    exit()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (64, 64))
features = gray.flatten().reshape(1, -1)
prediction = model.predict(features)[0]
print("Predicted Gesture:", prediction)
cv2.putText(
    img,
    f"Prediction: {prediction}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)
big_img = cv2.resize(
    img,
    (800, 800),
    interpolation=cv2.INTER_NEAREST
)
cv2.imshow("Hand Gesture Prediction", big_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
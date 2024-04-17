import cv2
import mediapipe as mp
import os

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_hand_landmarks(self, image):
        results = self.hands.process(image)
        return results

    def draw_landmarks_on_image(self, image, results):
        landmark_coords = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                for landmark in hand_landmarks.landmark:
                    landmark_x = landmark.x * image.shape[1]
                    landmark_y = landmark.y * image.shape[0]
                    landmark_coords.append((landmark_x, landmark_y))
        return image, landmark_coords

    def capture_image(self):
        hand_detector = HandDetector()
        cap = cv2.VideoCapture(0)
        frame_count = 0
        save_interval = 20  # seconds
        output_folder = "./gestures/test/"
        os.makedirs(output_folder, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Display the frame
            cv2.imshow("Frame", frame)

            # Capture and save frame every save_interval seconds
            if frame_count % (save_interval * cap.get(cv2.CAP_PROP_FPS)) == 0:
                # Detect hand landmarks
                detection_result = self.detect_hand_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Draw landmarks on the image and get coordinates
                annotated_image, landmark_coords = self.draw_landmarks_on_image(frame, detection_result)
                print("Hand Landmark Coordinates:", landmark_coords)
                # print(landmark_coords.shape)
                # Save the annotated image
                cv2.imwrite(f"{output_folder}annotated_frame{frame_count}.jpg", annotated_image)
                

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return(landmark_coords)

# if __name__ == "__main__":
#     main()

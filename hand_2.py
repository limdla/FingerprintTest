import cv2
import mediapipe as mp

def main():
    # Mediapipe Hand Tracking 모델을 로드합니다.
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 웹캠을 초기화합니다.
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        # 이미지를 BGR에서 RGB로 변환합니다.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mediapipe로 손 인식을 수행합니다.
        results = mp_hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 다섯 손가락 끝의 위치를 얻습니다.
                h, w, c = image.shape
                finger_tips = [(int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                                int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * h)),
                               (int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x * w),
                                int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y * h)),
                               (int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].x * w),
                                int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y * h)),
                               (int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].x * w),
                                int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].y * h)),
                               (int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x * w),
                                int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y * h))]

                # 모자이크 영역 크기
                size = 26
                # 모자이크 스케일
                scale = 3

                for finger_tip in finger_tips:
                    # 모자이크할 영역을 잘라냅니다.
                    roi = image[finger_tip[1] - size // 2:finger_tip[1] + size // 2,
                                finger_tip[0] - size // 2:finger_tip[0] + size // 2]

                    # 영역의 크기가 유효한지 확인합니다.
                    if roi.shape[0] == size and roi.shape[1] == size:
                        # 잘라낸 영역을 다시 확대/축소하여 모자이크 처리합니다.
                        small_roi = cv2.resize(roi, (size // scale, size // scale))
                        mosaic_roi = cv2.resize(small_roi, (size, size), interpolation=cv2.INTER_NEAREST)

                        # 원래 이미지에 모자이크 처리한 영역을 적용합니다.
                        image[finger_tip[1] - size // 2:finger_tip[1] + size // 2,
                              finger_tip[0] - size // 2:finger_tip[0] + size // 2] = mosaic_roi

        # 이미지를 화면에 표시합니다.
        cv2.imshow('Hand Tracking', image)

        # 'q' 키를 누르면 종료합니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료할 때, 리소스를 해제합니다.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import time
from db import FruitDB
from detector import detect_fruit


def main():
    db = FruitDB('fruits.db')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    last_label = None
    last_shown = 0

    print('Press q to quit. Showing detection and nutrition info when stable.')
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to grab frame')
            break

        label, conf = detect_fruit(frame)

        # show label on frame
        display = frame.copy()
        text = f'{label} ({conf:.2f})' if label else 'No detection'
        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow('Fruit Detector', display)

        # if same label persists for >1s, query db
        now = time.time()
        if label and label == last_label and now - last_shown > 1.0:
            info = db.get_info(label)
            print('\nDetected:', label, 'confidence:', conf)
            if info:
                print('Nutrition info:')
                for k,v in info.items():
                    print(f'  {k}: {v}')
            else:
                print('No info found in DB for', label)
            last_shown = now

        if label != last_label:
            last_label = label
            last_shown = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

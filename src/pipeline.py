import text_detector
import text_recognizer
import role_classifier
import bar_detection_raw_data_extraction


def run_pipeline():
    text_detector.main()
    text_recognizer.main()
    role_classifier.main()
    bar_detection_raw_data_extraction.main()


if __name__ == "__main__":
    run_pipeline()

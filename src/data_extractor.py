from __future__ import annotations

import config
import bar_detection_extraction


# Legacy entrypoint config expected by app.py.
TASK4_CONFIG = config.returnTestTask4_Config()


def main():
    if isinstance(TASK4_CONFIG, dict):
        mapped = dict(TASK4_CONFIG)

        if "output_excel" in mapped and "output_csv" not in mapped:
            mapped["output_csv"] = mapped["output_excel"]
        if "output_json" in mapped and "output_dir" not in mapped:
            mapped["output_dir"] = mapped["output_json"]

        config.Task5_detect_extraction.update(mapped)
        if "output_csv" in mapped:
            config.Output_Excel_Task_4 = mapped["output_csv"]

    bar_detection_extraction.main()


if __name__ == "__main__":
    main()

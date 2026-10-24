import json
from pathlib import Path

from ultralytics import YOLO


def main():
    project_dir = Path(__file__).resolve().parents[1]

    model_path = (
        project_dir
        / "outputs"
        / "training"
        / "lighter"
        / "weights"
        / "best.pt"
    )
    test_dir = project_dir / "data" / "test"
    output_dir = project_dir / "outputs"

    if not model_path.exists():
        raise FileNotFoundError(f"Не найдены веса модели: {model_path}")

    if not test_dir.exists():
        raise FileNotFoundError(f"Не найдена папка теста: {test_dir}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [
        path
        for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ]

    if not images:
        raise RuntimeError(f"В папке {test_dir} нет изображений")

    print(f"Модель: {model_path}")
    print(f"Тестовых изображений: {len(images)}")

    model = YOLO(str(model_path))

    results = model.predict(
        source=[str(path) for path in images],
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device="cpu",
        save=True,
        project=str(output_dir),
        name="test_predictions",
        exist_ok=True,
        verbose=False,
    )

    detections = []

    for result in results:
        height, width = result.orig_shape

        image_result = {
            "image": Path(result.path).name,
            "width": width,
            "height": height,
            "detections": [],
        }

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            class_ids = result.boxes.cls.cpu().tolist()

            for box, confidence, class_id in zip(
                boxes, confidences, class_ids
            ):
                class_id = int(class_id)

                image_result["detections"].append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(float(confidence), 4),
                    "bbox_xyxy": [
                        round(float(coordinate), 2)
                        for coordinate in box
                    ],
                })

        detections.append(image_result)

        print(
            f"{image_result['image']}: "
            f"найдено объектов — {len(image_result['detections'])}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "detections.json"

    report = {
        "model": "YOLO26n fine-tuned",
        "weights": str(model_path),
        "class_names": model.names,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "images": detections,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\nИнференс завершён.")
    print(f"Изображения с рамками: {output_dir / 'test_predictions'}")
    print(f"JSON с результатами: {json_path}")


if __name__ == "__main__":
    main()

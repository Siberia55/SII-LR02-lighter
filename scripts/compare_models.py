import json
from pathlib import Path

from ultralytics import YOLO


def run_model(model, images, output_dir, name):
    results = model.predict(
        source=[str(image) for image in images],
        conf=0.25,
        iou=0.45,
        imgsz=640,
        device="cpu",
        save=True,
        project=str(output_dir),
        name=name,
        exist_ok=True,
        verbose=False,
    )

    report = []

    for result in results:
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox_xyxy": [
                        round(float(value), 2)
                        for value in box.xyxy[0].tolist()
                    ],
                })

        report.append({
            "image": Path(result.path).name,
            "detections_count": len(detections),
            "detections": detections,
        })

        print(
            f"[{name}] {Path(result.path).name}: "
            f"{len(detections)} объектов"
        )

    return report


def main():
    root = Path(__file__).resolve().parents[1]
    test_dir = root / "data" / "test"
    output_dir = root / "outputs"

    images = sorted(
        path for path in test_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not images:
        raise RuntimeError(f"Нет изображений в {test_dir}")

    baseline_path = "yolo26n.pt"
    finetuned_path = (
        root / "outputs" / "training" / "lighter" / "weights" / "best.pt"
    )

    print("\n=== Baseline: исходная YOLO26n ===")
    baseline = YOLO(baseline_path)
    baseline_results = run_model(
        baseline, images, output_dir, "baseline_predictions"
    )

    print("\n=== Fine-tuned: дообученная YOLO26n ===")
    finetuned = YOLO(str(finetuned_path))
    finetuned_results = run_model(
        finetuned, images, output_dir, "finetuned_predictions"
    )

    report = {
        "test_images": len(images),
        "confidence_threshold": 0.25,
        "baseline_model": baseline_path,
        "finetuned_model": str(finetuned_path),
        "note": (
            "Исходная модель обучена на COCO и не имеет пользовательского "
            "класса lighter. Сравнение количества всех обнаружений "
            "не является прямым сравнением качества класса lighter."
        ),
        "baseline": baseline_results,
        "finetuned": finetuned_results,
    }

    report_path = output_dir / "model_comparison.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"\nОтчёт сохранён: {report_path}")
    print(f"Baseline изображения: {output_dir / 'baseline_predictions'}")
    print(f"Fine-tuned изображения: {output_dir / 'finetuned_predictions'}")


if __name__ == "__main__":
    main()

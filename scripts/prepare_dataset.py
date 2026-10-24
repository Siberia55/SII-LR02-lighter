from pathlib import Path
import random
import shutil

SOURCE = Path(
    "/mnt/nextcloud_HDD1_500G/lab2-lighter/datasets/"
    "job_1_dataset_2026_10_24_05_43_43_ultralytics yolo detection 1.0"
)
DEST = Path(
    "/mnt/nextcloud_HDD1_500G/lab2-lighter/project/data/dataset"
)

SEED = 42
VAL_COUNT = 8
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    image_dir = SOURCE / "images" / "train"
    label_dir = SOURCE / "labels" / "train"

    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("Не найдены images/train или labels/train")

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(images) < 2:
        raise RuntimeError(f"Недостаточно изображений: {len(images)}")

    missing_labels = [
        image.name for image in images
        if not (label_dir / f"{image.stem}.txt").is_file()
    ]
    if missing_labels:
        raise RuntimeError(
            "Нет файлов разметки для изображений: "
            + ", ".join(missing_labels)
        )

    # Проверяем формат разметки и координаты до копирования.
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        for line_no, line in enumerate(
            label.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{label.name}:{line_no}: ожидалось 5 значений"
                )

            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]

            if cls != 0:
                raise ValueError(
                    f"{label.name}:{line_no}: класс {cls}, ожидался 0"
                )
            if not all(0.0 <= v <= 1.0 for v in coords):
                raise ValueError(
                    f"{label.name}:{line_no}: координаты вне диапазона 0..1"
                )
            if coords[2] <= 0 or coords[3] <= 0:
                raise ValueError(
                    f"{label.name}:{line_no}: ширина и высота должны быть > 0"
                )

    if DEST.exists() and any(DEST.iterdir()):
        raise RuntimeError(
            f"Каталог назначения уже содержит данные: {DEST}\n"
            "Чтобы не перезаписать их, скрипт остановлен."
        )

    if VAL_COUNT >= len(images):
        raise RuntimeError("VAL_COUNT должен быть меньше числа изображений")

    random.Random(SEED).shuffle(images)
    val_images = images[:VAL_COUNT]
    train_images = images[VAL_COUNT:]

    for split, split_images in (
        ("train", train_images),
        ("val", val_images),
    ):
        dst_images = DEST / "images" / split
        dst_labels = DEST / "labels" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy2(image, dst_images / image.name)
            shutil.copy2(
                label_dir / f"{image.stem}.txt",
                dst_labels / f"{image.stem}.txt",
            )

    yaml_text = """path: .
train: images/train
val: images/val
names:
  0: lighter
"""
    (DEST / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("Датасет подготовлен.")
    print(f"Всего изображений: {len(images)}")
    print(f"Train: {len(train_images)}")
    print(f"Val: {len(val_images)}")
    print(f"Конфигурация: {DEST / 'data.yaml'}")


if __name__ == "__main__":
    main()

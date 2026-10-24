from pathlib import Path

from ultralytics import YOLO


def main():
    # Корневая папка проекта
    project_dir = Path(__file__).resolve().parents[1]

    # Конфигурация подготовленного датасета
    data_yaml = project_dir / "data" / "dataset" / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Не найден файл датасета: {data_yaml}"
        )

    print("=" * 60)
    print("Обучение YOLO26n для обнаружения зажигалок")
    print(f"Датасет: {data_yaml}")
    print("Устройство: CPU")
    print("Эпохи: 20")
    print("=" * 60)

    # Загружаем предобученную модель
    model = YOLO("yolo26n.pt")

    # Запускаем дообучение
    model.train(
        data=str(data_yaml),
        epochs=20,
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        project=str(project_dir / "outputs" / "training"),
        name="lighter",
        seed=42,
        deterministic=True,
        patience=10,
        pretrained=True,
        cache=False,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )

    print("\nОбучение завершено.")
    print(
        "Лучшие веса должны находиться здесь:"
        f"\n{project_dir / 'outputs' / 'training' / 'lighter' / 'weights' / 'best.pt'}"
    )


if __name__ == "__main__":
    main()

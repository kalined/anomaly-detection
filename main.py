import joblib

from anomaly_detection.models import ml_models


def main():
    print("Starting the random forest module from main.py...")
    ml_models.main()

    model = joblib.load("./models/random_forest_model.pkl")

    sample = [
        [
            15,
            15,
            0.4014,
            0.0,
            492508.0,
            0.051,
            4.94,
            5.35,
            4.8,
            0.5005,
            383445.0,
            92989.0,
            0.0,
            4.2,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ]

    scaler = joblib.load("./models/random_forest_scaler.pkl")
    sample_scaled = scaler.transform(sample)

    predicted_class = model.predict(sample_scaled)
    label_map = {
        0: "normal",
        1: "user_peak",
        2: "ddos",
        3: "cpu_load_anomaly",
        4: "server_unavailability",
        5: "memory_leak",
    }

    print("Predicted label:", label_map[int(predicted_class[0])])


if __name__ == "__main__":
    main()

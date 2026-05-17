
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import time
from datetime import datetime, timezone

from inference import predict_image
from model import MODEL_NAME, IMG_SIZE


app = FastAPI(title="Mosquito Classification API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "success": True,
        "message": "API is running"
    }


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)):
    start_time = time.perf_counter()

    image_bytes = await image.read()
    prediction = predict_image(image_bytes)

    latency = round((time.perf_counter() - start_time) * 1000)

    label = prediction["label"]
    confidence = prediction["confidence"]
    probabilities = prediction["probabilities"]

    sorted_probabilities = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True
    )

    colors = [
        4294523780,
        4294939032,
        4294947013,
        4294958421,
        4285128959,
        4280391411
    ]

    similarities = []

    for index, item in enumerate(sorted_probabilities):
        class_label, value = item

        similarities.append({
            "label": class_label,
            "value": value,
            "color": colors[index % len(colors)]
        })

    if confidence >= 0.95:
        confidence_badge = "High Confidence"
    elif confidence >= 0.70:
        confidence_badge = "Medium Confidence"
    else:
        confidence_badge = "Low Confidence"

    return {
        "success": True,
        "message": "Analysis complete",
        "data": {
            "label": label,
            "confidence": confidence,
            "confidence_badge": confidence_badge,
            "explanation_tabs": [
                {
                    "title": "Grad-CAM",
                    "heatmap_colors": [4281126911, 4294958421, 4294926902],
                    "overlay_colors": [1442803076, 1728043377]
                },
                {
                    "title": "Guided Grad-CAM",
                    "heatmap_colors": [4279004600, 4281974248, 4289655387],
                    "overlay_colors": [1714515929, 1442812567]
                },
                {
                    "title": "Grad-CAM++",
                    "heatmap_colors": [4279970726, 4280461543, 4294951237],
                    "overlay_colors": [1715158409, 1442288807]
                },
                {
                    "title": "Score-CAM",
                    "heatmap_colors": [4280042208, 4285128959, 4294962298],
                    "overlay_colors": [1715100617, 1442816192]
                },
                {
                    "title": "SHAP",
                    "heatmap_colors": [4280391411, 4293138685, 4294955500],
                    "overlay_colors": [1428264691, 1157627903]
                }
            ],
            "similarities": similarities,
            "details": [
                {
                    "label": "Model",
                    "value": MODEL_NAME
                },
                {
                    "label": "Input Size",
                    "value": f"{IMG_SIZE}x{IMG_SIZE}"
                },
                {
                    "label": "Latency",
                    "value": f"{latency} ms"
                },
                {
                    "label": "Date",
                    "value": datetime.now(timezone.utc).isoformat()
                }
            ]
        }
    }

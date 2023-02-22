import base64
from flask import Flask, request
import cv2
import numpy as np
from gradio_canny2image import process

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def hello_world():
    photo = request.files.get("photo").read()
    photo = cv2.imdecode(np.fromstring(photo, np.uint8), cv2.IMREAD_COLOR)
    for y, line in enumerate(photo):
        for x, p in enumerate(line):
            photo[y][x] = [p[2], p[1], p[0]]

    data = {
        "input_image":          photo,
        "prompt":               request.form.get("prompt"),
        "a_prompt":             request.form.get("added_prompt", "best quality, extremely detailed"),
        "n_prompt":             request.form.get("added_prompt", "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"),
        "num_samples":          int(request.form.get("images", 1)),
        "image_resolution":     int(request.form.get("image_resolution", 512)),
        "ddim_steps":           int(request.form.get("steps", 20)),
        "guess_mode":           bool(request.form.get("guess_mode", False)),
        "strength":             int(request.form.get("control_strength", 1)),
        "scale":                int(request.form.get("guidance_scale", 9)),
        "seed":                 int(request.form.get("seed", -1)),
        "eta":                  float(request.form.get("eta", 0.0)),
        "low_threshold":        int(request.form.get("low_threshold", 100)),
        "high_threshold":       int(request.form.get("high_threshold", 200)),
    }
    results = process(**data)

    images = []
    for result in results:
        for y, line in enumerate(result):
            for x, p in enumerate(line):
                result[y][x] = [p[2], p[1], p[0]]
        im, buffer = cv2.imencode('.png', result)
        bimage = base64.b64encode(buffer)
        images.append(bimage.decode())
    return {"images": images}


if __name__ == "__main__":
    app.run(host='0.0.0.0')


# ssh root@xxxx
# sudo su control
# conda activate control
# /home/control/apps/ControlNet-main
# gunicorn --bind 0.0.0.0:7860 -w 1 app:app

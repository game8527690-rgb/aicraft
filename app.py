import os
import requests
import base64
import time
import cloudinary
import cloudinary.uploader
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, redirect, url_for, abort
)
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user
)
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from models import db, User, Image

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-this-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///aicraft.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

# Fix Render's postgres:// -> postgresql://
if app.config["SQLALCHEMY_DATABASE_URI"].startswith("postgres://"):
    app.config["SQLALCHEMY_DATABASE_URI"] = app.config["SQLALCHEMY_DATABASE_URI"].replace("postgres://", "postgresql://", 1)

db.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to generate images."

# Cloudinary config
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

HF_TOKEN = os.getenv("HF_API_KEY")
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")
DAILY_LIMIT = 20

# Local output folder (fallback if Cloudinary not configured)
os.makedirs(os.path.join(os.path.dirname(__file__), "static", "output"), exist_ok=True)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ── Helpers ────────────────────────────────────────────────────────────────────

def upload_image(image_bytes: bytes, user_id: int) -> tuple:
    """Upload image to Cloudinary. Returns (url, public_id)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    public_id = f"aicraft/u{user_id}_{timestamp}"

    result = cloudinary.uploader.upload(
        image_bytes,
        public_id=public_id,
        resource_type="image",
        format="png",
        overwrite=False,
    )
    return result["secure_url"], result["public_id"]


def upload_image_from_url(image_url: str, user_id: int) -> tuple:
    """Download from URL then upload to Cloudinary."""
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    return upload_image(response.content, user_id)


def replicate_run(model: str, input_data: dict) -> str:
    headers = {
        "Authorization": f"Bearer {REPLICATE_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }
    resp = requests.post(
        f"https://api.replicate.com/v1/models/{model}/predictions",
        headers=headers,
        json={"input": input_data},
        timeout=120,
    )
    if resp.status_code not in (200, 201):
        raise Exception(f"Replicate error {resp.status_code}: {resp.text}")

    prediction = resp.json()
    while prediction.get("status") not in ("succeeded", "failed", "canceled"):
        time.sleep(3)
        poll = requests.get(
            prediction["urls"]["get"],
            headers={"Authorization": f"Bearer {REPLICATE_TOKEN}"},
            timeout=120,
        )
        prediction = poll.json()

    if prediction.get("status") != "succeeded":
        raise Exception(f"Prediction failed: {prediction.get('error')}")

    output = prediction.get("output")
    return output[0] if isinstance(output, list) else output


ASPECT_RATIOS = {
    "square":    (1024, 1024),
    "landscape": (1024, 576),
    "portrait":  (576, 1024),
}


# ── Auth ───────────────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        errors = []
        if not username or len(username) < 3:
            errors.append("Username must be at least 3 characters.")
        if not email or "@" not in email:
            errors.append("A valid email is required.")
        if not password or len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        if password != confirm:
            errors.append("Passwords do not match.")
        if User.query.filter_by(username=username).first():
            errors.append("Username already taken.")
        if User.query.filter_by(email=email).first():
            errors.append("Email already registered.")

        if errors:
            return render_template("register.html", errors=errors, username=username, email=email)

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for("index"))

    return render_template("register.html", errors=[], username="", email="")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = bool(request.form.get("remember"))

        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            return redirect(request.args.get("next") or url_for("index"))

        return render_template("login.html", error="Invalid email or password.", email=email)

    return render_template("login.html", error=None, email="")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ── Main ───────────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def index():
    remaining = current_user.remaining_generations(DAILY_LIMIT)
    return render_template("index.html", user=current_user, remaining=remaining, daily_limit=DAILY_LIMIT)


@app.route("/generate", methods=["POST"])
@login_required
def generate():
    if not current_user.can_generate(DAILY_LIMIT):
        return jsonify({"error": f"Daily limit of {DAILY_LIMIT} generations reached. Resets at midnight."}), 429

    data = request.json
    prompt = data.get("prompt", "").strip()
    aspect = data.get("aspect_ratio", "square")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    width, height = ASPECT_RATIOS.get(aspect, (1024, 1024))

    hf_url = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "width": width,
            "height": height,
        },
    }

    try:
        response = requests.post(hf_url, headers=headers, json=payload, timeout=120)
        if response.status_code == 503:
            time.sleep(20)
            response = requests.post(hf_url, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            return jsonify({"error": f"API error {response.status_code}: {response.text}"}), 500

        image_url, public_id = upload_image(response.content, current_user.id)
        image_b64 = base64.b64encode(response.content).decode("utf-8")

        img = Image(user_id=current_user.id, filename=public_id, prompt=prompt,
                    type="generate", image_url=image_url)
        db.session.add(img)
        current_user.increment_count()
        db.session.commit()

        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{image_b64}",
            "filename": public_id,
            "path": image_url,
            "share_url": url_for("share", image_id=img.id, _external=True),
            "remaining": current_user.remaining_generations(DAILY_LIMIT),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/edit", methods=["POST"])
@login_required
def edit():
    if not current_user.can_generate(DAILY_LIMIT):
        return jsonify({"error": f"Daily limit of {DAILY_LIMIT} generations reached. Resets at midnight."}), 429

    prompt = request.form.get("prompt", "").strip()
    image_file = request.files.get("image")

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    if not image_file:
        return jsonify({"error": "Image is required"}), 400

    try:
        image_bytes = image_file.read()
        mime = image_file.content_type or "image/png"
        image_data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

        output_url = replicate_run(
            "black-forest-labs/flux-kontext-pro",
            {"prompt": prompt, "input_image": image_data_url, "output_format": "png", "safety_tolerance": 5},
        )

        image_url, public_id = upload_image_from_url(output_url, current_user.id)

        # Read back for base64
        img_bytes = requests.get(image_url, timeout=30).content
        image_b64 = base64.b64encode(img_bytes).decode("utf-8")

        img = Image(user_id=current_user.id, filename=public_id, prompt=prompt,
                    type="edit", image_url=image_url)
        db.session.add(img)
        current_user.increment_count()
        db.session.commit()

        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{image_b64}",
            "filename": public_id,
            "path": image_url,
            "share_url": url_for("share", image_id=img.id, _external=True),
            "remaining": current_user.remaining_generations(DAILY_LIMIT),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/gallery")
@login_required
def gallery():
    images = (
        Image.query.filter_by(user_id=current_user.id)
        .order_by(Image.created_at.desc())
        .limit(50).all()
    )
    return jsonify([{
        "filename": img.filename,
        "prompt": img.prompt,
        "type": img.type,
        "created_at": img.created_at.isoformat(),
        "path": img.image_url,
        "share_url": url_for("share", image_id=img.id, _external=True),
    } for img in images])


@app.route("/share/<int:image_id>")
def share(image_id):
    img = db.session.get(Image, image_id)
    if not img:
        abort(404)
    return render_template(
        "share.html",
        prompt=img.prompt,
        image_url=img.image_url,
        share_url=url_for("share", image_id=img.id, _external=True),
        created_at=img.created_at.strftime("%B %d, %Y"),
        image_type=img.type,
    )


@app.route("/static/output/<filename>")
def serve_image(filename):
    filename = os.path.basename(filename)
    return send_from_directory(
        os.path.join(os.path.dirname(__file__), "static", "output"), filename)


@app.route("/google6e5d17f5072374ec.html")
def google_verify():
    return render_template("google6e5d17f5072374ec (1).html")
def file_too_large(e):
    return jsonify({"error": "File too large. Maximum size is 10MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=False, port=5000)

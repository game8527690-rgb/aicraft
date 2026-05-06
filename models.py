from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    daily_count = db.Column(db.Integer, default=0)
    last_reset = db.Column(db.Date, default=date.today)

    images = db.relationship("Image", backref="user", lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def reset_daily_count_if_needed(self):
        today = date.today()
        if self.last_reset != today:
            self.daily_count = 0
            self.last_reset = today
            db.session.commit()

    def remaining_generations(self, limit=20):
        self.reset_daily_count_if_needed()
        return max(0, limit - self.daily_count)

    def can_generate(self, limit=20):
        self.reset_daily_count_if_needed()
        return self.daily_count < limit

    def increment_count(self):
        self.reset_daily_count_if_needed()
        self.daily_count += 1
        db.session.commit()


class Image(db.Model):
    __tablename__ = "images"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    filename = db.Column(db.String(256), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(20), nullable=False, default="generate")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

"""Flask application entry point for the NBA Predictor web dashboard."""

from flask import Flask

from config import BUNDLE_DIR, FLASK_HOST, FLASK_PORT
from web.routes import bp


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BUNDLE_DIR / "vendor" / "frontend"),
    )
    app.register_blueprint(bp)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)

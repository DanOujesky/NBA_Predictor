"""Flask application entry point for the NBA Predictor web dashboard."""

from flask import Flask

from config import FLASK_HOST, FLASK_PORT
from web.routes import bp


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder="web/templates",
        static_folder="web/static",
    )
    app.register_blueprint(bp)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)

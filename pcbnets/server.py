"""Minimal Flask server for the interactive viewer."""

from __future__ import annotations

import pathlib

from flask import Flask, send_from_directory


def make_app(build_dir: pathlib.Path) -> Flask:
    build_dir = pathlib.Path(build_dir).resolve()
    static_dir = pathlib.Path(__file__).parent / 'static'

    if not (build_dir / 'meta.json').exists() or not (build_dir / 'netmap.svg').exists():
        raise FileNotFoundError(
            f"build directory missing SVG viewer artifacts: {build_dir}\n"
            f"Run `pcbnets render` first."
        )

    app = Flask(__name__)

    @app.route('/')
    def index():
        return send_from_directory(static_dir, 'index.html')

    @app.route('/<path:name>')
    def asset(name: str):
        # Static viewer assets win first, build artefacts second. This means
        # a user-provided meta.json in build_dir overrides any default.
        if (static_dir / name).is_file():
            return send_from_directory(static_dir, name)
        return send_from_directory(build_dir, name)

    return app


def serve(build_dir: pathlib.Path,
          host: str = '127.0.0.1',
          port: int = 8000) -> None:
    app = make_app(build_dir)
    print(f"pcbnets viewer: http://{host}:{port}")
    app.run(host=host, port=port)

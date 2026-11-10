#!/usr/bin/env python3
"""Serve the generated map locally and load its Google Maps key from .env."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
HTML_PATH = HERE / "output" / "location_map.html"
AUDIT_PATH = HERE / "output" / "data_audit.html"
ENV_PATH = HERE / ".env"
HOST = "127.0.0.1"
PORT = 8765


def read_api_key():
    if not ENV_PATH.exists():
        raise RuntimeError(f"Missing {ENV_PATH}; add GOOGLE_MAPS_API_KEY=... there.")
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "GOOGLE_MAPS_API_KEY":
            value = value.strip().strip("\"'")
            if value:
                return value
    raise RuntimeError("GOOGLE_MAPS_API_KEY is missing or empty in test_model/.env")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route in ("/audit", "/audit/"):
            file_path = AUDIT_PATH
            inject_maps = False
        elif route in ("/", "/index.html"):
            file_path = HTML_PATH
            inject_maps = True
        else:
            self.send_error(404)
            return
        if not file_path.exists():
            self.send_error(404, "Requested page not generated yet")
            return
        page = file_path.read_text(encoding="utf-8")
        if inject_maps:
            key = quote(read_api_key(), safe="")
            loader = f'<script async src="https://maps.googleapis.com/maps/api/js?key={key}&callback=initMap&loading=async"></script>'
            page = page.replace("</body>", loader + "</body>")
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Avoid logging request query strings or rendered page contents.
        print(f"{self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    read_api_key()  # Fail early if setup is incomplete.
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Map server running at http://localhost:{PORT}/ (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMap server stopped.")
    finally:
        server.server_close()

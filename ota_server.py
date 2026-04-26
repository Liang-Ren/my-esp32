#!/usr/bin/env python3
"""Minimal OTA server for Xiaozhi ESP32. Returns custom websocket config."""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os
from dotenv import load_dotenv

load_dotenv(r"C:\Users\liang\OneDrive\文档\myPython\xiaozhi.me\backend-example\.env")

MCP_ENDPOINT_URL = os.getenv("MCP_ENDPOINT_URL", "")
MCP_TOKEN = os.getenv("MCP_TOKEN", "")

RESPONSE = json.dumps({
    "websocket": {
        "url": MCP_ENDPOINT_URL,
        "token": MCP_TOKEN
    }
}).encode()

class OTAHandler(BaseHTTPRequestHandler):
    def do_GET(self): self._respond()
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if length: self.rfile.read(length)
        self._respond()

    def _respond(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(RESPONSE)))
        self.end_headers()
        self.wfile.write(RESPONSE)

    def log_message(self, fmt, *args):
        print(f"[OTA] {self.address_string()} - {fmt % args}")

if __name__ == "__main__":
    print(f"OTA server starting on 0.0.0.0:8000")
    print(f"Websocket URL: {MCP_ENDPOINT_URL}")
    HTTPServer(("0.0.0.0", 8000), OTAHandler).serve_forever()
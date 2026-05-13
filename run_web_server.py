
import http.server
import socketserver
import webbrowser
import os

PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler

# Enable CORS and standard headers
class CORSRequestHandler(Handler):
    def end_headers(self):
        # Enable Cross-Origin Isolation for high-performance WASM multi-threading
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
 
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def do_GET(self):
        # Silently handle favicon.ico to prevent console noise
        if self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
            return
        return super().do_GET()

def run_server():
    print(f"🚀 Launching Schrödinger Bridge Web Server on port {PORT}...")
    print(f"🔗 URL: http://localhost:{PORT}/onnx_generate_image.html")
    print(f"🔗 URL: http://localhost:{PORT}/onnx_inference_tester.html")

    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        # Auto-open browser
        webbrowser.open(f"http://localhost:{PORT}/onnx_generate_image.html")
        webbrowser.open(f"http://localhost:{PORT}/onnx_inference_tester.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")

if __name__ == "__main__":
    run_server()

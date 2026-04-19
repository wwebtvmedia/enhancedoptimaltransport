
import http.server
import socketserver
import webbrowser
import os

PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler

# Enable CORS and standard headers
class CORSRequestHandler(Handler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

def run_server():
    print(f"🚀 Launching Schrödinger Bridge Web Server on port {PORT}...")
    print(f"🔗 URL: http://localhost:{PORT}/onnx_generate_image.html")
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        # Auto-open browser
        webbrowser.open(f"http://localhost:{PORT}/onnx_generate_image.html")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")

if __name__ == "__main__":
    run_server()

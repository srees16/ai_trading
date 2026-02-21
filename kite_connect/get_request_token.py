"""
Automates the Kite Connect login flow to capture the request token.

How it works:
1. Starts a local HTTP server on http://127.0.0.1:5000
2. Opens the Kite Connect login URL in your browser
3. You log in with your Zerodha credentials (+ TOTP/2FA)
4. Zerodha redirects back to the local server with the request_token
5. The script captures it, updates kite_app.py, and returns the token

IMPORTANT: Set your Kite Connect app's redirect URL to:
    http://127.0.0.1:5000
    (Go to https://developers.kite.trade -> Your App -> Redirect URL)
"""

import webbrowser
import subprocess
import tempfile
import re
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

API_KEY = 'hzcjwdgbs8wpon7p'
LOGIN_URL = f'https://kite.zerodha.com/connect/login?api_key={API_KEY}'
KITE_APP_FILE = os.path.join(os.path.dirname(__file__), 'kite_app.py')

captured_token = None


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global captured_token
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if 'request_token' in query:
            captured_token = query['request_token'][0]
            status = query.get('status', ['unknown'])[0]

            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = f"""
            <html><body style="font-family:Arial; text-align:center; margin-top:80px;">
                <h2>Request Token Captured!</h2>
                <p><b>Token:</b> <code>{captured_token}</code></p>
                <p style="color:green;">You can close this tab now.</p>
            </body></html>
            """
            self.wfile.write(html.encode())
        else:
            self.send_response(400)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Error: No request_token found.</h2></body></html>")

    def log_message(self, format, *args):
        # Suppress default request logging
        pass


def update_kite_app(token):
    """Update the request_token value in kite_app.py."""
    try:
        with open(KITE_APP_FILE, 'r') as f:
            content = f.read()

        updated = re.sub(
            r"(request_token\s*=\s*')[^']*(')",
            rf"\g<1>{token}\g<2>",
            content
        )

        with open(KITE_APP_FILE, 'w') as f:
            f.write(updated)

        print(f"  [OK] Updated request_token in kite_app.py")
    except Exception as e:
        print(f"  [ERROR] Could not update kite_app.py: {e}")


def fetch_request_token():
    """
    Launch the Kite login flow, capture the request_token via local
    HTTP redirect, update kite_app.py, and return the new token.

    Can be called from other modules:
        from get_request_token import fetch_request_token
        token = fetch_request_token()
    """
    global captured_token
    captured_token = None  # reset for re-entry

    server_address = ('127.0.0.1', 5000)
    httpd = HTTPServer(server_address, CallbackHandler)

    print("=" * 60)
    print("  Kite Connect - Request Token Generator")
    print("=" * 60)
    print(f"\n  Local callback server started on http://127.0.0.1:5000")
    print(f"  Opening Kite login page in your browser...\n")

    # Open browser as an isolated subprocess so we can kill it after capturing the token.
    # Using a temp --user-data-dir ensures the browser runs as its own process
    # instead of delegating to an existing instance and exiting.
    browser_proc = None
    temp_profile = None
    if sys.platform == 'win32':
        local = os.environ.get('LOCALAPPDATA', '')
        program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
        program_files_x86 = os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')

        browser_paths = [
            os.path.join(local, r'BraveSoftware\Brave-Browser\Application\brave.exe'),
            os.path.join(program_files, r'BraveSoftware\Brave-Browser\Application\brave.exe'),
            os.path.join(program_files, r'Google\Chrome\Application\chrome.exe'),
            os.path.join(program_files_x86, r'Google\Chrome\Application\chrome.exe'),
            os.path.join(local, r'Google\Chrome\Application\chrome.exe'),
            os.path.join(program_files, r'Microsoft\Edge\Application\msedge.exe'),
            os.path.join(program_files_x86, r'Microsoft\Edge\Application\msedge.exe'),
            os.path.join(program_files, r'Mozilla Firefox\firefox.exe'),
            os.path.join(program_files_x86, r'Mozilla Firefox\firefox.exe'),
        ]

        for browser_path in browser_paths:
            if os.path.isfile(browser_path):
                try:
                    temp_profile = tempfile.mkdtemp(prefix='kite_login_')
                    browser_proc = subprocess.Popen(
                        [browser_path, f'--user-data-dir={temp_profile}',
                         '--no-first-run', '--no-default-browser-check',
                         LOGIN_URL]
                    )
                    print(f"  Using: {os.path.basename(browser_path)}")
                    break
                except Exception:
                    continue
    if browser_proc is None:
        webbrowser.open(LOGIN_URL)

    print("  Waiting for login redirect... (Ctrl+C to cancel)\n")

    while captured_token is None:
        httpd.handle_request()

    httpd.server_close()

    print("=" * 60)
    print(f"  Request Token: {captured_token}")
    print("=" * 60)

    update_kite_app(captured_token)

    # Close the browser window that was opened for login
    if browser_proc is not None:
        try:
            time.sleep(1)  # brief pause so user sees the success page
            browser_proc.kill()  # kill the isolated browser instance
            browser_proc.wait(timeout=5)
            print("  [OK] Browser window closed.")
        except Exception:
            pass
        # Clean up temp profile directory
        if temp_profile and os.path.isdir(temp_profile):
            try:
                import shutil
                shutil.rmtree(temp_profile, ignore_errors=True)
            except Exception:
                pass

    print("\n  Done!")

    return captured_token


if __name__ == '__main__':
    fetch_request_token()

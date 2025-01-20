import logging
import time
from pywinauto import Application
from pywinauto.keyboard import send_keys
import re
import os
from winreg import OpenKey, QueryValue, HKEY_CURRENT_USER, KEY_READ
import argparse

class WhatsAppAutomation:
    def __init__(self, app_path=None):
        self.app_path = app_path or self._find_whatsapp_path()
        self.window = None
        self.app = None

    def _find_whatsapp_path(self):
        """Try to automatically detect WhatsApp installation path."""
        env_path = os.getenv("WHATSAPP_PATH")
        if env_path and os.path.exists(env_path):
            logging.info(f"Found WhatsApp via environment variable at: {env_path}")
            return env_path

        common_paths = [
            os.path.expandvars("%LOCALAPPDATA%\\WhatsApp\\WhatsApp.exe"),
            os.path.expandvars("%PROGRAMFILES%\\WhatsApp\\WhatsApp.exe"),
            os.path.expandvars("%PROGRAMFILES(X86)%\\WhatsApp\\WhatsApp.exe"),
            os.path.expandvars("%APPDATA%\\WhatsApp\\WhatsApp.exe"),
            os.path.expandvars("%USERPROFILE%\\AppData\\Local\\WhatsApp\\WhatsApp.exe"),
            os.path.expandvars("%USERPROFILE%\\AppData\\Roaming\\WhatsApp\\WhatsApp.exe"),
        ]

        try:
            with OpenKey(HKEY_CURRENT_USER, r"Software\\WhatsApp", 0, KEY_READ) as key:
                install_path = QueryValue(key, "InstallPath")
                if install_path:
                    common_paths.insert(0, os.path.join(install_path, "WhatsApp.exe"))
        except Exception:
            pass

        for path in common_paths:
            if os.path.exists(path):
                logging.info(f"Found WhatsApp at: {path}")
                return path

        raise FileNotFoundError("Could not find WhatsApp installation. Ensure WhatsApp is installed or provide the path.")

    def start_app(self):
        """Start WhatsApp desktop application."""
        logging.info("Starting WhatsApp desktop app...")
        try:
            if self.app:
                logging.warning("WhatsApp is already running.")
                return

            try:
                self.app = Application(backend="uia").connect(title_re="WhatsApp", timeout=5)
                logging.info("Connected to existing WhatsApp instance.")
            except Exception:
                if "WindowsApps" in self.app_path:
                    import subprocess
                    subprocess.Popen(self.app_path, shell=True)
                    time.sleep(10)
                    self.app = Application(backend="uia").connect(title_re="WhatsApp", timeout=60)
                else:
                    self.app = Application(backend="uia").start(self.app_path, timeout=30)

            self.window = self.app.window(title="WhatsApp", control_type="Window")
            self.window.wait("ready", timeout=60)
            logging.info("Successfully connected to WhatsApp desktop app.")
        except Exception as e:
            logging.error(f"Failed to start WhatsApp: {e}")
            raise

    def normalize_text(self, text):
        """Normalize text by removing extra spaces, emojis, and converting to lowercase."""
        return re.sub(r"[^\w\s]", "", text).strip().lower()

    def find_group(self, group_name):
        """Search for and select a specific group."""
        max_retries = 3
        for retry_count in range(max_retries):
            try:
                logging.info(f"Attempt {retry_count + 1} to find group '{group_name}'...")
                search_boxes = self.window.child_window(control_type="Edit", top_level_only=False).children()

                for index, search_box in enumerate(search_boxes):
                    try:
                        logging.info(f"Trying search box {index + 1}...")
                        search_box.click_input()
                        send_keys("^a{BACKSPACE}")
                        time.sleep(1)
                        send_keys(group_name)
                        time.sleep(3)

                        group_list = self.window.child_window(control_type="List").children()
                        for group in group_list:
                            group_text = self.normalize_text(group.window_text())
                            logging.info(f"Found group: {group_text}")
                            if group_name.lower() in group_text:
                                logging.info(f"Group '{group_name}' found. Selecting...")
                                group.click_input()
                                return True
                    except Exception as e:
                        logging.warning(f"Error using search box {index + 1}: {e}")
            except Exception as e:
                logging.warning(f"Attempt {retry_count + 1} failed: {e}")
                time.sleep(2)

        logging.error(f"Could not find group '{group_name}' after multiple attempts.")
        return False

    def send_message(self, message):
        """Send a message to the currently selected group."""
        try:
            message_box = self.window.child_window(control_type="Edit").wait("exists enabled visible", timeout=10)
            logging.info("Message box found. Sending message...")
            message_box.click_input()
            send_keys(message)
            send_keys("{ENTER}")
            logging.info("Message sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send message: {e}")

    def close_app(self):
        """Close the WhatsApp desktop app."""
        if self.app:
            logging.info("Closing WhatsApp desktop app...")
            try:
                self.app.kill()
                logging.info("WhatsApp desktop app closed.")
            except Exception as e:
                logging.error(f"Failed to close WhatsApp: {e}")
        else:
            logging.info("WhatsApp app was not started or is already closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate WhatsApp messaging.")
    parser.add_argument("--group", required=True, help="Name of the group to send a message to.")
    parser.add_argument("--message", required=True, help="Message to send to the group.")
    parser.add_argument("--path", default=None, help="Path to WhatsApp.exe (optional).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("whatsapp_automation.log"), logging.StreamHandler()],
    )

    automation = WhatsAppAutomation(app_path=args.path)

    try:
        automation.start_app()
        if automation.find_group(args.group):
            automation.send_message(args.message)
        else:
            logging.error("Failed to locate the group. Exiting script.")
    finally:
        automation.close_app()
        
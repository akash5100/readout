#!/Users/akash/.virtualenvs/tts/bin/python
"""System-Wide Floating TTS & Menu Bar Control for macOS (geohot style).

Features:
  1. Top Menu Bar with minimal 4-bar soundwave template icon:
     - Enable / Disable Selection Overlay (sleeps with 0% CPU when disabled)
     - Model Selection Submenu:
         ✓ Pocket TTS (100M)
           SupraTTS (1M)
           Chatterbox Turbo (0.3B)
           Qwen3-TTS (0.6B)
     - Voice Selection Submenu:
         ✓ alba
     - Play / Pause (active only when applicable)
     - Quit
  2. Floating Capsule Button:
     - Sits beside highlighted text anywhere on macOS.
     - Pure monochrome frosted dark glass with wide, unsquished waveform.
     - Turns into pause bars while speaking.
"""

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import AppKit
import Foundation
import objc
import Quartz
import soundfile as sf
from pynput import mouse
from PyObjCTools import AppHelper


try:
  import setproctitle

  setproctitle.setproctitle("tts-companion")
except Exception:
  pass

DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DIR, "..", ".."))
AUDIO_PATH = "/tmp/tts_macos.wav"
SERVER_BASE = "http://127.0.0.1:5050"
SERVER_URL = f"{SERVER_BASE}/tts"
ICON_PATH = os.path.join(REPO_ROOT, "assets", "menubar_icon.png")
VOICES_DIR = os.path.join(REPO_ROOT, "voices")
PID_FILE = "/tmp/tts_macos_app.pid"

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from tts.text_processor import (  # noqa: E402
  get_model_config,
  normalize_text_for_speech,
  prepare_speech_chunks,
)


class WaveformButtonView(AppKit.NSView):
  PILL_RECT = AppKit.NSMakeRect(10, 10, 42, 28)

  def initWithFrame_(self, frame):
    self = objc.super(WaveformButtonView, self).initWithFrame_(frame)
    if self:
      self.hovered = False
      self.state = "idle"  # "idle" | "playing" | "paused"
      self.tracking_area = None
      self.on_click = None

      # Clean SF Symbols
      config_size = AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(13.0, AppKit.NSFontWeightMedium)

      img_wf = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_("waveform", None)
      self.img_waveform = img_wf.imageWithSymbolConfiguration_(config_size) if img_wf else None

      img_p = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_("pause.fill", None)
      self.img_pause = img_p.imageWithSymbolConfiguration_(config_size) if img_p else None

      img_pl = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_("play.fill", None)
      self.img_play = img_pl.imageWithSymbolConfiguration_(config_size) if img_pl else None

      self.icon_view = AppKit.NSImageView.alloc().initWithFrame_(self.PILL_RECT)
      self.icon_view.setImageScaling_(AppKit.NSImageScaleNone)
      self.icon_view.setImageAlignment_(AppKit.NSImageAlignCenter)
      self.icon_view.setImage_(self.img_waveform)
      self.addSubview_(self.icon_view)
    return self

  def drawRect_(self, dirtyRect):
    is_dark = False
    if self.effectiveAppearance():
      is_dark = "dark" in str(self.effectiveAppearance().name()).lower()

    shadow = AppKit.NSShadow.alloc().init()
    if is_dark:
      # dark: { bg: "#323234", icon: "#f2f2f7", shadow: "0 2px 5px rgba(0,0,0,0.05)" }
      bg_color = AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0x32 / 255.0, 0x32 / 255.0, 0x34 / 255.0, 1.0)
      icon_color = AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0xF2 / 255.0, 0xF2 / 255.0, 0xF7 / 255.0, 1.0)
      shadow.setShadowOffset_(AppKit.NSMakeSize(0, -2))
    else:
      # light: { bg: "#ebebeb", icon: "#1c1c1e", shadow: "0 1px 5px rgba(0,0,0,0.05)" }
      bg_color = AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0xEB / 255.0, 0xEB / 255.0, 0xEB / 255.0, 1.0)
      icon_color = AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0x1C / 255.0, 0x1C / 255.0, 0x1E / 255.0, 1.0)
      shadow.setShadowOffset_(AppKit.NSMakeSize(0, -1))

    shadow.setShadowBlurRadius_(5.0)
    shadow.setShadowColor_(AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0, 0, 0, 0.05))

    AppKit.NSGraphicsContext.saveGraphicsState()
    shadow.set()

    radius = 14.0  # Exact requested 14px radius
    path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(self.PILL_RECT, radius, radius)
    bg_color.setFill()
    path.fill()

    AppKit.NSGraphicsContext.restoreGraphicsState()
    self.icon_view.setContentTintColor_(icon_color)

  def acceptsFirstMouse_(self, event):
    return True

  def acceptsFirstResponder(self):
    return True

  def hitTest_(self, point):
    # Only capture clicks directly within the 42x28 pill, passing clicks through the outer shadow margin
    if Foundation.NSPointInRect(point, self.PILL_RECT):
      return self
    return None

  def updateTrackingAreas(self):
    if self.tracking_area:
      self.removeTrackingArea_(self.tracking_area)
    self.tracking_area = AppKit.NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
      self.PILL_RECT, AppKit.NSTrackingMouseEnteredAndExited | AppKit.NSTrackingActiveAlways, self, None
    )
    self.addTrackingArea_(self.tracking_area)

  def mouseEntered_(self, event):
    self.hovered = True
    self.setNeedsDisplay_(True)

  def mouseExited_(self, event):
    self.hovered = False
    self.setNeedsDisplay_(True)

  def mouseDown_(self, event):
    if self.on_click:
      self.on_click()

  def setState_(self, state):
    self.state = state
    if self.icon_view:
      img = self.img_pause if state == "playing" else (self.img_play if state == "paused" else self.img_waveform)
      self.icon_view.setImage_(img)
    self.setNeedsDisplay_(True)


class GlobalTTSApp(AppKit.NSObject):
  def init(self):
    self = objc.super(GlobalTTSApp, self).init()
    if not self:
      return None

    self.is_enabled = True
    self.is_paused = False
    self.proc = None
    self.selected_text = ""
    self.current_playing_text = ""
    self.is_showing = False

    self.current_model = "pocket-tts"
    self.selected_voice = "alba"

    self.server_proc = None
    self.cancel_event = threading.Event()
    self.session_id = 0
    self.down_x = 0
    self.down_y = 0
    self.down_time = 0
    self.last_up_time = 0

    # 1. Setup Menu Bar Item
    self.status_item = AppKit.NSStatusBar.systemStatusBar().statusItemWithLength_(AppKit.NSSquareStatusItemLength)
    if os.path.exists(ICON_PATH):
      img = AppKit.NSImage.alloc().initWithContentsOfFile_(ICON_PATH)
      img.setSize_(AppKit.NSMakeSize(18, 18))
      img.setTemplate_(True)
      self.status_item.button().setImage_(img)

    self.menu = AppKit.NSMenu.alloc().init()
    self.menu.setDelegate_(self)

    # Server Status item (inspected only when clicked, no continuous background polling)
    self.item_server = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Server: Checking...", None, "")
    self.item_server.setEnabled_(False)
    self.menu.addItem_(self.item_server)

    self.menu.addItem_(AppKit.NSMenuItem.separatorItem())

    # Enable / Disable toggle
    self.item_enable = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Enable Selection Overlay", "toggleEnable:", "")
    self.item_enable.setTarget_(self)
    self.item_enable.setState_(AppKit.NSControlStateValueOn)
    self.menu.addItem_(self.item_enable)

    self.menu.addItem_(AppKit.NSMenuItem.separatorItem())

    # Model Submenu
    self.item_model = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Model", None, "")
    self.menu_model = AppKit.NSMenu.alloc().init()

    self.sub_supra = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("SupraTTS (1M)", "selectModelSupra:", "")
    self.sub_supra.setTarget_(self)
    self.sub_supra.setState_(AppKit.NSControlStateValueOff)
    self.menu_model.addItem_(self.sub_supra)

    self.sub_pocket = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Pocket TTS (100M)", "selectModelPocket:", "")
    self.sub_pocket.setTarget_(self)
    self.sub_pocket.setState_(AppKit.NSControlStateValueOn)
    self.menu_model.addItem_(self.sub_pocket)

    self.sub_chatterbox = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Chatterbox Turbo (0.3B)", "selectModelChatterbox:", "")
    self.sub_chatterbox.setTarget_(self)
    self.sub_chatterbox.setState_(AppKit.NSControlStateValueOff)
    self.menu_model.addItem_(self.sub_chatterbox)

    self.sub_qwen = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Qwen3-TTS (0.6B)", "selectModelQwen:", "")
    self.sub_qwen.setTarget_(self)
    self.sub_qwen.setState_(AppKit.NSControlStateValueOff)
    self.menu_model.addItem_(self.sub_qwen)

    self.item_model.setSubmenu_(self.menu_model)
    self.menu.addItem_(self.item_model)

    # Voice Submenu
    self.item_voice = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Voice", None, "")
    self.menu_voice = AppKit.NSMenu.alloc().init()
    self.item_voice.setSubmenu_(self.menu_voice)
    self.item_voice.setEnabled_(True)  # enabled when Qwen3 is selected
    self.menu.addItem_(self.item_voice)

    self.menu.addItem_(AppKit.NSMenuItem.separatorItem())

    # Play / Resume (no shortcuts to prevent accidental triggers)
    self.item_play = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Play", "actionPlay:", "")
    self.item_play.setTarget_(self)
    self.item_play.setEnabled_(False)
    self.menu.addItem_(self.item_play)

    # Pause
    self.item_pause = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Pause", "actionPause:", "")
    self.item_pause.setTarget_(self)
    self.item_pause.setEnabled_(False)
    self.menu.addItem_(self.item_pause)

    self.menu.addItem_(AppKit.NSMenuItem.separatorItem())

    # Restart (auto restart server + app)
    item_restart = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Restart", "actionRestart:", "")
    item_restart.setTarget_(self)
    self.menu.addItem_(item_restart)

    # Quit
    item_quit = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Quit", "actionQuit:", "")
    item_quit.setTarget_(self)
    self.menu.addItem_(item_quit)

    self.status_item.setMenu_(self.menu)

    # 2. Setup Floating Panel (Non-Activating)
    self.panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
      AppKit.NSMakeRect(100, 100, 62, 48),
      AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel,
      AppKit.NSBackingStoreBuffered,
      False,
    )
    self.panel.setLevel_(AppKit.NSScreenSaverWindowLevel)
    self.panel.setBackgroundColor_(AppKit.NSColor.clearColor())
    self.panel.setOpaque_(False)
    self.panel.setHasShadow_(False)
    self.panel.setAcceptsMouseMovedEvents_(True)
    self.panel.setIgnoresMouseEvents_(False)

    self.btn = WaveformButtonView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 62, 48))
    self.btn.on_click = self.handle_button_click
    self.panel.contentView().addSubview_(self.btn)

    # Initial server status check
    threading.Thread(target=self.sync_server_state, daemon=True).start()

    # 3. Start Global Mouse Listener
    threading.Thread(target=self.start_mouse_listener, daemon=True).start()
    return self

  def check_server_status(self, timeout=0.3):
    try:
      req = urllib.request.Request(f"{SERVER_BASE}/")
      with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status == 200:
          data = json.loads(resp.read().decode())
          return True, data
    except Exception:
      pass
    return False, None

  def start_server_background(self):
    is_running, _ = self.check_server_status(timeout=0.3)
    if is_running:
      return True

    cmd = [sys.executable, os.path.join(DIR, "server.py"), "--host", "127.0.0.1", "--port", "5050"]
    try:
      log_file = open("/tmp/tts_server_daemon.log", "a")
      self.server_proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
      t0 = time.time()
      while time.time() - t0 < 12.0:
        time.sleep(0.3)
        running, _ = self.check_server_status(timeout=0.3)
        if running:
          if self.current_model in ("qwen3", "pocket-tts", "pocket_tts", "chatterbox", "supratts"):
            self.post_load_model(self.current_model)
          return True
    except Exception as e:
      print("[!] Failed to auto-start server:", e)
    return False

  def startServerFromMenu_(self, sender):
    self.item_server.setTitle_("Server: Starting...")
    self.item_server.setEnabled_(False)

    def bg():
      ok = self.start_server_background()

      def update_ui():
        if ok:
          self.item_server.setTitle_("Server: Online")
          self.item_server.setAction_(None)
          self.item_server.setTarget_(None)
          self.item_server.setEnabled_(False)
        else:
          self.item_server.setTitle_("Server: Offline (Click to Retry)")
          self.item_server.setTarget_(self)
          self.item_server.setAction_("startServerFromMenu:")
          self.item_server.setEnabled_(True)

      AppHelper.callAfter(update_ui)

    threading.Thread(target=bg, daemon=True).start()

  def sync_server_state(self):
    running, data = self.check_server_status(timeout=1.5)
    if running and data:
      m_type = data.get("model_type", "pocket_tts")
      self.current_model = m_type

  # --- Menu Delegate & Actions ---
  def menuWillOpen_(self, menu):
    # 1. Update Server Status on click (no background polling)
    is_running, sdata = self.check_server_status(timeout=0.25)
    if is_running and sdata:
      server_mtype = sdata.get("model_type", self.current_model)
      self.current_model = server_mtype
      m_raw = sdata.get("model_type") or self.current_model or "ready"
      if "pocket" in m_raw.lower():
        m_name = "Pocket TTS (100M)"
      elif "chatterbox" in m_raw.lower():
        m_name = "Chatterbox Turbo (0.3B)"
      elif "qwen" in m_raw.lower():
        m_name = "Qwen3-TTS (0.6B)"
      elif "supra" in m_raw.lower():
        m_name = "SupraTTS (1M)"
      else:
        m_name = m_raw.title()
      self.item_server.setTitle_(f"Server: Online ({m_name})")
      self.item_server.setAction_(None)
      self.item_server.setTarget_(None)
      self.item_server.setEnabled_(False)
    else:
      self.item_server.setTitle_("Server: Standby (Starts on Click)")
      self.item_server.setTarget_(self)
      self.item_server.setAction_("startServerFromMenu:")
      self.item_server.setEnabled_(True)

    self.item_enable.setState_(AppKit.NSControlStateValueOn if self.is_enabled else AppKit.NSControlStateValueOff)
    is_playing = self.proc is not None and self.proc.poll() is None
    self.item_pause.setEnabled_(is_playing and not self.is_paused)
    self.item_play.setEnabled_(is_playing and self.is_paused)

    # Update Model Checkmarks
    is_chatterbox = "chatterbox" in self.current_model.lower()
    is_qwen = self.current_model == "qwen3"
    is_pocket = self.current_model in ("pocket_tts", "pocket-tts", "pocket")
    is_supra = not (is_qwen or is_pocket or is_chatterbox)

    self.sub_supra.setState_(AppKit.NSControlStateValueOn if is_supra else AppKit.NSControlStateValueOff)
    self.sub_pocket.setState_(AppKit.NSControlStateValueOn if is_pocket else AppKit.NSControlStateValueOff)
    self.sub_chatterbox.setState_(AppKit.NSControlStateValueOn if is_chatterbox else AppKit.NSControlStateValueOff)
    self.sub_qwen.setState_(AppKit.NSControlStateValueOn if is_qwen else AppKit.NSControlStateValueOff)
    self.item_voice.setEnabled_(is_qwen or is_pocket or is_chatterbox)

    # Populate Voice Submenu
    self.refresh_voice_menu()

  def refresh_voice_menu(self):
    self.menu_voice.removeAllItems()
    voices = []
    if os.path.exists(VOICES_DIR):
      for f in sorted(os.listdir(VOICES_DIR)):
        if f.endswith((".wav", ".mp3", ".flac")) and not f.endswith(("_prompt.pt", "_pocket.safetensors")):
          stem = Path(f).stem
          if stem not in voices:
            voices.append(stem)

    if not voices:
      empty_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("No voices found", None, "")
      empty_item.setEnabled_(False)
      self.menu_voice.addItem_(empty_item)
      return

    for v in voices:
      item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(v, "selectVoice:", "")
      item.setTarget_(self)
      item.setRepresentedObject_(v)
      if v == self.selected_voice:
        item.setState_(AppKit.NSControlStateValueOn)
      else:
        item.setState_(AppKit.NSControlStateValueOff)
      self.menu_voice.addItem_(item)

  def selectModelSupra_(self, sender):
    self.current_model = "supratts"
    self.sub_supra.setState_(AppKit.NSControlStateValueOn)
    self.sub_pocket.setState_(AppKit.NSControlStateValueOff)
    self.sub_chatterbox.setState_(AppKit.NSControlStateValueOff)
    self.sub_qwen.setState_(AppKit.NSControlStateValueOff)
    self.item_voice.setEnabled_(False)
    threading.Thread(target=lambda: self.post_load_model("supratts"), daemon=True).start()

  def selectModelPocket_(self, sender):
    self.current_model = "pocket-tts"
    self.sub_supra.setState_(AppKit.NSControlStateValueOff)
    self.sub_pocket.setState_(AppKit.NSControlStateValueOn)
    self.sub_chatterbox.setState_(AppKit.NSControlStateValueOff)
    self.sub_qwen.setState_(AppKit.NSControlStateValueOff)
    self.item_voice.setEnabled_(True)
    threading.Thread(target=lambda: self.post_load_model("pocket-tts"), daemon=True).start()

  def selectModelChatterbox_(self, sender):
    self.current_model = "chatterbox"
    self.sub_supra.setState_(AppKit.NSControlStateValueOff)
    self.sub_pocket.setState_(AppKit.NSControlStateValueOff)
    self.sub_chatterbox.setState_(AppKit.NSControlStateValueOn)
    self.sub_qwen.setState_(AppKit.NSControlStateValueOff)
    self.item_voice.setEnabled_(True)
    threading.Thread(target=lambda: self.post_load_model("chatterbox"), daemon=True).start()

  def selectModelQwen_(self, sender):
    self.current_model = "qwen3"
    self.sub_supra.setState_(AppKit.NSControlStateValueOff)
    self.sub_pocket.setState_(AppKit.NSControlStateValueOff)
    self.sub_chatterbox.setState_(AppKit.NSControlStateValueOff)
    self.sub_qwen.setState_(AppKit.NSControlStateValueOn)
    self.item_voice.setEnabled_(True)
    threading.Thread(target=lambda: self.post_load_model("qwen3"), daemon=True).start()

  def selectVoice_(self, sender):
    v = sender.representedObject()
    if v:
      self.selected_voice = v
      self.refresh_voice_menu()
      threading.Thread(target=lambda: self.post_set_voice(v), daemon=True).start()

  def post_set_voice(self, voice_name):
    try:
      req = urllib.request.Request(
        f"{SERVER_BASE}/set_voice",
        data=json.dumps({"voice": voice_name}).encode(),
        headers={"Content-Type": "application/json"}
      )
      with urllib.request.urlopen(req, timeout=1.5):
        pass
    except Exception:
      pass

  def post_load_model(self, repo_alias):
    try:
      req = urllib.request.Request(
        f"{SERVER_BASE}/load_model", data=json.dumps({"repo_id": repo_alias}).encode(), headers={"Content-Type": "application/json"}
      )
      with urllib.request.urlopen(req, timeout=60):
        print(f"[*] Switched model to: {repo_alias}")
    except Exception as e:
      print(f"[!] Error switching model: {e}")

  def toggleEnable_(self, sender):
    self.is_enabled = not self.is_enabled
    if not self.is_enabled:
      self.hide_panel()
      self.stop_audio()
      # Immediately shut down server so total system memory drops to ~50MB instantly
      threading.Thread(target=self.stop_server_background, daemon=True).start()

  def stop_server_background(self):
    try:
      req = urllib.request.Request(f"{SERVER_BASE}/shutdown", data=b"{}", headers={"Content-Type": "application/json"})
      with urllib.request.urlopen(req, timeout=0.6):
        pass
    except Exception:
      pass
    if self.server_proc and self.server_proc.poll() is None:
      try:
        self.server_proc.terminate()
      except Exception:
        pass

  def actionPlay_(self, sender):
    if self.proc and self.proc.poll() is None and self.is_paused:
      try:
        os.kill(self.proc.pid, signal.SIGCONT)
        self.is_paused = False
        self.btn.setState_("playing")
      except Exception:
        pass

  def actionPause_(self, sender):
    if self.proc and self.proc.poll() is None and not self.is_paused:
      try:
        os.kill(self.proc.pid, signal.SIGSTOP)
        self.is_paused = True
        self.btn.setState_("paused")
      except Exception:
        pass

  def actionRestart_(self, sender):
    print("[*] User triggered full restart of server and app...")
    self.stop_audio()
    self.stop_server_background()

    if os.path.exists(PID_FILE):
      try:
        os.remove(PID_FILE)
      except Exception:
        pass

    python_bin = sys.executable
    app_script = os.path.abspath(__file__)
    server_script = os.path.join(DIR, "server.py")

    restart_cmd = (
      f"sleep 0.5; "
      f"pkill -9 -f 'tts-server' 2>/dev/null; "
      f"pkill -9 -f 'tts-companion' 2>/dev/null; "
      f"nohup '{python_bin}' '{server_script}' --port 5050 > /tmp/tts_server_daemon.log 2>&1 & "
      f"for i in $(seq 1 40); do curl -s http://127.0.0.1:5050/ >/dev/null && break; sleep 0.2; done; "
      f"nohup '{python_bin}' '{app_script}' > /tmp/tts_companion.log 2>&1 &"
    )
    subprocess.Popen(["/bin/zsh", "-c", restart_cmd], start_new_session=True, close_fds=True)
    AppKit.NSApplication.sharedApplication().terminate_(self)

  def actionQuit_(self, sender):
    self.stop_audio()
    if self.server_proc and self.server_proc.poll() is None:
      try:
        os.killpg(os.getpgid(self.server_proc.pid), signal.SIGTERM)
      except Exception:
        pass
    if os.path.exists(PID_FILE):
      try:
        os.remove(PID_FILE)
      except Exception:
        pass
    AppKit.NSApplication.sharedApplication().terminate_(self)

  # --- Selection & Floating Panel ---
  def simulate_cmd_c(self) -> str:
    pb = AppKit.NSPasteboard.generalPasteboard()
    old_count = pb.changeCount()

    c_down = Quartz.CGEventCreateKeyboardEvent(None, 8, True)
    Quartz.CGEventSetFlags(c_down, Quartz.kCGEventFlagMaskCommand)
    c_up = Quartz.CGEventCreateKeyboardEvent(None, 8, False)
    Quartz.CGEventSetFlags(c_up, Quartz.kCGEventFlagMaskCommand)

    Quartz.CGEventPost(Quartz.kCGHIDEventTap, c_down)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, c_up)

    t0 = time.time()
    while time.time() - t0 < 0.20:
      if pb.changeCount() != old_count:
        break
      time.sleep(0.01)

    # Only return newly copied text if changeCount updated
    if pb.changeCount() != old_count:
      new_text = pb.stringForType_(AppKit.NSPasteboardTypeString) or ""
      return new_text.strip()
    return ""

  def start_mouse_listener(self):
    import ApplicationServices

    # Request accessibility permissions dialog if untrusted
    opts = {ApplicationServices.kAXTrustedCheckOptionPrompt: True}
    ApplicationServices.AXIsProcessTrustedWithOptions(opts)

    def on_click(x, y, button, pressed):
      if not self.is_enabled:
        return

      now = time.time()
      if button == mouse.Button.left:
        if pressed:
          self.down_x = x
          self.down_y = y
          self.down_time = now

          loc = AppKit.NSEvent.mouseLocation()
          frame = self.panel.frame()
          if self.is_showing and not Foundation.NSPointInRect(loc, frame):
            AppHelper.callAfter(self.hide_panel)
        else:
          dx = x - self.down_x
          dy = y - self.down_y
          dist = (dx * dx + dy * dy) ** 0.5
          is_double = (now - self.last_up_time) < 0.40
          self.last_up_time = now

          if dist > 6 or is_double:
            time.sleep(0.06)
            text = self.simulate_cmd_c()
            if text and len(text) >= 2:
              norm_text = normalize_text_for_speech(text)
              if norm_text and norm_text != self.current_playing_text:
                self.stop_audio()
              self.selected_text = text
              loc = AppKit.NSEvent.mouseLocation()
              AppHelper.callAfter(self.show_panel, loc.x, loc.y)

    while True:
      try:
        with mouse.Listener(on_click=on_click) as listener:
          print("[✓] Mouse event listener successfully attached.")
          listener.join()
      except Exception as e:
        print(f"[!] Mouse listener exception: {e}")
      time.sleep(2.0)

  def show_panel(self, x, y):
    self.btn.setState_("idle")
    y_pos = (y - 34) if (y - 34 > 30) else (y + 12)
    # 62x48 panel with 10px margin around 42x28 pill keeps pill positioned at exactly (x + 6, y_pos)
    rect = AppKit.NSMakeRect(x + 6 - 10, y_pos - 10, 62, 48)
    self.panel.setFrame_display_(rect, True)
    self.panel.orderFrontRegardless()
    self.is_showing = True

  def hide_panel(self):
    if self.is_showing:
      self.panel.orderOut_(None)
      self.is_showing = False

  @staticmethod
  def chunk_text(text: str, model: str = "pocket-tts") -> list[str]:
    cfg = get_model_config(model)
    speech_chunks = prepare_speech_chunks(text, max_words=cfg["max_words"], strategy=cfg["strategy"])
    if speech_chunks:
      return [c.text for c in speech_chunks]
    norm = normalize_text_for_speech(text)
    return [norm] if norm else []

  def stop_audio(self):
    self.cancel_event.set()
    if self.proc:
      try:
        # If paused via SIGSTOP, SIGTERM won't wake it. SIGKILL unconditionally terminates.
        os.kill(self.proc.pid, signal.SIGKILL)
      except Exception:
        try:
          self.proc.kill()
        except Exception:
          pass
    self.proc = None
    self.is_paused = False
    self.current_playing_text = ""

  def handle_button_click(self):
    # Only toggle pause/resume if there is active/paused audio for the EXACT SAME selected text
    norm_selected = normalize_text_for_speech(self.selected_text)
    if self.proc and self.proc.poll() is None and norm_selected and norm_selected == self.current_playing_text:
      if self.is_paused:
        self.actionPlay_(None)
      else:
        self.actionPause_(None)
      return

    # New text selection or fresh playback: Stop previous audio immediately
    self.stop_audio()

    text = self.selected_text
    if not text or not text.strip():
      self.hide_panel()
      return

    self.current_playing_text = text
    self.btn.setState_("playing")
    self.cancel_event.clear()
    self.session_id += 1
    current_session = self.session_id

    cfg = get_model_config(self.current_model)
    # Model-aware chunking (multi-sentence batching for fast models, streaming for Qwen3)
    chunks = self.chunk_text(text, model=self.current_model)
    if not chunks:
      chunks = [text]

    # If model strategy specifies single-shot generation (e.g. SupraTTS), keep as unified generation
    if not cfg.get("chain_streaming", True):
      chunks = [text]

    supports_voice = self.current_model in ("qwen3", "pocket_tts", "pocket-tts", "pocket", "chatterbox")
    voice_to_use = self.selected_voice if supports_voice else None
    model_to_use = self.current_model

    def chain_worker():
      audio_queue = queue.Queue(maxsize=3)

      def producer():
        for idx, chunk in enumerate(chunks):
          if self.session_id != current_session or self.cancel_event.is_set():
            break

          running, _ = self.check_server_status(timeout=0.2)
          if not running:
            if not self.start_server_background():
              break

          payload = {
            "text": chunk,
            "model": model_to_use,
          }
          if supports_voice and voice_to_use:
            payload["voice"] = voice_to_use

          try:
            req = urllib.request.Request(SERVER_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            chunk_path = f"/tmp/tts_{current_session}_{idx}.wav"
            with urllib.request.urlopen(req, timeout=60) as resp:
              with open(chunk_path, "wb") as f:
                f.write(resp.read())

            if self.session_id != current_session or self.cancel_event.is_set():
              try:
                os.remove(chunk_path)
              except Exception:
                pass
              break

            audio_queue.put((idx, chunk_path))
          except Exception as e:
            print(f"[!] Chained generation error chunk {idx}: {e}")
            break

        audio_queue.put(None)

      # Start producer thread in background
      threading.Thread(target=producer, daemon=True).start()

      # Consumer: plays seamless audio stream across chunks
      play_bin = "/opt/homebrew/bin/play" if os.path.exists("/opt/homebrew/bin/play") else shutil.which("play")
      try:
        if play_bin:
          # Persistent single audio process: CoreAudio stream stays continuously active across all chunks!
          self.proc = subprocess.Popen(
            [play_bin, "-q", "-t", "raw", "-r", "24000", "-c", "1", "-b", "16", "-e", "signed-integer", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
          )
          while not self.cancel_event.is_set() and self.session_id == current_session:
            item = audio_queue.get()
            if item is None:
              break
            idx, chunk_path = item
            if self.session_id != current_session or self.cancel_event.is_set():
              try:
                os.remove(chunk_path)
              except Exception:
                pass
              break

            try:
              audio_data, sr = sf.read(chunk_path, dtype="int16")
              self.proc.stdin.write(audio_data.tobytes())
              self.proc.stdin.flush()
            except Exception as pe:
              print(f"[!] Chained audio pipe error chunk {idx}: {pe}")
            finally:
              try:
                os.remove(chunk_path)
              except Exception:
                pass

          if self.proc and self.proc.stdin:
            try:
              self.proc.stdin.close()
            except Exception:
              pass
          if self.proc:
            self.proc.wait()
        else:
          # Fallback to afplay per chunk with safe acoustic pre-roll padding
          while not self.cancel_event.is_set() and self.session_id == current_session:
            item = audio_queue.get()
            if item is None:
              break
            idx, chunk_path = item
            if self.session_id != current_session or self.cancel_event.is_set():
              try:
                os.remove(chunk_path)
              except Exception:
                pass
              break

            self.proc = subprocess.Popen(["/usr/bin/afplay", chunk_path])
            self.proc.wait()

            try:
              os.remove(chunk_path)
            except Exception:
              pass
            time.sleep(0.02)
      except Exception as e:
        print("[!] Chained playback error:", e)
      finally:
        if self.session_id == current_session:
          self.proc = None
          self.is_paused = False
          self.current_playing_text = ""
          AppHelper.callAfter(lambda: self.btn.setState_("idle"))
          AppHelper.callAfter(self.hide_panel)

    threading.Thread(target=chain_worker, daemon=True).start()


if __name__ == "__main__":
  if os.path.exists(PID_FILE):
    try:
      with open(PID_FILE) as f:
        old_pid = int(f.read().strip())
      os.kill(old_pid, 0)
      print(f"[*] TTS App is already running (PID: {old_pid}).")
      sys.exit(0)
    except (ProcessLookupError, ValueError):
      pass
    except PermissionError:
      print("[*] TTS App is already running.")
      sys.exit(0)

  try:
    with open(PID_FILE, "w") as f:
      f.write(str(os.getpid()))
  except Exception:
    pass

  app = AppKit.NSApplication.sharedApplication()
  app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
  global_app = GlobalTTSApp.alloc().init()
  AppKit.NSApplication.sharedApplication().run()

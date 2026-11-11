import AppKit
import ApplicationServices
import Foundation

/// Monitors mouse selection events across macOS using CoreGraphics CGEventTap.
/// Full parity with src/tts/app.py mouse listener:
/// - Only dismisses the floating pill when clicking OUTSIDE the panel frame.
/// - Clicks on the floating pill pass through directly to the pill view.
/// - When new text is highlighted, stops previous audio immediately and shows pill.
@MainActor
final class SelectionMonitor {
    static let shared = SelectionMonitor()

    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var downPoint: CGPoint = .zero
    private var downTime: TimeInterval = 0
    private var lastUpTime: TimeInterval = 0
    private var isRunning = false

    private init() {}

    func start() {
        guard !isRunning else { return }

        // Prompt for Accessibility if not already granted
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)

        let eventMask = (1 << CGEventType.leftMouseDown.rawValue) | (1 << CGEventType.leftMouseUp.rawValue)

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .listenOnly,
            eventsOfInterest: CGEventMask(eventMask),
            callback: { (proxy, type, event, refcon) -> Unmanaged<CGEvent>? in
                guard let refcon = refcon else { return Unmanaged.passUnretained(event) }
                let monitor = Unmanaged<SelectionMonitor>.fromOpaque(refcon).takeUnretainedValue()
                return monitor.handle(event: event, type: type)
            },
            userInfo: Unmanaged.passUnretained(self).toOpaque()
        ) else {
            print("[SelectionMonitor] Note: CGEventTap could not be created (Accessibility permission required).")
            return
        }

        self.eventTap = tap
        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        self.runLoopSource = source
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        self.isRunning = true
        print("[✓] Native SelectionMonitor CGEventTap active.")
    }

    func stop() {
        guard isRunning, let tap = eventTap else { return }
        CGEvent.tapEnable(tap: tap, enable: false)
        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes)
        }
        self.eventTap = nil
        self.runLoopSource = nil
        self.isRunning = false
    }

    private func handle(event: CGEvent, type: CGEventType) -> Unmanaged<CGEvent>? {
        // Auto-recover tap if temporarily disabled by macOS under event pressure
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            if let tap = self.eventTap {
                CGEvent.tapEnable(tap: tap, enable: true)
            }
            return Unmanaged.passUnretained(event)
        }

        guard AppState.shared.selectionOverlayEnabled else {
            return Unmanaged.passUnretained(event)
        }

        let loc = event.location
        let now = Date().timeIntervalSince1970
        let cocoaMouseLoc = NSEvent.mouseLocation

        // Playback controls are a separate fixed panel, not a new selection.
        if PlaybackIndicatorController.shared.contains(cocoaMouseLoc) {
            return Unmanaged.passUnretained(event)
        }

        if type == .leftMouseDown {
            self.downPoint = loc
            self.downTime = now

            // 1:1 Parity with app.py line 629:
            // if self.is_showing and not Foundation.NSPointInRect(loc, frame):
            //     AppHelper.callAfter(self.hide_panel)
            if FloatingCapsuleController.shared.isShowing {
                if !FloatingCapsuleController.shared.contains(cocoaMouseLoc) {
                    FloatingCapsuleController.shared.hide()
                }
            }
        } else if type == .leftMouseUp {
            // If the user clicked inside the floating capsule panel, let the click proceed to WaveformPillView
            if FloatingCapsuleController.shared.isShowing && FloatingCapsuleController.shared.contains(cocoaMouseLoc) {
                return Unmanaged.passUnretained(event)
            }

            let dx = loc.x - self.downPoint.x
            let dy = loc.y - self.downPoint.y
            let dist = sqrt(dx * dx + dy * dy)
            let isDouble = (now - self.lastUpTime) < 0.40
            self.lastUpTime = now

            // Significant mouse drag (> 10px) or double-click: check if text was selected
            if dist > 10.0 || isDouble {
                DispatchQueue.global(qos: .userInteractive).asyncAfter(deadline: .now() + 0.08) {
                    if let selected = self.getSelectedText(), selected.count >= 2 {
                        Task { @MainActor in
                            // 1:1 Parity with app.py line 645:
                            // Stop previous audio immediately when new text is selected
                            if selected != FloatingCapsuleController.shared.currentPlayingText {
                                AppState.shared.stop()
                            }
                            let currentLoc = NSEvent.mouseLocation
                            FloatingCapsuleController.shared.show(
                                at: currentLoc.x,
                                y: currentLoc.y,
                                text: selected
                            )
                        }
                    }
                }
            }
        }

        return Unmanaged.passUnretained(event)
    }

    /// Reads the selected text directly across native macOS applications using the Accessibility API.
    /// Completely non-intrusive:
    /// - 0 keystrokes simulated (Cmd+C, Cmd+X, and user shortcuts are never hijacked or interrupted)
    /// - 0 pasteboard modification (user's clipboard is never overwritten or polluted)
    /// - 0 modifier key corruption
    nonisolated private func getSelectedText() -> String? {
        guard let frontApp = NSWorkspace.shared.frontmostApplication else { return nil }
        let pid = frontApp.processIdentifier
        let axApp = AXUIElementCreateApplication(pid)

        // Ensure Chromium/Electron apps have enhanced accessibility tree active
        _ = AXUIElementSetAttributeValue(axApp, "AXEnhancedUserInterface" as CFString, kCFBooleanTrue)

        // 1. Try focused UI element
        var focusedElementRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(axApp, kAXFocusedUIElementAttribute as CFString, &focusedElementRef) == .success,
           let focused = focusedElementRef {
            if let text = extractText(from: focused as! AXUIElement) {
                return text
            }
        }

        // 2. Try focused window's focused UI element
        var focusedWindowRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(axApp, kAXFocusedWindowAttribute as CFString, &focusedWindowRef) == .success,
           let window = focusedWindowRef {
            var windowElemRef: CFTypeRef?
            if AXUIElementCopyAttributeValue(window as! AXUIElement, kAXFocusedUIElementAttribute as CFString, &windowElemRef) == .success,
               let windowElem = windowElemRef {
                if let text = extractText(from: windowElem as! AXUIElement) {
                    return text
                }
            }
            if let text = extractText(from: window as! AXUIElement) {
                return text
            }
        }

        // 3. System-wide focused element fallback
        let systemWide = AXUIElementCreateSystemWide()
        var sysElemRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(systemWide, kAXFocusedUIElementAttribute as CFString, &sysElemRef) == .success,
           let sysElem = sysElemRef {
            if let text = extractText(from: sysElem as! AXUIElement) {
                return text
            }
        }

        return nil
    }

    nonisolated private func extractText(from element: AXUIElement) -> String? {
        // 1. Direct selected text attribute (String or NSAttributedString)
        var selectedTextRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(element, kAXSelectedTextAttribute as CFString, &selectedTextRef) == .success,
           let val = selectedTextRef {
            if let str = val as? String {
                let trimmed = str.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { return trimmed }
            } else if let attrStr = val as? NSAttributedString {
                let trimmed = attrStr.string.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { return trimmed }
            }
        }

        // 2. Selected text range + string for range
        var selectedRangeRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, &selectedRangeRef) == .success,
           let rangeVal = selectedRangeRef {
            var stringRef: CFTypeRef?
            if AXUIElementCopyParameterizedAttributeValue(element, kAXStringForRangeParameterizedAttribute as CFString, rangeVal, &stringRef) == .success,
               let val = stringRef {
                if let str = val as? String {
                    let trimmed = str.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty { return trimmed }
                } else if let attrStr = val as? NSAttributedString {
                    let trimmed = attrStr.string.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty { return trimmed }
                }
            }
        }

        return nil
    }
}

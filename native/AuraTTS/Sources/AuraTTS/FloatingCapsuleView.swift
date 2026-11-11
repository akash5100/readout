import AppKit
import Foundation
import QuartzCore

/// Pure AppKit port of the original WaveformButtonView from src/tts/app.py:
/// Exactly 42x28 rounded pill (radius 14), clean monochrome SF Symbol,
/// 10px margin in a 62x48 borderless nonactivating panel.
/// Exact colors:
///   Dark mode:  bg #323234, icon #f2f2f7, shadow 0 2px 5px rgba(0,0,0,0.05)
///   Light mode: bg #ebebeb, icon #1c1c1e, shadow 0 1px 5px rgba(0,0,0,0.05)
final class WaveformPillView: NSView {
    static let pillRect = NSRect(x: 10, y: 10, width: 42, height: 28)

    enum State {
        case idle
        case playing
        case paused
    }

    var state: State = .idle {
        didSet {
            guard oldValue != state else { return }
            updateIcon(animated: true)
        }
    }

    var onClick: (() -> Void)?

    private var hovered = false
    private var trackingArea: NSTrackingArea?
    private let iconView = NSImageView()

    private let imgWaveform: NSImage?
    private let imgPause: NSImage?
    private let imgPlay: NSImage?

    override init(frame: NSRect) {
        let config = NSImage.SymbolConfiguration(pointSize: 13.0, weight: .medium)
        self.imgWaveform = NSImage(systemSymbolName: "waveform", accessibilityDescription: nil)?
            .withSymbolConfiguration(config)
        self.imgPause = NSImage(systemSymbolName: "pause.fill", accessibilityDescription: nil)?
            .withSymbolConfiguration(config)
        self.imgPlay = NSImage(systemSymbolName: "play.fill", accessibilityDescription: nil)?
            .withSymbolConfiguration(config)

        super.init(frame: frame)

        self.wantsLayer = true

        iconView.wantsLayer = true
        iconView.frame = Self.pillRect
        iconView.imageScaling = .scaleNone
        iconView.imageAlignment = .alignCenter
        iconView.image = imgWaveform
        addSubview(iconView)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func resetWithoutAnimation() {
        state = .idle
        updateIcon(animated: false)
    }

    private func image(for state: State) -> NSImage? {
        switch state {
        case .idle:
            return imgWaveform
        case .playing:
            return imgPause
        case .paused:
            return imgPlay
        }
    }

    private func updateIcon(animated: Bool) {
        guard let targetImage = image(for: state) else { return }

        if animated {
            if #available(macOS 15.0, *) {
                iconView.setSymbolImage(
                    targetImage,
                    contentTransition: .replace.magic(fallback: .downUp),
                    options: .default.speed(1.2)
                )
            } else {
                iconView.setSymbolImage(
                    targetImage,
                    contentTransition: .replace.byLayer,
                    options: .default.speed(1.2)
                )
            }

            if state == .playing {
                // Subtle live breathing pulse while speech is streaming
                iconView.addSymbolEffect(.pulse.byLayer, options: .repeating.speed(0.8))
            } else {
                iconView.removeAllSymbolEffects()
            }
        } else {
            iconView.removeAllSymbolEffects()
            iconView.image = targetImage
        }
        needsDisplay = true
    }

    /// Spring scale press feedback for native tactile feel
    /// The original scale feedback is now an opacity change to keep the button still.
    private func animateClickFeedback() {
        guard let layer = self.layer else { return }
        let feedback = CABasicAnimation(keyPath: "opacity")
        feedback.fromValue = 0.8
        feedback.toValue = 1.0
        feedback.duration = 0.12
        layer.add(feedback, forKey: "clickFeedback")
    }

    override func draw(_ dirtyRect: NSRect) {
        let isDark = effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua

        let bgColor: NSColor
        let iconColor: NSColor
        let shadow = NSShadow()

        if isDark {
            // Dark mode: bg #323234 with subtle hover highlight
            let r: CGFloat = hovered ? 0x3E / 255.0 : 0x32 / 255.0
            let g: CGFloat = hovered ? 0x3E / 255.0 : 0x32 / 255.0
            let b: CGFloat = hovered ? 0x42 / 255.0 : 0x34 / 255.0
            bgColor = NSColor(srgbRed: r, green: g, blue: b, alpha: 1.0)
            iconColor = NSColor(srgbRed: 0xF2 / 255.0, green: 0xF2 / 255.0, blue: 0xF7 / 255.0, alpha: 1.0)
            shadow.shadowOffset = NSSize(width: 0, height: -2)
        } else {
            // Light mode: bg #ebebeb with subtle hover highlight
            let r: CGFloat = hovered ? 0xE2 / 255.0 : 0xEB / 255.0
            let g: CGFloat = hovered ? 0xE2 / 255.0 : 0xEB / 255.0
            let b: CGFloat = hovered ? 0xE2 / 255.0 : 0xEB / 255.0
            bgColor = NSColor(srgbRed: r, green: g, blue: b, alpha: 1.0)
            iconColor = NSColor(srgbRed: 0x1C / 255.0, green: 0x1C / 255.0, blue: 0x1E / 255.0, alpha: 1.0)
            shadow.shadowOffset = NSSize(width: 0, height: -1)
        }

        shadow.shadowBlurRadius = 5.0
        shadow.shadowColor = NSColor(srgbRed: 0, green: 0, blue: 0, alpha: 0.05)

        NSGraphicsContext.saveGraphicsState()
        shadow.set()

        let radius: CGFloat = 14.0  // Exact requested 14px radius
        let path = NSBezierPath(roundedRect: Self.pillRect, xRadius: radius, yRadius: radius)
        bgColor.setFill()
        path.fill()

        NSGraphicsContext.restoreGraphicsState()
        iconView.contentTintColor = iconColor
    }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool {
        true
    }

    override var acceptsFirstResponder: Bool {
        true
    }

    override func hitTest(_ point: NSPoint) -> NSView? {
        // Only capture clicks directly within the 42x28 pill, passing clicks through the outer shadow margin
        if Self.pillRect.contains(point) {
            return self
        }
        return nil
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let area = trackingArea {
            removeTrackingArea(area)
        }
        let area = NSTrackingArea(
            rect: Self.pillRect,
            options: [.mouseEnteredAndExited, .activeAlways],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(area)
        self.trackingArea = area
    }

    override func mouseEntered(with event: NSEvent) {
        hovered = true
        needsDisplay = true
    }

    override func mouseExited(with event: NSEvent) {
        hovered = false
        needsDisplay = true
    }

    override func mouseDown(with event: NSEvent) {
        animateClickFeedback()
        onClick?()
    }
}

// MARK: - Floating NSPanel (Exact original size: 62x48)
final class FloatingCapsulePanel: NSPanel {
    init(contentRect: NSRect) {
        super.init(
            contentRect: contentRect,
            styleMask: [.nonactivatingPanel, .borderless],
            backing: .buffered,
            defer: false
        )
        self.isFloatingPanel = true
        self.level = .statusBar
        self.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        self.backgroundColor = .clear
        self.isOpaque = false
        self.hasShadow = false
        self.ignoresMouseEvents = false
        self.acceptsMouseMovedEvents = true
    }

    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

// MARK: - Controller matching original app.py exact behavior
@MainActor
final class FloatingCapsuleController {
    static let shared = FloatingCapsuleController()

    private var panel: FloatingCapsulePanel?
    private var pillView: WaveformPillView?

    var currentSelectedText: String = ""
    var currentPlayingText: String = ""
    private(set) var isShowing = false

    private init() {}

    func setupPanelIfNeeded() {
        guard panel == nil else { return }

        let rect = NSRect(x: 100, y: 100, width: 62, height: 48)
        let p = FloatingCapsulePanel(contentRect: rect)
        let view = WaveformPillView(frame: NSRect(x: 0, y: 0, width: 62, height: 48))

        view.onClick = { [weak self] in
            self?.handlePillClick()
        }

        p.contentView = view
        self.panel = p
        self.pillView = view
    }

    /// Hit-test against panel frame in Cocoa screen coordinates (matching NSEvent.mouseLocation)
    func contains(_ screenPoint: NSPoint) -> Bool {
        guard isShowing, let panel = panel else { return false }
        return panel.frame.contains(screenPoint)
    }

    /// Positions and shows the floating pill exactly as in the original app:
    /// y_pos = (y - 34) if (y - 34 > 30) else (y + 12)
    /// rect = NSRect(x + 6 - 10, y_pos - 10, 62, 48)
    func show(at x: CGFloat, y: CGFloat, text: String) {
        guard AppState.shared.selectionOverlayEnabled else { return }
        setupPanelIfNeeded()

        self.currentSelectedText = text
        pillView?.resetWithoutAnimation()

        let yPos: CGFloat = (y - 34 > 30) ? (y - 34) : (y + 12)
        let rect = NSRect(x: x + 6 - 10, y: yPos - 10, width: 62, height: 48)

        panel?.setFrame(rect, display: true)
        panel?.orderFrontRegardless()
        isShowing = true
    }

    func hide() {
        guard isShowing else { return }
        panel?.orderOut(nil)
        isShowing = false
    }

    func toggle() {
        if isShowing {
            hide()
        }
    }

    /// Exact original behavior (1:1 parity with Python app.py handle_button_click):
    /// - If currently playing or paused on the same text: toggle Pause / Resume
    /// - If new text selected: stops prior audio, switches to playing, speaks text
    func handlePillClick() {
        let text = currentSelectedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            hide()
            return
        }

        let appState = AppState.shared

        // If audio is active or paused for the EXACT same text: toggle Pause / Resume
        if text == currentPlayingText && (appState.isSpeaking || appState.isPaused || pillView?.state == .playing || pillView?.state == .paused) {
            if pillView?.state == .paused || appState.isPaused {
                appState.resume()
                pillView?.state = .playing
            } else {
                appState.pause()
                pillView?.state = .paused
            }
            return
        }

        // Fresh playback
        appState.stop()
        currentPlayingText = text
        pillView?.state = .playing
        appState.speak(text: text)
    }

    func resetPillState() {
        pillView?.resetWithoutAnimation()
        currentPlayingText = ""
    }

    func notifyPlaybackFinished() {
        resetPillState()
        // Exact parity with app.py line 851: auto-hide the floating capsule when playback ends
        hide()
    }
}

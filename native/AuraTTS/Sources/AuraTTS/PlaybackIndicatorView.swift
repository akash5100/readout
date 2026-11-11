import AppKit

/// A stable, nonactivating playback indicator. Its dots follow output audio;
/// the playback control stays visible without moving or resizing the capsule.
final class PlaybackIndicatorView: NSView {
    enum State {
        case generating
        case playing
        case paused
    }

    static let panelSize = NSSize(width: 164, height: 48)
    private let capsule = NSRect(x: 10, y: 10, width: 144, height: 28)
    private let toggleButton = NSButton()
    private var timer: Timer?
    private var presented = false
    private var targetLevel: CGFloat = 0
    private var level: CGFloat = 0
    private var history = Array(repeating: CGFloat.zero, count: 41)
    private var phase: CGFloat = 0

    var onToggle: (() -> Void)?
    var onStop: (() -> Void)?
    var state: State = .generating {
        didSet {
            updateControls()
            updateAnimation()
        }
    }

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        setAccessibilityElement(true)
        setAccessibilityRole(.button)
        setAccessibilityHelp("Press to pause or resume, or cancel while preparing")
        setAccessibilityCustomActions([
            NSAccessibilityCustomAction(name: "Stop speech", target: self, selector: #selector(accessibilityStop))
        ])

        toggleButton.frame = NSRect(x: 127, y: 14, width: 20, height: 20)
        toggleButton.isBordered = false
        toggleButton.target = self
        toggleButton.action = #selector(togglePressed)
        addSubview(toggleButton)

        updateControls()
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    @objc private func togglePressed() {
        if state == .generating { onStop?() } else { onToggle?() }
    }
    @objc private func accessibilityStop() -> Bool { onStop?(); return true }

    private func updateControls() {
        let name: String
        let label: String
        switch state {
        case .generating: name = "xmark"; label = "Cancel speech"
        case .playing: name = "pause.fill"; label = "Pause speech"
        case .paused: name = "play.fill"; label = "Resume speech"
        }
        toggleButton.image = NSImage(systemSymbolName: name, accessibilityDescription: label)
        toggleButton.toolTip = label
        setAccessibilityLabel(label)
        needsDisplay = true
    }

    func setPresented(_ value: Bool) {
        presented = value
        updateAnimation()
    }

    func setAudioLevel(_ value: Float) {
        guard state == .playing, !NSWorkspace.shared.accessibilityDisplayShouldReduceMotion else { return }
        targetLevel = min(1, CGFloat(sqrt(max(0, value))) * 2.4)
    }

    private func updateAnimation() {
        timer?.invalidate()
        timer = nil
        guard presented, state != .paused, !NSWorkspace.shared.accessibilityDisplayShouldReduceMotion else {
            targetLevel = 0
            level = 0
            history = Array(repeating: 0, count: history.count)
            needsDisplay = true
            return
        }
        let timer = Timer(timeInterval: 1.0 / 30.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.phase += 0.11
            self.level += (self.targetLevel - self.level) * 0.24
            self.targetLevel *= 0.88
            self.history.removeFirst()
            self.history.append(self.level)
            self.needsDisplay = true
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override func accessibilityPerformPress() -> Bool {
        if state == .generating { onStop?() } else { onToggle?() }
        return true
    }

    override func hitTest(_ point: NSPoint) -> NSView? {
        guard capsule.contains(point) else { return nil }
        return super.hitTest(point)
    }

    override func mouseDown(with event: NSEvent) {
        if state == .generating { onStop?() } else { onToggle?() }
    }

    override func draw(_ dirtyRect: NSRect) {
        let isDark = effectiveAppearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
        let background = isDark ? NSColor(srgbRed: 0.17, green: 0.17, blue: 0.18, alpha: 1) : NSColor.white
        let dotColor = isDark
            ? NSColor(srgbRed: 0.90, green: 0.75, blue: 0.52, alpha: 1)
            : NSColor(srgbRed: 0.48, green: 0.34, blue: 0.20, alpha: 1)
        let shadow = NSShadow()
        shadow.shadowOffset = NSSize(width: 0, height: -2)
        shadow.shadowBlurRadius = 9
        shadow.shadowColor = NSColor.black.withAlphaComponent(isDark ? 0.32 : 0.18)
        NSGraphicsContext.saveGraphicsState()
        shadow.set()
        let path = NSBezierPath(roundedRect: capsule, xRadius: 14, yRadius: 14)
        background.setFill()
        path.fill()
        NSGraphicsContext.restoreGraphicsState()
        NSColor.labelColor.withAlphaComponent(0.10).setStroke()
        path.lineWidth = 0.5
        path.stroke()

        for column in 0..<history.count {
            let x = capsule.minX + 10 + CGFloat(column) * 2.35
            let amplitude = history[column]
            for row in -2...2 {
                let wave = sin(CGFloat(column) * 0.32 - phase) * amplitude * 1.8
                let distance = abs(CGFloat(row) - wave)
                let active = max(0, 1 - distance / (0.55 + amplitude * 1.45))
                let loading = state == .generating ? 0.12 * (0.5 + 0.5 * sin(phase - CGFloat(column) * 0.18)) : 0
                dotColor.withAlphaComponent(min(0.8, 0.13 + loading + active * (0.14 + amplitude * 0.5))).setFill()
                let size: CGFloat = 1.55
                NSBezierPath(ovalIn: NSRect(x: x, y: capsule.midY + CGFloat(row) * 2.35 - size / 2, width: size, height: size)).fill()
            }
        }
    }
}

@MainActor
final class PlaybackIndicatorController {
    static let shared = PlaybackIndicatorController()
    private var panel: FloatingCapsulePanel?
    private var view: PlaybackIndicatorView?
    private(set) var isShowing = false

    private init() {}

    func show() {
        if panel == nil {
            let size = PlaybackIndicatorView.panelSize
            let panel = FloatingCapsulePanel(contentRect: NSRect(origin: .zero, size: size))
            let view = PlaybackIndicatorView(frame: NSRect(origin: .zero, size: size))
            view.onToggle = { [weak self] in self?.togglePlayback() }
            view.onStop = { AppState.shared.stop() }
            panel.contentView = view
            self.panel = panel
            self.view = view
        }
        let mouse = NSEvent.mouseLocation
        if let screen = NSScreen.screens.first(where: { $0.frame.contains(mouse) }) ?? NSScreen.main {
            let visible = screen.visibleFrame
            let size = PlaybackIndicatorView.panelSize
            let origin = NSPoint(x: visible.midX - size.width / 2, y: visible.minY + 12)
            panel?.setFrame(NSRect(origin: origin, size: size), display: true)
        }
        view?.state = .generating
        panel?.orderFrontRegardless()
        view?.setPresented(true)
        isShowing = true
    }

    func contains(_ point: NSPoint) -> Bool {
        isShowing && (panel?.frame.contains(point) ?? false)
    }

    func playbackStarted() { if isShowing { view?.state = .playing } }
    func playbackPaused() { if isShowing { view?.state = .paused } }
    func playbackResumed() { if isShowing { view?.state = .playing } }
    func updateAudioLevel(_ value: Float) { if isShowing { view?.setAudioLevel(value) } }

    func togglePlayback() {
        guard isShowing else { return }
        if AppState.shared.isPaused {
            AppState.shared.resume()
        } else if AppState.shared.isSpeaking {
            AppState.shared.pause()
        }
    }

    func hide() {
        view?.setPresented(false)
        panel?.orderOut(nil)
        isShowing = false
    }
}

import AppKit
import SwiftUI

@MainActor
final class StatusBarController: NSObject, NSMenuDelegate {
    static let shared = StatusBarController()

    private var statusItem: NSStatusItem?
    private let menu = NSMenu()

    override private init() {
        super.init()
        setupStatusItem()
    }

    private func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        guard let button = statusItem?.button else { return }

        button.image = createSoundwaveTemplateImage()
        button.imagePosition = .imageOnly
        button.toolTip = "readout"

        menu.delegate = self
        statusItem?.menu = menu
        rebuildMenu()
    }

    // MARK: - NSMenuDelegate
    func menuWillOpen(_ menu: NSMenu) {
        AppState.shared.checkServer()
        AppState.shared.refreshVoices()
        rebuildMenu()
    }

    // MARK: - Menu Construction
    private func rebuildMenu() {
        menu.removeAllItems()

        // 1. Server Status
        let serverTitle: String
        if AppState.shared.isServerOnline {
            serverTitle = "Server: Online (\(AppState.shared.selectedModel.name))"
        } else {
            serverTitle = "Server: Offline"
        }
        let serverItem = NSMenuItem(title: serverTitle, action: nil, keyEquivalent: "")
        serverItem.isEnabled = false
        menu.addItem(serverItem)

        menu.addItem(NSMenuItem.separator())

        // 2. Enable Selection Overlay Toggle
        let overlayItem = NSMenuItem(
            title: "Enable Selection Overlay",
            action: #selector(toggleSelectionOverlay),
            keyEquivalent: ""
        )
        overlayItem.target = self
        overlayItem.state = AppState.shared.selectionOverlayEnabled ? .on : .off
        menu.addItem(overlayItem)

        menu.addItem(NSMenuItem.separator())

        // 3. Model Submenu
        let modelItem = NSMenuItem(title: "Model", action: nil, keyEquivalent: "")
        let modelMenu = NSMenu()
        for model in AppState.shared.availableModels {
            let item = NSMenuItem(
                title: "\(model.name) (\(model.badge))",
                action: #selector(selectModel(_:)),
                keyEquivalent: ""
            )
            item.target = self
            item.representedObject = model.id
            item.state = (model.id == AppState.shared.selectedModelId) ? .on : .off
            modelMenu.addItem(item)
        }
        modelItem.submenu = modelMenu
        menu.addItem(modelItem)

        // 4. Voice Submenu
        let voiceItem = NSMenuItem(title: "Voice", action: nil, keyEquivalent: "")
        let voiceMenu = NSMenu()
        if AppState.shared.voices.isEmpty {
            let emptyItem = NSMenuItem(title: "No voices found", action: nil, keyEquivalent: "")
            emptyItem.isEnabled = false
            voiceMenu.addItem(emptyItem)
        } else {
            for voice in AppState.shared.voices {
                let item = NSMenuItem(
                    title: voice.name,
                    action: #selector(selectVoice(_:)),
                    keyEquivalent: ""
                )
                item.target = self
                item.representedObject = voice.name
                item.state = (voice.name == AppState.shared.selectedVoiceName) ? .on : .off
                voiceMenu.addItem(item)
            }
        }
        voiceItem.submenu = voiceMenu
        menu.addItem(voiceItem)

        menu.addItem(NSMenuItem.separator())

        // 5. Speech Control (Stop Speech when active)
        if AppState.shared.isSpeaking || AppState.shared.isGenerating {
            let stopItem = NSMenuItem(title: "Stop Speech", action: #selector(stopSpeech), keyEquivalent: "")
            stopItem.target = self
            menu.addItem(stopItem)
            menu.addItem(NSMenuItem.separator())
        }

        // 6. Open Aura Companion Window
        let openItem = NSMenuItem(title: "Open readout…", action: #selector(openMainWindow), keyEquivalent: "")
        openItem.target = self
        menu.addItem(openItem)

        menu.addItem(NSMenuItem.separator())

        // 7. Quit Aura
        let quitItem = NSMenuItem(title: "Quit readout", action: #selector(quitApp), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)
    }

    // MARK: - Actions
    @objc private func toggleSelectionOverlay() {
        AppState.shared.selectionOverlayEnabled.toggle()
    }

    @objc private func selectModel(_ sender: NSMenuItem) {
        if let modelId = sender.representedObject as? String {
            AppState.shared.selectedModelId = modelId
        }
    }

    @objc private func selectVoice(_ sender: NSMenuItem) {
        if let voiceName = sender.representedObject as? String {
            AppState.shared.selectedVoiceName = voiceName
        }
    }

    @objc private func stopSpeech() {
        AppState.shared.stop()
    }

    @objc private func openMainWindow() {
        MainWindowController.shared.show(tab: .speak)
    }

    @objc private func quitApp() {
        NSApplication.shared.terminate(nil)
    }

    // MARK: - Menu Bar Icon
    /// Renders a crisp 4-bar soundwave icon configured as a template image for dark/light menubars.
    private func createSoundwaveTemplateImage() -> NSImage {
        // First check if assets/menubar_icon.png is available in repository
        let repoAsset = AppState.shared.repositoryURL.appendingPathComponent("assets/menubar_icon.png")
        if FileManager.default.fileExists(atPath: repoAsset.path),
           let fileImage = NSImage(contentsOf: repoAsset) {
            fileImage.size = NSSize(width: 18, height: 18)
            fileImage.isTemplate = true
            return fileImage
        }

        // Programmatic 4-bar soundwave template
        let size = NSSize(width: 18, height: 18)
        let image = NSImage(size: size, flipped: false) { rect in
            NSColor.black.setFill()

            // 4 vertical rounded bars with varying heights: 8, 14, 11, 6
            let barWidth: CGFloat = 2.2
            let spacing: CGFloat = 1.8
            let heights: [CGFloat] = [7.0, 14.0, 11.0, 6.0]
            let startX: CGFloat = (rect.width - (CGFloat(heights.count) * barWidth + CGFloat(heights.count - 1) * spacing)) / 2.0

            for (index, height) in heights.enumerated() {
                let x = startX + CGFloat(index) * (barWidth + spacing)
                let y = (rect.height - height) / 2.0
                let barRect = NSRect(x: x, y: y, width: barWidth, height: height)
                let path = NSBezierPath(roundedRect: barRect, xRadius: barWidth / 2.0, yRadius: barWidth / 2.0)
                path.fill()
            }
            return true
        }
        image.isTemplate = true
        return image
    }
}

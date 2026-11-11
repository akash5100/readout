import AppKit
import SwiftUI

@main
struct AuraApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        Settings {
            SettingsTabView()
                .frame(width: 500, height: 400)
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open readout") {
                    MainWindowController.shared.show(tab: .speak)
                }

                Button("History") {
                    MainWindowController.shared.show(tab: .history)
                }
            }

            CommandGroup(replacing: .appSettings) {
                Button("Settings…") {
                    MainWindowController.shared.show(tab: .settings)
                }
            }

            CommandMenu("Speech") {
                Button("Stop Speech") {
                    AppState.shared.stop()
                }

                Divider()

                Button("Toggle Selection Overlay") {
                    AppState.shared.selectionOverlayEnabled.toggle()
                }

                Button("Toggle Floating Capsule") {
                    FloatingCapsuleController.shared.toggle()
                }
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)

        // Initialize status bar item and controller
        _ = StatusBarController.shared

        // Start global selection monitoring for floating capsule HUD
        SelectionMonitor.shared.start()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        MainWindowController.shared.show()
        return true
    }
}

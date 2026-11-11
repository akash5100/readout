import SwiftUI

struct MenuBarView: View {
    @ObservedObject var appState: AppState = AppState.shared
    var onClosePopover: (() -> Void)? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header: "Ready • Pocket TTS (100M)"
            HStack(spacing: 8) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)

                Text(appState.statusHeader)
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.primary)

                Spacer()

                if appState.isSpeaking || appState.isGenerating {
                    Button("Stop") {
                        appState.stop()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .controlSize(.mini)
                }
            }
            .padding(.horizontal, 4)

            Divider()

            // Active Voice Section
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Active Voice")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(appState.selectedVoiceName)
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.accentColor.opacity(0.15))
                        .foregroundColor(.accentColor)
                        .cornerRadius(4)
                }

                // Voice quick picker if multiple voices exist
                if !appState.voices.isEmpty {
                    Menu {
                        ForEach(appState.voices) { voice in
                            Button(action: {
                                appState.selectedVoiceName = voice.name
                            }) {
                                HStack {
                                    Text(voice.name)
                                    if voice.name == appState.selectedVoiceName {
                                        Image(systemName: "checkmark")
                                    }
                                }
                            }
                        }
                    } label: {
                        HStack {
                            Image(systemName: "person.wave.2")
                            Text("Switch Voice (\(appState.selectedVoiceName))")
                            Spacer()
                        }
                        .font(.system(size: 12))
                    }
                    .menuStyle(.borderlessButton)
                }
            }
            .padding(.horizontal, 4)

            Divider()

            // Toggle: Selection Overlay (Enabled/Disabled)
            Toggle(isOn: $appState.selectionOverlayEnabled) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Selection Overlay")
                        .font(.system(size: 12, weight: .medium))
                    Text(appState.selectionOverlayEnabled ? "Capsule enabled on text selection" : "Disabled (0% CPU sleep)")
                        .font(.system(size: 10))
                        .foregroundColor(.secondary)
                }
            }
            .toggleStyle(.switch)
            .padding(.horizontal, 4)

            Divider()

            // Actions
            VStack(spacing: 4) {
                MenuBarButton(
                    title: "Open readout",
                    systemImage: "app.window.checkmark"
                ) {
                    onClosePopover?()
                    MainWindowController.shared.show(tab: .speak)
                }

                MenuBarButton(
                    title: "History",
                    systemImage: "clock.arrow.circlepath"
                ) {
                    onClosePopover?()
                    MainWindowController.shared.show(tab: .history)
                }

                MenuBarButton(
                    title: "Settings",
                    systemImage: "gearshape"
                ) {
                    onClosePopover?()
                    MainWindowController.shared.show(tab: .settings)
                }
            }

            Divider()

            // Quit Aura
            MenuBarButton(
                title: "Quit readout",
                systemImage: "power",
                isDestructive: true
            ) {
                NSApplication.shared.terminate(nil)
            }
        }
        .padding(12)
        .frame(width: 270)
    }

    private var statusColor: Color {
        if appState.isGenerating {
            return .purple
        } else if appState.isSpeaking {
            return .blue
        } else if appState.errorMessage != nil {
            return .red
        } else {
            return .green
        }
    }
}

struct MenuBarButton: View {
    let title: String
    let systemImage: String
    var isDestructive: Bool = false
    let action: () -> Void

    @State private var isHovered: Bool = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: systemImage)
                    .font(.system(size: 12))
                    .frame(width: 16)
                    .foregroundColor(isDestructive ? .red : .primary)

                Text(title)
                    .font(.system(size: 12))
                    .foregroundColor(isDestructive ? .red : .primary)

                Spacer()
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(isHovered ? Color.primary.opacity(0.08) : Color.clear)
            .cornerRadius(5)
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
    }
}

import AppKit
import SwiftUI

// Shared surfaces keep the window consistent across sections and appearances.
struct AuraSurface: ViewModifier {
    func body(content: Content) -> some View {
        content
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.primary.opacity(0.08)))
    }
}

extension View {
    func auraSurface() -> some View { modifier(AuraSurface()) }
}

// MARK: - Main Window View
struct MainWindow: View {
    @ObservedObject var appState: AppState = AppState.shared

    var body: some View {
        NavigationSplitView {
            SidebarView(selectedTab: $appState.selectedTab)
                .navigationSplitViewColumnWidth(min: 200, ideal: 220, max: 260)
        } detail: {
            Group {
                switch appState.selectedTab {
                case .speak:
                    SpeakTabView()
                case .voices:
                    VoicesTabView()
                case .history:
                    HistoryTabView()
                case .models:
                    ModelsTabView()
                case .settings:
                    SettingsTabView()
                }
            }
            .frame(minWidth: 620, minHeight: 520)
        }
        .frame(minWidth: 840, minHeight: 560)
    }
}

// MARK: - Sidebar View
struct SidebarView: View {
    @Binding var selectedTab: SidebarTab
    @ObservedObject var appState: AppState = AppState.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // App Branding Header
            HStack(spacing: 8) {
                Image(systemName: "waveform")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.secondary)
                Text("readout")
                    .font(.system(size: 14, weight: .semibold))
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.top, 16)
            .padding(.bottom, 10)

            // Navigation Items
            List(SidebarTab.allCases, selection: $selectedTab) { tab in
                NavigationLink(value: tab) {
                    Label(tab.rawValue, systemImage: tab.iconName)
                        .font(.system(size: 13, weight: .medium))
                }
            }
            .listStyle(.sidebar)

            Spacer()

            // Bottom Server & Voice Status Card
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Circle()
                        .fill(appState.isServerOnline ? Color.green : Color.red)
                        .frame(width: 7, height: 7)

                    Text(appState.isServerOnline ? "Ready" : "Server offline")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(appState.isServerOnline ? .primary : .secondary)

                    Spacer()

                    Button(action: { appState.checkServer() }) {
                        Image(systemName: "arrow.clockwise")
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Refresh server connection")
                }

                HStack(spacing: 4) {
                    Text(appState.selectedVoiceName)
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)

                    Spacer()

                    Text(appState.selectedModel.name)
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
        }
    }
}

// MARK: - Speak Tab
struct SpeakTabView: View {
    @ObservedObject var appState: AppState = AppState.shared
    @State private var inputText: String = ""

    var characterCount: Int { inputText.count }
    var wordCount: Int {
        inputText.split { $0.isWhitespace || $0.isNewline }.count
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Hero Card
                HStack(alignment: .center) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Speak")
                            .font(.system(size: 18, weight: .semibold))

                        Text("Type here, or select text anywhere on your Mac.")
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                    }

                    Spacer()
                }

                // Error message if any
                if let error = appState.errorMessage {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.red)
                        Text(error)
                            .font(.system(size: 12))
                            .foregroundColor(.red)
                        Spacer()
                        Button("Dismiss") { appState.errorMessage = nil }
                            .buttonStyle(.plain)
                            .font(.system(size: 11))
                    }
                    .padding(10)
                    .background(Color.red.opacity(0.1))
                    .cornerRadius(8)
                }

                // Text Input Field Card
                VStack(alignment: .leading, spacing: 10) {
                    ZStack(alignment: .topLeading) {
                        if inputText.isEmpty {
                            Text("Type or paste any text to synthesize...")
                                .foregroundColor(.secondary.opacity(0.6))
                                .font(.system(size: 13))
                                .padding(.horizontal, 4)
                                .padding(.vertical, 8)
                        }

                        TextEditor(text: $inputText)
                            .font(.system(size: 13))
                            .lineSpacing(4)
                            .frame(minHeight: 140)
                            .scrollContentBackground(.hidden)
                    }

                    Divider()

                    // Bottom Bar
                    HStack {
                        Text("\(characterCount) chars • \(wordCount) words")
                            .font(.system(size: 11))
                            .foregroundColor(.secondary)

                        if !inputText.isEmpty {
                            Button("Clear") { inputText = "" }
                                .buttonStyle(.plain)
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        // Voice Selector Picker
                        Picker("Voice", selection: $appState.selectedVoiceName) {
                            ForEach(appState.voices) { voice in
                                Text(voice.name).tag(voice.name)
                            }
                        }
                        .labelsHidden()
                        .pickerStyle(.menu)
                        .frame(width: 120)

                        // Speak / Stop Button
                        if appState.isSpeaking || appState.isGenerating {
                            Button(action: { appState.stop() }) {
                                HStack(spacing: 6) {
                                    Image(systemName: "stop.fill")
                                    Text("Stop")
                                }
                                .frame(minWidth: 80)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.red)
                        } else {
                            Button(action: { appState.speak(text: inputText) }) {
                                HStack(spacing: 6) {
                                    Image(systemName: "waveform")
                                    Text("Speak")
                                }
                                .frame(minWidth: 80)
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        }
                    }
                }
                .padding(14)
                .auraSurface()

                // Recent Speech Card with Replay
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Recent Speech")
                            .font(.system(size: 14, weight: .semibold))

                        Spacer()

                        if appState.lastSpokenRecord != nil {
                            Button("History (\(appState.history.count))") {
                                appState.selectedTab = .history
                            }
                            .buttonStyle(.plain)
                            .font(.system(size: 12))
                            .foregroundColor(.accentColor)
                        }
                    }

                    if let recent = appState.lastSpokenRecord {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("\"\(recent.text)\"")
                                .font(.system(size: 13, weight: .regular))
                                .lineLimit(3)
                                .foregroundColor(.primary)

                            HStack {
                                HStack(spacing: 4) {
                                    Image(systemName: "person.wave.2")
                                    Text(recent.voice)
                                }
                                .font(.system(size: 11, weight: .medium))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.primary.opacity(0.06))
                                .cornerRadius(4)

                                Text("•")
                                    .foregroundColor(.secondary)

                                Text(recent.model)
                                    .font(.system(size: 11))
                                    .foregroundColor(.secondary)

                                Text("•")
                                    .foregroundColor(.secondary)

                                Text(recent.timeFormatted)
                                    .font(.system(size: 11))
                                    .foregroundColor(.secondary)

                                Spacer()

                                Button(action: {
                                    NSPasteboard.general.clearContents()
                                    NSPasteboard.general.setString(recent.text, forType: .string)
                                }) {
                                    Image(systemName: "doc.on.doc")
                                        .font(.system(size: 11))
                                }
                                .buttonStyle(.plain)
                                .help("Copy text")

                                Button(action: {
                                    appState.speak(text: recent.text)
                                }) {
                                    HStack(spacing: 4) {
                                        Image(systemName: "arrow.counterclockwise")
                                        Text("Replay")
                                    }
                                    .font(.system(size: 11, weight: .medium))
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                        }
                        .padding(12)
                        .background(Color(nsColor: .textBackgroundColor).opacity(0.4))
                        .cornerRadius(8)
                    } else {
                        HStack {
                            Spacer()
                            Text("No recent speech items yet.")
                                .font(.system(size: 12))
                                .foregroundColor(.secondary)
                            Spacer()
                        }
                        .padding(.vertical, 16)
                    }
                }
                .padding(14)
                .auraSurface()
            }
            .padding(20)
        }
    }
}

// MARK: - Voices Tab
struct VoicesTabView: View {
    @ObservedObject var appState: AppState = AppState.shared

    // YouTube Cloner Fields
    @State private var youtubeURL: String = ""
    @State private var voiceName: String = ""
    @State private var selectedDuration: Int = 5
    @State private var isCloning: Bool = false
    @State private var cloneMessage: String? = nil
    @State private var cloneError: String? = nil
    @State private var detectedTimestamp: Int = 0
    @State private var isClonerExpanded = false

    // Search
    @State private var searchText: String = ""

    var filteredVoices: [VoiceEntry] {
        if searchText.isEmpty {
            return appState.voices
        }
        return appState.voices.filter { $0.name.localizedCaseInsensitiveContains(searchText) }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Card 2: Voice Library Grid
                VStack(alignment: .leading, spacing: 14) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Voices")
                                .font(.system(size: 18, weight: .semibold))
                            Text("Choose a voice for spoken text.")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        // Search Field
                        TextField("Filter voices...", text: $searchText)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 160)

                        Button(action: { appState.refreshVoices() }) {
                            Image(systemName: "arrow.clockwise")
                        }
                        .buttonStyle(.plain)
                        .help("Refresh library")
                    }

                    if filteredVoices.isEmpty {
                        HStack {
                            Spacer()
                            Text("No voices match your search.")
                                .font(.system(size: 12))
                                .foregroundColor(.secondary)
                            Spacer()
                        }
                        .padding(.vertical, 24)
                    } else {
                        VStack(spacing: 0) {
                            ForEach(filteredVoices) { voice in
                                VoiceCardView(voice: voice)
                                if voice.id != filteredVoices.last?.id {
                                    Divider().padding(.leading, 54)
                                }
                            }
                        }
                        .auraSurface()
                    }
                }

                DisclosureGroup(isExpanded: $isClonerExpanded) {
                // Card 1: Clone Voice from YouTube
                VStack(alignment: .leading, spacing: 14) {
                    Text("Use a short, clear clip to create a voice.")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)

                    // YouTube URL Field with auto timestamp extraction
                    VStack(alignment: .leading, spacing: 4) {
                        Text("YouTube URL (with timestamp, e.g. ?t=42)")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(.secondary)

                        HStack {
                            TextField("https://youtu.be/video_id?t=42", text: $youtubeURL)
                                .textFieldStyle(.roundedBorder)
                                .onChange(of: youtubeURL) { _, newValue in
                                    let ts = YouTubeClip.timestamp(newValue)
                                    detectedTimestamp = ts
                                    if voiceName.isEmpty && !newValue.isEmpty {
                                        let sample = "voice_\(ts > 0 ? "\(ts)s" : "clip")"
                                        voiceName = YouTubeClip.safeName(sample)
                                    }
                                }

                            if detectedTimestamp > 0 {
                                HStack(spacing: 3) {
                                    Image(systemName: "clock")
                                    Text("\(detectedTimestamp)s")
                                }
                                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 4)
                                .background(Color.green.opacity(0.15))
                                .foregroundColor(.green)
                                .cornerRadius(6)
                            }
                        }
                    }

                    // Voice Name & Duration Picker
                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Voice Name")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundColor(.secondary)

                            TextField("e.g. mark, sam, lex", text: $voiceName)
                                .textFieldStyle(.roundedBorder)
                        }

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Clip Duration")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundColor(.secondary)

                            // 5s / 6s / 7s duration picker
                            Picker("Duration", selection: $selectedDuration) {
                                Text("5s").tag(5)
                                Text("6s").tag(6)
                                Text("7s").tag(7)
                            }
                            .labelsHidden()
                            .pickerStyle(.segmented)
                            .frame(width: 140)
                        }
                    }

                    // Clone Output Messages
                    if let msg = cloneMessage {
                        HStack(spacing: 6) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                            Text(msg)
                                .font(.system(size: 11))
                                .foregroundColor(.green)
                            Spacer()
                        }
                    }

                    if let err = cloneError {
                        HStack(spacing: 6) {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundColor(.red)
                            Text(err)
                                .font(.system(size: 11))
                                .foregroundColor(.red)
                            Spacer()
                        }
                    }

                    // Clone & Cache Button
                    HStack {
                        Spacer()

                        if isCloning {
                            HStack(spacing: 8) {
                                ProgressView()
                                    .controlSize(.small)
                                Text("Downloading section & building FlowLM cache...")
                                    .font(.system(size: 12))
                                    .foregroundColor(.secondary)
                            }
                        }

                        Button(action: {
                            runYouTubeCloner()
                        }) {
                            HStack(spacing: 6) {
                                Image(systemName: "bolt.badge.automatic.fill")
                                Text("Clone & Cache")
                            }
                            .frame(minWidth: 120)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(youtubeURL.isEmpty || voiceName.isEmpty || isCloning)
                    }
                }
                .padding(.top, 12)
                } label: {
                    Label("Add a voice from YouTube", systemImage: "plus")
                        .font(.system(size: 13, weight: .medium))
                }
                .padding(16)
                .auraSurface()
            }
            .padding(20)
        }
    }

    private func runYouTubeCloner() {
        guard !youtubeURL.isEmpty, !voiceName.isEmpty else { return }

        isCloning = true
        cloneMessage = nil
        cloneError = nil

        let safe = YouTubeClip.safeName(voiceName)

        Task {
            do {
                let result = try await appState.cloneVoice(
                    url: youtubeURL,
                    name: safe,
                    duration: selectedDuration
                )
                if result.status == 0 {
                    cloneMessage = "Successfully cloned '\(safe)' and generated caches!"
                    youtubeURL = ""
                    voiceName = ""
                } else {
                    cloneError = "Cloning exited with code \(result.status): \(result.output.suffix(160))"
                }
            } catch {
                cloneError = error.localizedDescription
            }
            isCloning = false
        }
    }
}

// MARK: - Voice Card View
struct VoiceCardView: View {
    let voice: VoiceEntry
    @ObservedObject var appState: AppState = AppState.shared

    var isActive: Bool {
        appState.selectedVoiceName == voice.name
    }

    var body: some View {
        HStack(spacing: 12) {
            // Avatar circle
            ZStack {
                Circle()
                    .fill(Color.primary.opacity(0.06))
                    .frame(width: 32, height: 32)
                Text(String(voice.name.prefix(1)).uppercased())
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.secondary)
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(voice.name)
                    .font(.system(size: 13, weight: .medium))
                // Cache Badges
                Text(voice.formats.isEmpty ? "Standard voice" : voice.formats)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            // Actions: Play Sample + Use Voice
            Button(action: { appState.audition(voice: voice) }) {
                Label("Sample", systemImage: "play.fill")
            }
            .controlSize(.small)

            if isActive {
                Label("Selected", systemImage: "checkmark")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.secondary)
                    .frame(width: 76, alignment: .trailing)
            } else {
                Button("Select") { appState.selectedVoiceName = voice.name }
                    .controlSize(.small)
                    .frame(width: 76, alignment: .trailing)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }
}

struct BadgePill: View {
    let title: String
    let color: Color

    var body: some View {
        Text(title)
            .font(.system(size: 9, weight: .medium))
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundColor(color)
            .cornerRadius(4)
    }
}

// MARK: - History Tab
struct HistoryTabView: View {
    @ObservedObject var appState: AppState = AppState.shared
    @State private var query: String = ""

    var filteredItems: [HistoryItem] {
        if query.isEmpty { return appState.history }
        return appState.history.filter {
            $0.text.localizedCaseInsensitiveContains(query) ||
            $0.voice.localizedCaseInsensitiveContains(query)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header Bar
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Speech History")
                        .font(.system(size: 16, weight: .bold))
                    Text("\(appState.history.count) items recorded.")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }

                Spacer()

                TextField("Search history...", text: $query)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 180)

                if !appState.history.isEmpty {
                    Button("Clear History") {
                        appState.clearHistory()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
            }
            .padding(16)

            Divider()

            if filteredItems.isEmpty {
                Spacer()
                VStack(spacing: 8) {
                    Image(systemName: "clock.arrow.circlepath")
                        .font(.system(size: 32))
                        .foregroundColor(.secondary.opacity(0.5))
                    Text("No history entries found.")
                        .font(.system(size: 13))
                        .foregroundColor(.secondary)
                }
                Spacer()
            } else {
                List(filteredItems) { item in
                    VStack(alignment: .leading, spacing: 6) {
                        Text(item.text)
                            .font(.system(size: 13))
                            .lineLimit(3)

                        HStack(spacing: 8) {
                            Text(item.voice)
                                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.15))
                                .foregroundColor(.accentColor)
                                .cornerRadius(4)

                            Text(item.model)
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)

                            Text("•")
                                .foregroundColor(.secondary)

                            Text(item.timeFormatted)
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)

                            Spacer()

                            Button(action: {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(item.text, forType: .string)
                            }) {
                                Image(systemName: "doc.on.doc")
                                    .font(.system(size: 11))
                            }
                            .buttonStyle(.plain)
                            .help("Copy text")

                            Button(action: {
                                appState.speak(text: item.text)
                            }) {
                                HStack(spacing: 4) {
                                    Image(systemName: "arrow.counterclockwise")
                                    Text("Replay")
                                }
                                .font(.system(size: 11))
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)

                            Button(action: {
                                appState.deleteHistoryItem(item)
                            }) {
                                Image(systemName: "trash")
                                    .font(.system(size: 11))
                                    .foregroundColor(.red.opacity(0.7))
                            }
                            .buttonStyle(.plain)
                            .help("Delete item")
                        }
                    }
                    .padding(.vertical, 6)
                }
                .listStyle(.inset)
            }
        }
    }
}

// MARK: - Models Tab
struct ModelsTabView: View {
    @ObservedObject var appState: AppState = AppState.shared

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Speech Synthesis Models")
                        .font(.system(size: 18, weight: .bold))
                    Text("Select the neural speech architecture that matches your hardware and speed requirements.")
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                }

                VStack(spacing: 12) {
                    ForEach(appState.availableModels) { model in
                        ModelCardView(
                            model: model,
                            isSelected: appState.selectedModelId == model.id
                        ) {
                            appState.selectedModelId = model.id
                        }
                    }
                }
            }
            .padding(20)
        }
    }
}

struct ModelCardView: View {
    let model: ModelOption
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 14) {
            Image(systemName: "cpu.fill")
                .font(.system(size: 22))
                .foregroundColor(isSelected ? .accentColor : .secondary)
                .frame(width: 36)

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(model.name)
                        .font(.system(size: 14, weight: .bold))

                    Text(model.badge)
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.primary.opacity(0.08))
                        .cornerRadius(4)

                    if model.isRecommended {
                        Text("Recommended")
                            .font(.system(size: 9, weight: .bold))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(Color.green.opacity(0.15))
                            .foregroundColor(.green)
                            .cornerRadius(4)
                    }
                }

                Text(model.description)
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)

                HStack(spacing: 12) {
                    HStack(spacing: 3) {
                        Image(systemName: "bolt.fill")
                        Text(model.latency)
                    }
                    .font(.system(size: 10, weight: .medium))
                    .foregroundColor(.accentColor)

                    HStack(spacing: 3) {
                        Image(systemName: "memorychip")
                        Text(model.memory)
                    }
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
                }
            }

            Spacer()

            if isSelected {
                HStack(spacing: 4) {
                    Image(systemName: "checkmark")
                    Text("Active")
                }
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(.accentColor)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Color.accentColor.opacity(0.12))
                .cornerRadius(6)
            } else {
                Button("Select", action: onSelect)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 1.5)
        )
    }
}

// MARK: - Settings Tab
struct AuraSettingsRow<Content: View>: View {
    let icon: String
    let title: String
    let detail: String?
    let content: Content

    init(_ title: String, icon: String, detail: String? = nil, @ViewBuilder content: () -> Content) {
        self.title = title
        self.icon = icon
        self.detail = detail
        self.content = content()
    }

    private var label: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(.secondary)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.system(size: 13, weight: .medium))
                if let detail {
                    Text(detail).font(.system(size: 11)).foregroundColor(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 20) {
                label.frame(maxWidth: .infinity, alignment: .leading)
                content
            }
            VStack(alignment: .leading, spacing: 10) {
                label
                content.padding(.leading, 30)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 13)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct SettingsTabView: View {
    @ObservedObject var appState: AppState = AppState.shared

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Settings")
                    .font(.system(size: 18, weight: .semibold))

                VStack(spacing: 0) {
                    // Speech Rate Section
                    AuraSettingsRow("Speech rate", icon: "speedometer") {
                        HStack(spacing: 8) {
                            Slider(value: $appState.speechRate, in: 0.5...2.0, step: 0.05)
                                .frame(width: 140)
                                .accessibilityLabel("Speech rate")
                            Text(String(format: "%.2fx", appState.speechRate))
                                .font(.system(size: 11, design: .monospaced))
                                .frame(width: 42, alignment: .trailing)
                            Button("Reset") { appState.speechRate = 1.0 }
                                .controlSize(.small)
                        }
                    }
                    Divider().padding(.leading, 46)
                    // Selection Overlay
                    AuraSettingsRow("Selection overlay", icon: "selection.pin.in.out", detail: "Show a speech button when you select text.") {
                        Toggle("Enable selection overlay", isOn: $appState.selectionOverlayEnabled)
                            .labelsHidden()
                            .toggleStyle(.switch)
                    }
                }
                .auraSurface()

                // Server Connection Section
                VStack(alignment: .leading, spacing: 8) {
                    Text("Connection").font(.system(size: 12, weight: .medium)).foregroundColor(.secondary)
                    VStack(spacing: 0) {
                        AuraSettingsRow("Speech server", icon: "server.rack", detail: appState.isServerOnline ? "Connected" : "Disconnected") {
                            HStack(spacing: 8) {
                                TextField("127.0.0.1", text: $appState.serverHost)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 140)
                                    .accessibilityLabel("Server host")
                                TextField("5050", value: $appState.serverPort, format: .number.grouping(.never))
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 70)
                                    .accessibilityLabel("Server port")
                                Button("Test") { appState.checkServer() }
                                    .controlSize(.small)
                                    .fixedSize()
                            }
                        }
                    }
                    .auraSurface()
                }

                // Repository & Cache Storage
                VStack(alignment: .leading, spacing: 8) {
                    Text("Storage").font(.system(size: 12, weight: .medium)).foregroundColor(.secondary)
                    VStack(spacing: 0) {
                        AuraSettingsRow("Repository", icon: "folder") {
                            HStack(spacing: 8) {
                                TextField("/path/to/tts-ext", text: $appState.repositoryPath)
                                    .textFieldStyle(.roundedBorder)
                                    .frame(width: 220)
                                    .accessibilityLabel("Repository path")
                                Button("Browse…") {
                                    let panel = NSOpenPanel()
                                    panel.canChooseFiles = false
                                    panel.canChooseDirectories = true
                                    panel.allowsMultipleSelection = false
                                    if panel.runModal() == .OK, let url = panel.url {
                                        appState.repositoryPath = url.path
                                    }
                                }
                                .controlSize(.small)
                                .fixedSize()
                            }
                        }
                        Divider().padding(.leading, 46)
                        AuraSettingsRow("Voice cache", icon: "internaldrive", detail: VoiceEntry.cacheDirectory.path) {
                            Button("Reveal in Finder") {
                                NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: VoiceEntry.cacheDirectory.path)
                            }
                            .controlSize(.small)
                        }
                    }
                    .auraSurface()
                }
            }
            .padding(24)
        }
    }
}

// MARK: - Main Window Controller
@MainActor
final class MainWindowController: NSObject, NSWindowDelegate {
    static let shared = MainWindowController()

    private var window: NSWindow?

    override private init() {
        super.init()
    }

    func show(tab: SidebarTab? = nil) {
        if let tab = tab {
            AppState.shared.selectedTab = tab
        }

        if let existing = window {
            existing.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let newWindow = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 880, height: 600),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        newWindow.title = "readout"
        newWindow.titleVisibility = .hidden
        newWindow.titlebarAppearsTransparent = true
        newWindow.isReleasedWhenClosed = false
        newWindow.delegate = self
        newWindow.center()
        newWindow.setFrameAutosaveName("AuraTTSMainWindow")

        let hostingView = NSHostingView(rootView: MainWindow())
        newWindow.contentView = hostingView

        self.window = newWindow
        newWindow.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func windowWillClose(_ notification: Notification) {
        // Keep controller reference ready for reopen
    }
}

import AppKit
import Foundation

enum SidebarTab: String, CaseIterable, Identifiable {
    case speak = "Speak"
    case voices = "Voices"
    case history = "History"
    case models = "Models"
    case settings = "Settings"

    var id: String { rawValue }

    var iconName: String {
        switch self {
        case .speak: return "waveform.and.mic"
        case .voices: return "person.wave.2.fill"
        case .history: return "clock.arrow.circlepath"
        case .models: return "cpu.fill"
        case .settings: return "gearshape.fill"
        }
    }
}

struct ModelOption: Identifiable, Hashable {
    let id: String
    let name: String
    let badge: String
    let description: String
    let latency: String
    let memory: String
    let isRecommended: Bool
}

struct HistoryItem: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    let text: String
    let voice: String
    let model: String
    let timestamp: Date

    var timeFormatted: String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .short
        return formatter.localizedString(for: timestamp, relativeTo: Date())
    }
}

@MainActor
final class AppState: ObservableObject {
    static let shared = AppState()

    // MARK: - Models
    let availableModels: [ModelOption] = [
        ModelOption(
            id: "pocket",
            name: "Pocket TTS",
            badge: "100M",
            description: "Ultra-low latency FlowLM model with pre-cached voice conditioning.",
            latency: "~100ms TTFA",
            memory: "100 MB",
            isRecommended: true
        ),
        ModelOption(
            id: "pocket-18m",
            name: "Pocket TTS",
            badge: "18M",
            description: "Fast English speech with model-specific voice caching.",
            latency: "Fast on CPU",
            memory: "18M parameters",
            isRecommended: false
        ),
        ModelOption(
            id: "supra",
            name: "SupraTTS",
            badge: "1M",
            description: "Ultra-lightweight fast neural synthesis for instant responses.",
            latency: "<50ms TTFA",
            memory: "30 MB",
            isRecommended: false
        ),
        ModelOption(
            id: "chatterbox",
            name: "Chatterbox Turbo",
            badge: "0.3B",
            description: "Expressive high quality speech synthesis powered by MLX community.",
            latency: "~200ms TTFA",
            memory: "350 MB",
            isRecommended: false
        ),
        ModelOption(
            id: "qwen3",
            name: "Qwen3-TTS",
            badge: "0.6B",
            description: "Zero-shot voice cloning with deep natural prosody.",
            latency: "~400ms TTFA",
            memory: "650 MB",
            isRecommended: false
        ),
    ]

    // MARK: - Published State
    @Published var selectedTab: SidebarTab = .speak
    @Published var selectedModelId: String {
        didSet { UserDefaults.standard.set(selectedModelId, forKey: "selectedModelId") }
    }
    @Published var selectedVoiceName: String {
        didSet { UserDefaults.standard.set(selectedVoiceName, forKey: "selectedVoiceName") }
    }
    @Published var selectionOverlayEnabled: Bool {
        didSet { UserDefaults.standard.set(selectionOverlayEnabled, forKey: "selectionOverlayEnabled") }
    }
    @Published var speechRate: Double {
        didSet { UserDefaults.standard.set(speechRate, forKey: "speechRate") }
    }
    @Published var serverHost: String {
        didSet { UserDefaults.standard.set(serverHost, forKey: "serverHost") }
    }
    @Published var serverPort: Int {
        didSet { UserDefaults.standard.set(serverPort, forKey: "serverPort") }
    }
    @Published var repositoryPath: String {
        didSet { UserDefaults.standard.set(repositoryPath, forKey: "repositoryPath") }
    }

    @Published var voices: [VoiceEntry] = []
    @Published var history: [HistoryItem] = []
    @Published var isSpeaking: Bool = false
    @Published var isPaused: Bool = false
    @Published var isGenerating: Bool = false
    @Published var isServerOnline: Bool = false
    @Published var isCheckingServer: Bool = false
    @Published var statusText: String = "Ready"
    @Published var errorMessage: String? = nil

    let ttsClient = TTSClient()
    private var speechTask: Task<Void, Never>?

    var selectedModel: ModelOption {
        availableModels.first { $0.id == selectedModelId } ?? availableModels[0]
    }

    var selectedVoice: VoiceEntry {
        voices.first { $0.name == selectedVoiceName } ?? (voices.first ?? VoiceEntry(name: "alba"))
    }

    var statusHeader: String {
        "\(statusText) • \(selectedModel.name) (\(selectedModel.badge))"
    }

    var lastSpokenRecord: HistoryItem? {
        history.first
    }

    var repositoryURL: URL {
        URL(fileURLWithPath: repositoryPath)
    }

    private init() {
        self.selectedModelId = UserDefaults.standard.string(forKey: "selectedModelId") ?? "pocket"
        self.selectedVoiceName = UserDefaults.standard.string(forKey: "selectedVoiceName") ?? "alba"
        self.selectionOverlayEnabled = UserDefaults.standard.object(forKey: "selectionOverlayEnabled") == nil ? true : UserDefaults.standard.bool(forKey: "selectionOverlayEnabled")
        self.speechRate = UserDefaults.standard.double(forKey: "speechRate") == 0 ? 1.0 : UserDefaults.standard.double(forKey: "speechRate")
        self.serverHost = UserDefaults.standard.string(forKey: "serverHost") ?? "127.0.0.1"
        self.serverPort = UserDefaults.standard.integer(forKey: "serverPort") == 0 ? 5050 : UserDefaults.standard.integer(forKey: "serverPort")
        self.repositoryPath = UserDefaults.standard.string(forKey: "repositoryPath") ?? "/Users/akash/Workspace/_experiments/tts-ext"

        self.ttsClient.host = self.serverHost
        self.ttsClient.port = self.serverPort
        self.ttsClient.onAudioLevel = { level in
            Task { @MainActor in
                PlaybackIndicatorController.shared.updateAudioLevel(level)
            }
        }
        self.ttsClient.onFinished = { [weak self] in
            Task { @MainActor in
                self?.isSpeaking = false
                self?.isPaused = false
                self?.statusText = "Ready"
                FloatingCapsuleController.shared.notifyPlaybackFinished()
                PlaybackIndicatorController.shared.hide()
            }
        }

        loadHistory()
        refreshVoices()
        checkServer()
    }

    // MARK: - Actions
    func refreshVoices() {
        let loaded = VoiceLibrary.load(repository: repositoryURL)
        self.voices = loaded
        if !loaded.contains(where: { $0.name == selectedVoiceName }) {
            if let first = loaded.first {
                self.selectedVoiceName = first.name
            }
        }
    }

    func speak(text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        // Stop prior audio cleanly
        speechTask?.cancel()
        ttsClient.stop()
        isSpeaking = false
        isPaused = false
        errorMessage = nil
        isGenerating = true
        statusText = "Generating..."
        FloatingCapsuleController.shared.hide()
        PlaybackIndicatorController.shared.show()

        speechTask = Task {
            do {
                ttsClient.host = serverHost
                ttsClient.port = serverPort
                let voice = self.selectedVoice
                let model = self.selectedModel
                try await ttsClient.stream(text: trimmed, model: model.id, voice: voice) {
                    self.isGenerating = false
                    self.isSpeaking = true
                    self.isPaused = false
                    self.statusText = "Speaking..."
                    PlaybackIndicatorController.shared.playbackStarted()
                    self.recordSpeech(HistoryItem(text: trimmed, voice: voice.name, model: model.name, timestamp: Date()))
                }
            } catch {
                guard !Task.isCancelled else { return }
                self.ttsClient.stop()
                self.isGenerating = false
                self.isSpeaking = false
                self.isPaused = false
                self.statusText = "Error"
                self.errorMessage = error.localizedDescription
                FloatingCapsuleController.shared.notifyPlaybackFinished()
                PlaybackIndicatorController.shared.hide()
            }
        }
    }

    func pause() {
        ttsClient.pause()
        isPaused = true
        statusText = "Paused"
        PlaybackIndicatorController.shared.playbackPaused()
    }

    func resume() {
        ttsClient.resume()
        isPaused = false
        statusText = "Speaking..."
        PlaybackIndicatorController.shared.playbackResumed()
    }

    func stop() {
        speechTask?.cancel()
        speechTask = nil
        ttsClient.stop()
        isSpeaking = false
        isPaused = false
        isGenerating = false
        statusText = "Ready"
        FloatingCapsuleController.shared.resetPillState()
        PlaybackIndicatorController.shared.hide()
    }

    func audition(voice: VoiceEntry) {
        if let audioURL = voice.audio {
            do {
                try ttsClient.audition(audioURL)
                isSpeaking = true
                isPaused = false
                statusText = "Auditioning \(voice.name)..."
            } catch {
                errorMessage = error.localizedDescription
            }
        } else {
            speak(text: "Hello! This is \(voice.name) speaking through Aura.")
        }
    }

    func cloneVoice(url: String, name: String, duration: Int) async throws -> CloneResult {
        let safe = YouTubeClip.safeName(name)
        guard !safe.isEmpty else {
            throw AuraError(message: "Voice name cannot be empty.")
        }
        let result = try await VoiceCloner.run(
            repository: repositoryURL,
            url: url,
            name: safe,
            duration: duration
        )
        if result.status == 0 {
            refreshVoices()
            selectedVoiceName = safe
        }
        return result
    }

    func checkServer() {
        guard !isCheckingServer else { return }
        isCheckingServer = true
        Task {
            guard let url = URL(string: "http://\(serverHost):\(serverPort)/") else {
                self.isServerOnline = false
                self.isCheckingServer = false
                return
            }
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.timeoutInterval = 2.0
            do {
                let (_, response) = try await URLSession.shared.data(for: request)
                if let http = response as? HTTPURLResponse {
                    self.isServerOnline = (http.statusCode == 200)
                } else {
                    self.isServerOnline = false
                }
            } catch {
                self.isServerOnline = false
            }
            self.isCheckingServer = false
        }
    }

    // MARK: - History Persistence
    private func loadHistory() {
        if let data = UserDefaults.standard.data(forKey: "speechHistory"),
           let decoded = try? JSONDecoder().decode([HistoryItem].self, from: data) {
            self.history = decoded
        }
    }

    private func saveHistory() {
        if let encoded = try? JSONEncoder().encode(history) {
            UserDefaults.standard.set(encoded, forKey: "speechHistory")
        }
    }

    func recordSpeech(_ item: HistoryItem) {
        history.insert(item, at: 0)
        if history.count > 100 {
            history.removeLast()
        }
        saveHistory()
    }

    func deleteHistoryItem(_ item: HistoryItem) {
        history.removeAll { $0.id == item.id }
        saveHistory()
    }

    func clearHistory() {
        history.removeAll()
        saveHistory()
    }
}

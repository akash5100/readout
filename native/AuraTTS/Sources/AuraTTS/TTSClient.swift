import AVFoundation
import Foundation

struct AuraError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

enum WAVDecoder {
    static func decode(_ data: Data) throws -> AVAudioPCMBuffer {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        try data.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let file = try AVAudioFile(forReading: tempURL)
        guard file.length > 0, file.length <= Int64(UInt32.max),
              let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw AuraError(message: "Failed to allocate audio buffer for synthesized speech.")
        }
        try file.read(into: buffer)
        return buffer
    }
}

@MainActor
final class TTSClient {
    var host: String = "127.0.0.1"
    var port: Int = 5050
    var onFinished: (@Sendable () -> Void)?
    var onAudioLevel: (@Sendable (Float) -> Void)?

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var playbackID = UUID()
    private var queuedBuffers = 0
    private var queuedSeconds = 0.0
    private var streamComplete = false
    private var streamStarted = false
    private var meterInstalled = false

    init() {
        engine.attach(player)
    }

    func synthesize(text: String, model: String, voice: VoiceEntry) async throws -> Data {
        guard let url = URL(string: "http://\(host):\(port)/tts") else {
            throw AuraError(message: "Invalid TTS server URL.")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var body: [String: Any] = ["text": text, "model": model, "voice": voice.name]
        if model == "pocket", let cache = voice.pocketCache {
            body["ref_audio"] = cache.path
        } else if let audio = voice.audio {
            body["ref_audio"] = audio.path
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw AuraError(message: "The speech server returned an invalid response.")
        }
        guard (200..<300).contains(http.statusCode) else {
            let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            throw AuraError(message: payload?["message"] as? String ?? "Speech server error (\(http.statusCode)).")
        }
        return data
    }

    func stream(text: String, model: String, voice: VoiceEntry, onStart: () -> Void) async throws {
        guard let url = URL(string: "http://\(host):\(port)/tts/stream") else {
            throw AuraError(message: "Invalid TTS server URL.")
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        var body: [String: Any] = ["text": text, "model": model, "voice": voice.name]
        if model == "pocket", let cache = voice.pocketCache {
            body["ref_audio"] = cache.path
        } else if let audio = voice.audio {
            body["ref_audio"] = audio.path
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        stop()
        let id = playbackID
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw AuraError(message: "The speech server could not start generation.")
        }
        var receivedEnd = false
        for try await line in bytes.lines {
            try Task.checkCancellation()
            guard playbackID == id else { throw CancellationError() }
            guard let data = line.data(using: .utf8),
                  let frame = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let type = frame["type"] as? String else {
                throw AuraError(message: "The speech server returned invalid audio data.")
            }
            if type == "error" {
                throw AuraError(message: frame["message"] as? String ?? "Speech generation failed.")
            }
            if type == "end" {
                receivedEnd = true
                break
            }
            guard type == "audio", let encoded = frame["wav"] as? String,
                  let wav = Data(base64Encoded: encoded) else {
                throw AuraError(message: "The speech server returned invalid audio data.")
            }
            // Keep the playback queue bounded for long selections.
            while queuedSeconds > 20 {
                try await Task.sleep(nanoseconds: 100_000_000)
                guard playbackID == id else { throw CancellationError() }
            }
            let buffer = try WAVDecoder.decode(wav)
            let isFirstBuffer = !streamStarted
            if isFirstBuffer {
                engine.disconnectNodeOutput(player)
                engine.connect(player, to: engine.mainMixerNode, format: buffer.format)
                installAudioMeter()
                try engine.start()
                streamStarted = true
            }
            let duration = Double(buffer.frameLength) / buffer.format.sampleRate
            queuedBuffers += 1
            queuedSeconds += duration
            player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
                Task { @MainActor in
                    guard let self, self.playbackID == id else { return }
                    self.queuedBuffers -= 1
                    self.queuedSeconds = max(0, self.queuedSeconds - duration)
                    self.finishStreamIfNeeded()
                }
            }
            if isFirstBuffer {
                player.play()
                onStart()
            }
        }
        guard receivedEnd, streamStarted else {
            throw AuraError(message: "Speech generation ended before audio was complete.")
        }
        streamComplete = true
        finishStreamIfNeeded()
    }

    private func finishStreamIfNeeded() {
        guard streamComplete && queuedBuffers == 0 else { return }
        streamComplete = false
        streamStarted = false
        removeAudioMeter()
        engine.stop()
        onFinished?()
    }

    func play(wav: Data) throws {
        let buffer = try WAVDecoder.decode(wav)
        stop()
        let id = UUID()
        playbackID = id
        engine.disconnectNodeOutput(player)
        engine.connect(player, to: engine.mainMixerNode, format: buffer.format)
        installAudioMeter()
        try engine.start()
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.playbackID == id else { return }
                self.removeAudioMeter()
                self.engine.stop()
                self.onFinished?()
            }
        }
        player.play()
    }

    func audition(_ url: URL) throws {
        let file = try AVAudioFile(forReading: url)
        guard file.length > 0, file.length <= Int64(UInt32.max),
              let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw AuraError(message: "The voice sample is empty or too large.")
        }
        try file.read(into: buffer)
        stop()
        let id = UUID()
        playbackID = id
        engine.disconnectNodeOutput(player)
        engine.connect(player, to: engine.mainMixerNode, format: buffer.format)
        installAudioMeter()
        try engine.start()
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.playbackID == id else { return }
                self.removeAudioMeter()
                self.engine.stop()
                self.onFinished?()
            }
        }
        player.play()
    }

    func pause() {
        if player.isPlaying {
            player.pause()
        }
    }

    func resume() {
        if !player.isPlaying {
            player.play()
        }
    }

    func stop() {
        playbackID = UUID()
        player.stop()
        removeAudioMeter()
        engine.stop()
        queuedBuffers = 0
        queuedSeconds = 0
        streamComplete = false
        streamStarted = false
    }

    private func installAudioMeter() {
        guard !meterInstalled else { return }
        let id = playbackID
        player.installTap(onBus: 0, bufferSize: 2048, format: nil) { [weak self] buffer, _ in
            guard let channel = buffer.floatChannelData?[0], buffer.frameLength > 0 else { return }
            let count = Int(buffer.frameLength)
            var energy: Float = 0
            for index in stride(from: 0, to: count, by: 4) {
                let sample = channel[index]
                energy += sample * sample
            }
            let level = sqrt(energy / Float((count + 3) / 4))
            Task { @MainActor [weak self] in
                guard let self, self.playbackID == id else { return }
                self.onAudioLevel?(level)
            }
        }
        meterInstalled = true
    }

    private func removeAudioMeter() {
        guard meterInstalled else { return }
        player.removeTap(onBus: 0)
        meterInstalled = false
    }
}

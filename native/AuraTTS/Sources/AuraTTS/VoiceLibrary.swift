import Foundation

struct VoiceEntry: Identifiable, Hashable {
    var id: String { name }
    let name: String
    var audio: URL?
    var pocketCache: URL?
    var qwenCache: URL?
    var formats: String {
        [audio != nil ? "Reference audio" : nil, pocketCache != nil ? "Pocket cached" : nil,
         qwenCache != nil ? "Qwen cached" : nil].compactMap { $0 }.joined(separator: " · ")
    }

    static let cacheDirectory = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".cache/tts-ext/voices", isDirectory: true)

    static func load(repository: URL) -> [VoiceEntry] {
        var entries: [String: VoiceEntry] = [:]
        let local = repository.appendingPathComponent("voices")
        for url in (try? FileManager.default.contentsOfDirectory(at: local, includingPropertiesForKeys: nil)) ?? [] {
            guard ["wav", "mp3", "flac"].contains(url.pathExtension.lowercased()) else { continue }
            let name = url.deletingPathExtension().lastPathComponent
            entries[name] = VoiceEntry(name: name, audio: url)
        }
        for url in (try? FileManager.default.contentsOfDirectory(at: cacheDirectory, includingPropertiesForKeys: nil)) ?? [] {
            let stem = url.deletingPathExtension().lastPathComponent
            let parts = stem.components(separatedBy: "__")
            guard parts.count == 2, !parts[1].isEmpty else { continue }
            let kind = parts[0], name = parts[1], ext = url.pathExtension.lowercased()
            guard (kind == "audio" && ext == "wav") || (kind == "pocket" && ext == "safetensors") || (kind == "qwen3" && ext == "pt") else { continue }
            var entry = entries[name] ?? VoiceEntry(name: name)
            switch kind {
            case "audio": entry.audio = url
            case "pocket": entry.pocketCache = url
            default: entry.qwenCache = url
            }
            entries[name] = entry
        }
        // Pocket provides a built-in English voice without a local WAV.
        if entries["alba"] == nil { entries["alba"] = VoiceEntry(name: "alba") }
        return entries.values.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }
}

enum VoiceLibrary {
    static func load(repository: URL) -> [VoiceEntry] {
        VoiceEntry.load(repository: repository)
    }
}

enum YouTubeClip {
    static func isValid(_ string: String) -> Bool {
        guard let url = URL(string: string), ["https", "http"].contains(url.scheme?.lowercased() ?? ""),
              let host = url.host?.lowercased() else { return false }
        return ["youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"].contains(host)
    }

    static func timestamp(_ string: String) -> Int {
        guard let components = URLComponents(string: string) else { return 0 }
        let items = components.queryItems ?? []
        guard let value = (items.first { $0.name == "t" } ?? items.first { $0.name == "start" })?.value else { return 0 }
        if let seconds = Int(value), seconds >= 0 { return seconds }
        guard let regex = try? NSRegularExpression(pattern: "^(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s?)?$"),
              let match = regex.firstMatch(in: value, range: NSRange(value.startIndex..., in: value)) else { return 0 }
        let units = [3600, 60, 1]
        return (1...3).reduce(0) { sum, index in
            guard let range = Range(match.range(at: index), in: value), let number = Int(value[range]), number <= 1_000_000 else { return sum }
            return sum + number * units[index - 1]
        }
    }

    static func safeName(_ name: String) -> String {
        name.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "[^a-z0-9_\\-]", with: "_", options: .regularExpression)
    }
}

struct CloneResult { let output: String; let status: Int32 }

enum VoiceCloner {
    static func run(repository: URL, url: String, name: String, duration: Int) async throws -> CloneResult {
        try await Task.detached(priority: .userInitiated) {
            let script = repository.appendingPathComponent("src/tts/clone_voice.py")
            guard FileManager.default.fileExists(atPath: script.path) else {
                throw AuraError(message: "Choose the tts-ext repository folder in Settings first.")
            }
            let process = Process()
            process.currentDirectoryURL = repository
            var environment = ProcessInfo.processInfo.environment
            let home = FileManager.default.homeDirectoryForCurrentUser.path
            environment["PATH"] = "\(home)/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + (environment["PATH"] ?? "")
            environment["PYTHONUNBUFFERED"] = "1"
            process.environment = environment

            // Python interpreter discovery: check virtualenv first
            let candidatePythons = [
                FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".virtualenvs/tts/bin/python"),
                repository.appendingPathComponent(".venv/bin/python"),
                URL(fileURLWithPath: "/opt/homebrew/bin/python3"),
                URL(fileURLWithPath: "/usr/local/bin/python3")
            ]

            if let validPython = candidatePythons.first(where: { FileManager.default.isExecutableFile(atPath: $0.path) }) {
                process.executableURL = validPython
                process.arguments = [script.path, url, "--name", name, "--duration", String(duration)]
            } else {
                process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
                process.arguments = ["python3", script.path, url, "--name", name, "--duration", String(duration)]
            }

            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            return CloneResult(output: String(decoding: data, as: UTF8.self), status: process.terminationStatus)
        }.value
    }
}

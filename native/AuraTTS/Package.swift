// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "AuraTTS",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "AuraTTS", targets: ["AuraTTS"])],
    targets: [
        .executableTarget(name: "AuraTTS"),
        .testTarget(name: "AuraTTSTests", dependencies: ["AuraTTS"])
    ]
)

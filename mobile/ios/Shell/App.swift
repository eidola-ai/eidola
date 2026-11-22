// The app: one WKWebView on the shell's origin, the bridge, and the native bundle sheet.
import UIKit
import WebKit
import os

let log = Logger(subsystem: "ai.eidola.shell", category: "shell")
let digestShortcut = "ai.eidola.shell.digest"

@main
final class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(
        _ application: UIApplication, configurationForConnecting session: UISceneSession,
        options: UIScene.ConnectionOptions
    ) -> UISceneConfiguration {
        let config = UISceneConfiguration(name: nil, sessionRole: session.role)
        config.delegateClass = SceneDelegate.self
        return config
    }
}

final class SceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?
    let shell = ShellViewController()

    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options: UIScene.ConnectionOptions) {
        guard let scene = scene as? UIWindowScene else { return }
        let window = UIWindow(windowScene: scene)
        window.rootViewController = shell
        window.makeKeyAndVisible()
        self.window = window
        if options.shortcutItem?.type == digestShortcut { shell.showDigestWhenReady = true }
    }

    func windowScene(
        _ windowScene: UIWindowScene, performActionFor item: UIApplicationShortcutItem,
        completionHandler: @escaping (Bool) -> Void
    ) {
        if item.type == digestShortcut { shell.showDigest() }
        completionHandler(item.type == digestShortcut)
    }
}

final class ShellViewController: UIViewController, WKNavigationDelegate {
    private var webView: WKWebView!
    private let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Eidola", isDirectory: true)
    private lazy var store = try! BundleStore(dir: support.appendingPathComponent("bundles", isDirectory: true))
    private let pipe = Pipe()
    var showDigestWhenReady = false

    /// Blocks every load outside the shell's scheme (and inline `data:`/`blob:` content).
    private static let rules = """
        [{"trigger":{"url-filter":".*"},"action":{"type":"block"}},
         {"trigger":{"url-filter":"^eidola:"},"action":{"type":"ignore-previous-rules"}},
         {"trigger":{"url-filter":"^data:"},"action":{"type":"ignore-previous-rules"}},
         {"trigger":{"url-filter":"^blob:"},"action":{"type":"ignore-previous-rules"}}]
        """

    override func loadView() {
        Durable.base = support.path
        let files = try! Files(dir: support.appendingPathComponent("files", isDirectory: true))
        let config = WKWebViewConfiguration()
        config.setURLSchemeHandler(SchemeHandler(store: store), forURLScheme: Origin.scheme)
        config.websiteDataStore = .nonPersistent()
        config.preferences.isFraudulentWebsiteWarningEnabled = false
        config.userContentController.addScriptMessageHandler(
            Bridge(pipe: pipe, files: files, store: store), contentWorld: .page, name: "eidola")
        #if DEBUG
            config.userContentController.add(ConsoleLog(), name: "log")
            config.userContentController.addUserScript(
                WKUserScript(source: ConsoleLog.script, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        #endif
        webView = WKWebView(frame: .zero, configuration: config)
        #if DEBUG
            webView.isInspectable = true
        #endif
        webView.navigationDelegate = self
        webView.isOpaque = false
        webView.backgroundColor = .systemBackground
        view = webView
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        WKContentRuleListStore.default().compileContentRuleList(
            forIdentifier: "shell-origin-only", encodedContentRuleList: Self.rules
        ) { [weak self] list, error in
            guard let self, let list else { fatalError("content rules: \(String(describing: error))") }
            self.webView.configuration.userContentController.add(list)
            self.webView.load(URLRequest(url: Origin.root))
        }
    }

    // MARK: Navigation

    func webView(
        _ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        let url = action.request.url
        guard url?.scheme == Origin.scheme, url?.host() == Origin.host else { return decisionHandler(.cancel) }
        if action.targetFrame?.isMainFrame == true && !isFragmentNavigation(action) {
            let s = store.selectForDocument()
            pipe.closeAll()
            log.info("document load: bundle \(s.version, privacy: .public) confirmed=\(s.confirmed)")
        }
        decisionHandler(.allow)
    }

    /// A navigation within the current document (a fragment change) loads nothing.
    private func isFragmentNavigation(_ action: WKNavigationAction) -> Bool {
        guard action.navigationType != .reload, let to = action.request.url, to.fragment != nil,
            let from = webView.url
        else { return false }
        func strip(_ u: URL) -> String { u.absoluteString.components(separatedBy: "#")[0] }
        return strip(to) == strip(from)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        if showDigestWhenReady {
            showDigestWhenReady = false
            showDigest()
        }
    }

    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        log.error("web content process terminated; reloading")
        webView.reload()
    }

    // MARK: Bundle sheet

    /// The serving bundle's version and digest, drawn natively, out of the bundle's reach.
    func showDigest() {
        guard presentedViewController == nil else { return }
        let (s, active) = store.servingInfo()
        let digest = s.digest ?? "(computing)"
        let grouped = stride(from: 0, to: digest.count, by: 16).map { i -> String in
            let a = digest.index(digest.startIndex, offsetBy: i)
            return String(digest[a..<digest.index(a, offsetBy: min(16, digest.count - i))])
        }.joined(separator: "\n")
        var lines = [
            "Version: \(s.version) (\(s.confirmed ? "confirmed" : "on trial"))", "", "SHA-256 digest:", grouped, "",
        ]
        if active != s.version { lines.append("Next launch: \(active)") }
        lines.append("Shell \(shellVersion), contract \(contractVersion)")
        let alert = UIAlertController(title: "Running bundle", message: lines.joined(separator: "\n"), preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "Copy digest", style: .default) { _ in UIPasteboard.general.string = s.digest })
        alert.addAction(UIAlertAction(title: "Done", style: .cancel))
        present(alert, animated: true)
    }
}

#if DEBUG
    /// Debug builds only: the page's console in the system log.
    final class ConsoleLog: NSObject, WKScriptMessageHandler {
        func userContentController(_ c: WKUserContentController, didReceive message: WKScriptMessage) {
            log.info("console: \(String(describing: message.body), privacy: .public)")
        }
        static let script = """
            for (const level of ["log", "info", "warn", "error"]) {
              const orig = console[level].bind(console);
              console[level] = (...args) => {
                try { window.webkit.messageHandlers.log.postMessage(level + ": " + args.map(String).join(" ")); } catch (_) {}
                orig(...args);
              };
            }
            addEventListener("error", (e) => console.error("uncaught", e.message, e.filename + ":" + e.lineno));
            addEventListener("unhandledrejection", (e) => console.error("unhandled", String(e.reason)));
            """
    }
#endif

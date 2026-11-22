// The app: one WebView on the shell's origin, the bridge, the native bundle sheet, and the
// padding that keeps the page clear of system bars and the keyboard.
package ai.eidola.shell

import android.app.Activity
import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.content.pm.ShortcutInfo
import android.content.pm.ShortcutManager
import android.content.res.Configuration
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.text.SpannableString
import android.text.style.TypefaceSpan
import android.util.Log
import android.view.ViewGroup
import android.view.Window
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.webkit.ConsoleMessage
import android.webkit.JavascriptInterface
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebChromeClient
import android.webkit.WebMessage
import android.webkit.WebMessagePort
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import java.io.ByteArrayInputStream
import java.io.InputStream

const val TAG = "EidolaShell"
const val ORIGIN = "https://app.eidola.invalid"
const val ACTION_DIGEST = "ai.eidola.shell.SHOW_DIGEST"

const val CSP = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; " +
    "img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'; object-src 'none'; " +
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"

/** Content types by extension; text types carry their charset separately. */
val TYPES = mapOf(
    "html" to "text/html", "js" to "text/javascript", "mjs" to "text/javascript", "wasm" to "application/wasm",
    "css" to "text/css", "json" to "application/json", "map" to "application/json", "txt" to "text/plain",
    "svg" to "image/svg+xml", "png" to "image/png", "jpg" to "image/jpeg", "jpeg" to "image/jpeg",
    "webp" to "image/webp", "ico" to "image/x-icon", "woff2" to "font/woff2", "woff" to "font/woff",
    "ttf" to "font/ttf", "otf" to "font/otf",
)
val CHARSET_TYPES = setOf("html", "js", "mjs", "css", "txt")

fun isOurs(u: Uri) = u.scheme == "https" && u.host == "app.eidola.invalid" && u.port == -1

/** The shell's parts outlive an activity (a recreated one after the renderer died). */
object Shell {
    lateinit var store: BundleStore
    lateinit var files: Files
    val pipe = Pipe()

    fun init(a: Activity) {
        if (::store.isInitialized) return
        Durable.base = a.noBackupFilesDir
        if (a.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) Durable.log = ArrayList()
        store = BundleStore(java.io.File(a.noBackupFilesDir, "bundles"), a.applicationContext.assets)
        files = Files(java.io.File(a.noBackupFilesDir, "files"))
    }
}

class MainActivity : Activity() {
    private lateinit var web: WebView
    private var port: WebMessagePort? = null
    private val bridgeThread = HandlerThread("bridge").apply { start() }
    private lateinit var bridge: Bridge
    private var showDigestWhenReady = false

    override fun onCreate(state: Bundle?) {
        super.onCreate(state)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        Shell.init(this)
        val debuggable = applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0
        WebView.setWebContentsDebuggingEnabled(debuggable)
        val version = packageManager.getPackageInfo(packageName, 0).versionName ?: "0"
        bridge = Bridge(Shell.pipe, Shell.files, Shell.store, version)

        web = WebView(this)
        web.settings.apply {
            javaScriptEnabled = true
            allowFileAccess = false
            allowContentAccess = false
            domStorageEnabled = false
            safeBrowsingEnabled = false
            setGeolocationEnabled(false)
        }
        web.webViewClient = Client()
        if (debuggable) {
            web.webChromeClient = object : WebChromeClient() {
                override fun onConsoleMessage(m: ConsoleMessage): Boolean {
                    Log.i(TAG, "console: ${m.message()} (${m.sourceId()}:${m.lineNumber()})")
                    return true
                }
            }
        }
        web.addJavascriptInterface(Opener(), "eidolaShellAndroid")

        // Edge to edge, the WebView reports no safe areas and does not shrink for the keyboard,
        // so the shell pads its container by the bars, the cutout and the keyboard.
        val night = resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK == Configuration.UI_MODE_NIGHT_YES
        val root = FrameLayout(this)
        root.setBackgroundColor(if (night) 0xff15181c.toInt() else 0xfffbf9f4.toInt())
        root.addView(web)
        root.setOnApplyWindowInsetsListener { v, insets ->
            val i = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout() or WindowInsets.Type.ime())
            v.setPadding(i.left, i.top, i.right, i.bottom)
            WindowInsets.CONSUMED
        }
        setContentView(root)
        val light = WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS or WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS
        window.insetsController?.setSystemBarsAppearance(if (night) 0 else light, light)

        getSystemService(ShortcutManager::class.java)?.dynamicShortcuts = listOf(
            ShortcutInfo.Builder(this, "digest").setShortLabel("Bundle digest")
                .setIntent(Intent(ACTION_DIGEST, null, this, MainActivity::class.java)).build(),
        )
        showDigestWhenReady = intent?.action == ACTION_DIGEST
        web.loadUrl("$ORIGIN/")
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        if (intent.action == ACTION_DIGEST) showDigest()
    }

    override fun onDestroy() {
        // Cleared too, so a reply still pending is dropped rather than posted on a closed port.
        port?.close()
        port = null
        web.destroy()
        bridgeThread.quitSafely()
        super.onDestroy()
    }

    /** `eidolaShellAndroid.open()`: hands the page a fresh message channel. */
    inner class Opener {
        @JavascriptInterface
        fun open() {
            runOnUiThread {
                port?.close()
                val (mine, theirs) = web.createWebMessageChannel()
                mine.setWebMessageCallback(object : WebMessagePort.WebMessageCallback() {
                    override fun onMessage(p: WebMessagePort, m: WebMessage) {
                        val text = m.data ?: return
                        // A reply goes back on its own channel; once that is replaced, it is dropped.
                        bridge.receive(text) { reply -> runOnUiThread { if (port === mine) mine.postMessage(WebMessage(reply)) } }
                    }
                }, Handler(bridgeThread.looper))
                port = mine
                web.postWebMessage(WebMessage("eidola-shell", arrayOf(theirs)), Uri.parse(ORIGIN))
            }
        }
    }

    private inner class Client : WebViewClient() {
        override fun shouldOverrideUrlLoading(v: WebView, r: WebResourceRequest) = !isOurs(r.url)

        override fun shouldInterceptRequest(v: WebView, r: WebResourceRequest): WebResourceResponse {
            val u = r.url
            if (!isOurs(u)) return respond(403, "Forbidden")
            if (r.isForMainFrame) {
                val s = Shell.store.selectForDocument()
                Shell.pipe.closeAll()
                Log.i(TAG, "document load: bundle ${s.version} confirmed=${s.confirmed}")
            }
            if (r.method != "GET") return respond(405, "Method Not Allowed")
            val raw = u.encodedPath ?: "/"
            val path = if (raw == "/") "index.html" else raw.removePrefix("/")
            if (!Names.bundlePath(path)) return respond(404, "Not Found")
            val body = Shell.store.open(path) ?: return respond(404, "Not Found")
            val ext = path.substringAfterLast('.', "").lowercase()
            return respond(200, "OK", TYPES[ext] ?: "application/octet-stream", if (ext in CHARSET_TYPES) "utf-8" else null, body)
        }

        override fun onPageFinished(v: WebView, url: String) {
            if (showDigestWhenReady) {
                showDigestWhenReady = false
                showDigest()
            }
        }

        override fun onRenderProcessGone(v: WebView, detail: RenderProcessGoneDetail): Boolean {
            Log.e(TAG, "renderer gone; recreating")
            (v.parent as? ViewGroup)?.removeView(v)
            recreate()
            return true
        }
    }

    private fun respond(
        status: Int, reason: String, type: String = "text/plain", charset: String? = "utf-8",
        body: InputStream = ByteArrayInputStream(ByteArray(0)),
    ) = WebResourceResponse(
        null, null, status, reason,
        mapOf(
            // Given a type, WebView adds its own header without the charset; this one is whole.
            "Content-Type" to if (charset == null) type else "$type; charset=$charset",
            "Content-Security-Policy" to CSP, "Cache-Control" to "no-store", "X-Content-Type-Options" to "nosniff",
        ),
        body,
    )

    /** The serving bundle's version and digest, drawn natively, out of the bundle's reach. */
    fun showDigest() {
        val (s, active) = Shell.store.servingInfo()
        val digest = s.digest ?: "(computing)"
        val text = buildString {
            append("Version: ${s.version} (${if (s.confirmed) "confirmed" else "on trial"})\n\nSHA-256 digest:\n")
            append(digest.chunked(16).joinToString("\n")).append("\n\n")
            if (active != s.version) append("Next launch: $active\n")
            append("Shell ${packageManager.getPackageInfo(packageName, 0).versionName}, contract $CONTRACT_VERSION")
        }
        val styled = SpannableString(text)
        styled.setSpan(TypefaceSpan(Typeface.MONOSPACE), 0, text.length, 0)
        AlertDialog.Builder(this).setTitle("Running bundle").setMessage(styled)
            .setNeutralButton("Copy digest") { _, _ ->
                getSystemService(ClipboardManager::class.java).setPrimaryClip(ClipData.newPlainText("bundle digest", s.digest))
            }
            .setPositiveButton("Done", null).show()
    }
}

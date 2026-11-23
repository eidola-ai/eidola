// Drives the app's phone interface in the Simulator with real taps and typing: a development
// bundle (mobile/build-bundle.sh --dev-backend) whose default agent answers from
// mobile/tools/openai_stub.py on this machine, so a whole conversation runs through the shell's
// pipe and app-core's real turn. The bundle mirrors its probes into the page as labelled,
// touch-transparent elements; this test finds them by probe name and taps their centres, which
// land on the app's canvas. Run by ios/test-ui.sh, which also records the screen.
import XCTest

final class PhoneUITests: XCTestCase {
    private let shots = ProcessInfo.processInfo.environment["SHOTS"]
    private var app: XCUIApplication!

    override func setUp() {
        continueAfterFailure = false
        app = XCUIApplication()
    }

    private func shot(_ name: String) {
        let png = XCUIScreen.main.screenshot().pngRepresentation
        let attachment = XCTAttachment(data: png, uniformTypeIdentifier: "public.png")
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        if let shots { try? png.write(to: URL(fileURLWithPath: shots).appendingPathComponent("\(name).png")) }
    }

    /// The mirrored probe named `name`.
    private func probe(_ name: String) -> XCUIElement {
        app.webViews.firstMatch.descendants(matching: .any)
            .matching(NSPredicate(format: "label == %@", name)).firstMatch
    }

    private func probes(prefix: String) -> [XCUIElement] {
        app.webViews.firstMatch.descendants(matching: .any)
            .matching(NSPredicate(format: "label BEGINSWITH %@", prefix)).allElementsBoundByIndex
    }

    @discardableResult
    private func waitFor(_ name: String, timeout: TimeInterval = 30) -> XCUIElement {
        let element = probe(name)
        if !element.waitForExistence(timeout: timeout) {
            shot("failure-\(name.replacingOccurrences(of: "/", with: "-"))")
            XCTFail("no \(name)")
        }
        return element
    }

    private func waitGone(_ name: String, timeout: TimeInterval = 60) {
        let gone = NSPredicate(format: "exists == false")
        let expectation = XCTNSPredicateExpectation(predicate: gone, object: probe(name))
        XCTAssertEqual(XCTWaiter().wait(for: [expectation], timeout: timeout), .completed, "\(name) stayed")
    }

    /// Tap a probe's centre: the mirror lets the touch through to the canvas.
    private func tap(_ name: String) {
        waitFor(name).coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
    }

    /// The transcript's visible band: below the header, above the composer.
    private func visibleBand() -> ClosedRange<CGFloat> {
        let top = probe("conversation/back").frame.maxY + 8
        let field = probe("composer/field")
        let bottom = field.exists ? field.frame.minY - 8 : app.frame.maxY - 120
        return top...max(top + 1, bottom)
    }

    /// Drag the transcript by `dy` points (positive: content moves down, toward the start),
    /// in strokes that stay inside the transcript (a stroke from the bottom edge is the home
    /// gesture).
    private func drag(_ dy: CGFloat) {
        var left = dy
        while abs(left) > 1 {
            let band = visibleBand()
            let stroke = max(-(band.upperBound - band.lowerBound) * 0.8,
                             min((band.upperBound - band.lowerBound) * 0.8, left))
            let midY = (band.lowerBound + band.upperBound) / 2
            let window = app.windows.firstMatch
            let origin = window.coordinate(withNormalizedOffset: .zero)
            let start = origin.withOffset(CGVector(dx: window.frame.midX, dy: midY - stroke / 2))
            let end = origin.withOffset(CGVector(dx: window.frame.midX, dy: midY + stroke / 2))
            start.press(forDuration: 0.05, thenDragTo: end, withVelocity: .slow, thenHoldForDuration: 0.3)
            left -= stroke
        }
    }

    /// Scroll a transcript probe into the visible band, then tap it. A row the list has not
    /// laid out yet is found by scrolling toward the start (or, with `towardEnd`, the end).
    private func tapInTranscript(_ name: String, towardEnd: Bool = false) {
        let element = probe(name)
        for _ in 0..<15 where !element.exists {
            drag(towardEnd ? -500 : 500)
        }
        XCTAssertTrue(element.waitForExistence(timeout: 5), "no \(name)")
        for _ in 0..<12 {
            let band = visibleBand()
            let y = element.frame.midY
            if band.contains(y) { break }
            drag(y < band.lowerBound ? min(400, band.lowerBound - y + 80) : -min(400, y - band.upperBound + 80))
        }
        element.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).tap()
    }

    /// The mean luminance (0 dark … 1 light) of a strip of the screen just under the status bar's
    /// height, where every screen paints only its paper: the palette in force.
    private func paperLuminance() -> CGFloat {
        let image = XCUIScreen.main.screenshot().image
        guard let cg = image.cgImage else { return -1 }
        let width = cg.width, height = cg.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        context?.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        // A row near the bottom, above the home indicator: paper, under no text.
        let y = height * 9 / 10
        var sum: CGFloat = 0
        var count: CGFloat = 0
        for x in stride(from: width / 10, to: width * 9 / 10, by: 7) {
            let i = (y * width + x) * 4
            sum += (0.2126 * CGFloat(pixels[i]) + 0.7152 * CGFloat(pixels[i + 1]) + 0.0722 * CGFloat(pixels[i + 2])) / 255
            count += 1
        }
        return sum / count
    }

    /// Scroll a screen's body (not a transcript) by `dy` points (positive: toward the start), in
    /// strokes that stay well inside the window (a stroke from an edge is a system gesture).
    private func scrollBody(_ dy: CGFloat) {
        let window = app.windows.firstMatch
        let origin = window.coordinate(withNormalizedOffset: .zero)
        let midY = window.frame.midY
        let most = window.frame.height * 0.5
        var left = dy
        while abs(left) > 1 {
            let stroke = max(-most, min(most, left))
            origin.withOffset(CGVector(dx: window.frame.midX, dy: midY - stroke / 2))
                .press(forDuration: 0.05,
                       thenDragTo: origin.withOffset(CGVector(dx: window.frame.midX, dy: midY + stroke / 2)),
                       withVelocity: .slow, thenHoldForDuration: 0.3)
            left -= stroke
        }
    }

    /// Tap a probe on a scrolling screen, scrolling toward the end until it is in view.
    private func tapScrolling(_ name: String) {
        let element = probe(name)
        let window = app.windows.firstMatch
        for _ in 0..<10 {
            if element.exists && element.frame.midY < window.frame.maxY - 60 && element.frame.midY > 120 { break }
            scrollBody(-300)
        }
        tap(name)
    }

    /// The first run, on a fresh install, as far as the choice of an account.
    private func throughTheIntroduction(shots prefix: String?) {
        waitFor("onboarding/cta/understood", timeout: 60)
        if let prefix { shot("\(prefix)-about") }
        tap("onboarding/cta/understood")
        waitFor("onboarding/cta/understood")
        if let prefix { shot("\(prefix)-responsibility") }
        tap("onboarding/cta/understood")
        waitFor("onboarding/cta/new-account")
        if let prefix { shot("\(prefix)-get-started") }
    }

    private func keyboardUp() -> Bool {
        let up = app.keyboards.firstMatch.waitForExistence(timeout: 10)
        // The first keyboard of a fresh simulator teaches swipe typing over itself.
        let tutorial = app.buttons.matching(NSPredicate(format: "label == 'Continue' OR label == '继续'")).firstMatch
        if tutorial.waitForExistence(timeout: 1) { tutorial.tap() }
        return up
    }

    /// Put the keyboard away: a tap on the conversation does.
    private func keyboardDown() {
        let band = visibleBand()
        let window = app.windows.firstMatch
        window.coordinate(withNormalizedOffset: .zero)
            .withOffset(CGVector(dx: window.frame.maxX - 12, dy: band.lowerBound + 4)).tap()
        sleep(1)
    }

    /// Send `text` from the composer and wait for the answer to stream and land.
    private func send(_ text: String, answer: Int) {
        tap("composer/field")
        XCTAssertTrue(keyboardUp(), "the keyboard did not come up")
        app.typeText(text)
        tap("composer/send")
        waitFor("conversation/post/\(answer)", timeout: 120)
    }

    func testConversation() throws {
        XCUIDevice.shared.appearance = .light
        app.launch()

        // The Library (testOnboardingAccount set the account up on this install, and ran first).
        waitFor("library/new", timeout: 60)
        shot("01-library-empty")

        // A new conversation, the keyboard up over an empty page.
        tap("library/new")
        waitFor("composer/field")
        tap("composer/field")
        XCTAssertTrue(keyboardUp(), "the keyboard did not come up")
        shot("02-empty-keyboard-up")

        // Typing, then a streaming answer with its reasoning open.
        app.typeText("How do tides work, briefly?")
        shot("03-typed")
        tap("composer/send")
        waitFor("conversation/post/0", timeout: 30)
        waitFor("conversation/streaming/0", timeout: 30)
        keyboardDown()
        tapInTranscript("conversation/row/1/thinking")
        shot("04-streaming-with-reasoning")

        // The answer lands as a post: markdown (heading, list, code, table).
        waitGone("conversation/streaming/0", timeout: 120)
        waitFor("conversation/post/1")
        XCTAssertFalse(probe("conversation/error").exists, "the turn failed")
        drag(700)
        shot("05-markdown-answer-top")
        drag(-1400)
        shot("05-markdown-answer-end")

        // A reply to the first post starts a branch: the answer shows "2 of 2".
        tapInTranscript("conversation/post/0/reply")
        XCTAssertTrue(keyboardUp())
        waitFor("composer/cancel-reply")
        shot("06-replying")
        app.typeText("And why are there two a day?")
        tap("composer/send")
        waitFor("conversation/post/1/branch-previous", timeout: 30)
        waitGone("conversation/streaming/1", timeout: 120)
        keyboardDown()
        tapInTranscript("conversation/post/1/branch-previous")
        shot("07-branch-switched")
        XCTAssertTrue(probe("conversation/post/1/branch-next").exists)
        tapInTranscript("conversation/post/1/branch-next")
        shot("08-branch-back")

        // A long conversation, scrolled.
        // Back while an answer streams, and straight back in: the same conversation, its answer
        // still streaming, and landing.
        tap("composer/field")
        XCTAssertTrue(keyboardUp(), "the keyboard did not come up")
        app.typeText("Tell me more.")
        tap("composer/send")
        waitFor("conversation/streaming/2", timeout: 30)
        tap("conversation/back")
        tap("library/row/0")
        waitFor("conversation/streaming/2", timeout: 10)
        shot("08a-rejoined-while-streaming")
        waitGone("conversation/streaming/2", timeout: 120)
        waitFor("conversation/post/4")
        send("And the moon?", answer: 6)
        keyboardDown()
        drag(900)
        drag(900)
        shot("09-long-scrolled-up")
        drag(-900)
        drag(-900)
        drag(-900)
        shot("10-long-at-the-end")

        // Back to the Library: the conversation is listed.
        tap("conversation/back")
        waitFor("library/row/0")
        shot("11-library-day")

        // A new conversation left untouched is not kept.
        tap("library/new")
        waitFor("composer/field")
        tap("conversation/back")
        waitFor("library/row/0")
        sleep(1)
        XCTAssertFalse(probe("library/row/1").exists, "an untouched conversation was kept")
    }

    /// Chinese through the system's Pinyin keyboard, on a simulator whose language is Chinese
    /// (test-ui.sh runs it on "eidola-phone-zh", where Pinyin is the first keyboard and the app
    /// speaks zh-Hans): the marked text of a composition, a candidate chosen, the words sent
    /// and answered.
    func testChineseInput() throws {
        app.launch()
        // A fresh install opens on the first run, in the app's zh-Hans words; going on without
        // an account leads to the Library.
        throughTheIntroduction(shots: "60-zh-onboarding")
        tap("onboarding/cta/skip-account")
        waitFor("library/new", timeout: 60)
        shot("15-library-zh")
        // The other screens, in Chinese.
        tap("library/settings")
        waitFor("settings/account")
        shot("61-zh-settings")
        tap("settings/account")
        waitFor("account/set-up")
        shot("62-zh-account-none")
        tap("account/back")
        tap("settings/appearance")
        waitFor("appearance/day-night/system")
        shot("63-zh-appearance")
        tap("appearance/back")
        tap("settings/updates")
        waitFor("updates/check")
        sleep(2)
        shot("64-zh-updates")
        tap("updates/back")
        tap("settings/about")
        waitFor("about/licences")
        shot("65-zh-about")
        tap("about/back")
        tap("settings/back")
        waitFor("library/new")
        tap("library/new")
        tap("composer/field")
        XCTAssertTrue(keyboardUp(), "the keyboard did not come up")
        for key in ["n", "i", "h", "a", "o"] { app.keyboards.keys[key].tap() }
        sleep(1)
        shot("16-ime-composing")
        let candidate = app.descendants(matching: .any).matching(NSPredicate(format: "label == '你好'")).firstMatch
        XCTAssertTrue(candidate.waitForExistence(timeout: 5), "no 你好 candidate")
        candidate.tap()
        sleep(1)
        shot("17-ime-committed")
        tap("composer/send")
        waitFor("conversation/post/1", timeout: 120)
        keyboardDown()
        drag(3000)
        shot("18-chinese-answer")
        tap("conversation/back")
        waitFor("library/row/0")
    }

    /// Run after `testConversation` with the simulator switched to dark (test-ui.sh): a cold
    /// launch finds the conversation, in the night palette.
    func testRelaunchAtNight() throws {
        app.launch()
        waitFor("library/row/0", timeout: 60)
        shot("12-library-night")
        tap("library/row/0")
        // A conversation reopens at its newest post.
        let anyPost = app.webViews.firstMatch.descendants(matching: .any)
            .matching(NSPredicate(format: "label BEGINSWITH 'conversation/post/'")).firstMatch
        XCTAssertTrue(anyPost.waitForExistence(timeout: 30), "the conversation did not reopen")
        sleep(1)
        shot("13-conversation-night-end")
        drag(3000)
        shot("14-conversation-night-start")
        // The other screens at night.
        tap("conversation/back")
        tap("library/settings")
        waitFor("settings/account")
        shot("70-night-settings")
        tap("settings/account")
        waitFor("account/balance", timeout: 60)
        shot("71-night-account")
        tapScrolling("account/purchase")
        waitFor("purchase/plan/0", timeout: 60)
        shot("72-night-plans")
        tap("purchase/back")
        tap("account/back")
        tap("settings/appearance")
        waitFor("appearance/day-night/system")
        shot("73-night-appearance")
        tap("appearance/back")
        tap("settings/updates")
        waitFor("updates/check")
        sleep(3)
        shot("74-night-updates")
        tap("updates/back")
        tap("settings/about")
        waitFor("about/licences")
        shot("75-night-about")
    }

    /// A fresh install's first run against the local dev stack's real gateway (behind its
    /// attesting shim): the introduction, a new account created under the terms snapshot the
    /// consent page rendered, its credentials, the plans, a checkout handed to the system
    /// browser, and back in the app the balance read again with the purchase in it.
    func testOnboardingAccount() throws {
        XCUIDevice.shared.appearance = .light
        app.launch()
        throughTheIntroduction(shots: "20-onboarding")
        tap("onboarding/cta/new-account")
        // The documents this account will accept, as the gateway lists them.
        waitFor("onboarding/terms/terms-of-service", timeout: 60)
        waitFor("onboarding/terms/privacy-policy")
        shot("23-onboarding-consent")
        // Not agreed: the Create button is off, and a tap on it creates nothing.
        tapScrolling("onboarding/cta/create")
        sleep(2)
        XCTAssertFalse(probe("onboarding/copy/account-id/copy").exists, "created without agreement")
        tapScrolling("onboarding/agree")
        shot("24-onboarding-agreed")
        tapScrolling("onboarding/cta/create")
        waitFor("onboarding/copy/account-id/copy", timeout: 90)
        waitFor("onboarding/copy/account-secret/copy")
        shot("25-onboarding-new-account")
        tapScrolling("onboarding/cta/saved")
        waitFor("onboarding/cta/plans")
        shot("26-onboarding-add-credit")
        tap("onboarding/cta/plans")
        waitFor("purchase/plan/0", timeout: 60)
        shot("27-plans")

        // Checkout leaves for the system browser.
        tap("purchase/plan/0")
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        XCTAssertTrue(safari.wait(for: .runningForeground, timeout: 30), "the browser did not open")
        sleep(4)
        shot("28-checkout-in-the-browser")
        XCTAssertEqual(app.state, .runningBackground)
        app.activate()
        waitFor("purchase/opened", timeout: 30)
        // Back in the foreground, the balance is read again: the purchase is in it.
        sleep(4)
        shot("29-back-from-the-browser")

        tap("purchase/back")
        waitFor("onboarding/cta/purchase-later")
        tap("onboarding/cta/purchase-later")
        waitFor("library/settings", timeout: 30)
        shot("30-library-after-the-first-run")

        // The account: its credentials, and the balance read again since the browser, with the
        // purchase in it as a pool of its own.
        tap("library/settings")
        waitFor("settings/account")
        shot("31-settings")
        tap("settings/account")
        waitFor("account/balance", timeout: 60)
        waitFor("account/pool/0", timeout: 30)
        shot("32-account")
        tap("account/secret/reveal")
        sleep(1)
        shot("33-account-secret-shown")
        tap("account/secret/reveal")
        tapScrolling("account/reset")
        waitFor("account/reset/warning")
        shot("34-account-reset-armed")
        tapScrolling("account/reset/keep")
        waitFor("account/reset")
        tap("account/back")
        tap("settings/back")
        waitFor("library/new")
    }

    /// Settings that apply at once: the night palette from Appearance, the text size; then the
    /// wallet, About and the licences.
    func testSettings() throws {
        XCUIDevice.shared.appearance = .light
        app.launch()
        waitFor("library/settings", timeout: 60)
        tap("library/settings")
        tap("settings/appearance")
        waitFor("appearance/day-night/night")
        sleep(1)
        let day = paperLuminance()
        shot("40-appearance-day")
        tap("appearance/day-night/night")
        sleep(2)
        let night = paperLuminance()
        shot("41-appearance-night-live")
        XCTAssertGreaterThan(day, 0.6, "not the day palette before")
        XCTAssertLessThan(night, 0.3, "the night palette did not apply live")
        tap("appearance/text-size/larger")
        tap("appearance/text-size/larger")
        sleep(1)
        shot("42-appearance-larger-text-night")
        tap("appearance/text-size/reset")
        tap("appearance/day-night/system")
        sleep(2)
        XCTAssertGreaterThan(paperLuminance(), 0.6, "System did not follow the light simulator")
        tap("appearance/back")
        tap("settings/wallet")
        waitFor("wallet/refresh")
        sleep(2)
        shot("43-wallet")
        tap("wallet/back")
        tap("settings/about")
        waitFor("about/licences")
        sleep(2)
        shot("44-about")
        scrollBody(-600)
        shot("45-about-end")
        tapScrolling("about/licences")
        waitFor("licences/row/0", timeout: 30)
        shot("46-licences")
        tap("licences/row/0")
        sleep(1)
        shot("47-licence-text")
        tap("licences/back")
        tap("about/back")
        tap("settings/back")
        waitFor("library/new")
    }

    /// The verified update, approved here: the check offers a release (a fixture the harness
    /// serves, taken as verified: no release recording a bundle can be signed outside a
    /// release); nothing is staged until the approval; approved, it is downloaded, checked,
    /// staged and activated, and the app restarts into it and confirms it. Then an update that
    /// never starts: the next launch is back on the confirmed one, and says so. Then the way
    /// back to the bundle built into the app.
    func testUpdate() throws {
        XCUIDevice.shared.appearance = .light
        app.launch()
        waitFor("library/settings", timeout: 60)
        tap("library/settings")
        tap("settings/updates")
        waitFor("updates/serving/embedded", timeout: 30)
        if probe("updates/accept-claims").waitForExistence(timeout: 30) {
            shot("50-update-claims-changed")
            tapScrolling("updates/accept-claims")
        }
        waitFor("updates/approve", timeout: 30)
        shot("51-update-available")
        // The approval gate: offered and verified, but nothing is staged until the tap.
        sleep(3)
        XCTAssertTrue(probe("updates/serving/embedded").exists)
        tapScrolling("updates/approve")
        // Staged, activated, the app reloaded into it; its first frame confirmed it.
        waitFor("library/settings", timeout: 120)
        tap("library/settings")
        tap("settings/updates")
        waitFor("updates/serving/update-confirmed", timeout: 30)
        sleep(2)
        scrollBody(-1500)
        shot("52-update-running-confirmed")

        // The next release never starts.
        scrollBody(1500)
        if probe("updates/accept-claims").waitForExistence(timeout: 30) {
            tapScrolling("updates/accept-claims")
        }
        waitFor("updates/approve", timeout: 60)
        tapScrolling("updates/approve")
        sleep(8)
        shot("53-broken-update-on-trial")
        app.terminate()
        app.launch()
        waitFor("library/settings", timeout: 60)
        tap("library/settings")
        waitFor("settings/updates")
        shot("54-settings-update-failed")
        tap("settings/updates")
        waitFor("updates/fell-back", timeout: 30)
        waitFor("updates/serving/update-confirmed")
        sleep(2)
        shot("55-update-fell-back")
        scrollBody(-6000)
        shot("55b-update-fell-back-running")

        // Back to the bundle built into the app.
        tapScrolling("updates/revert")
        waitFor("library/settings", timeout: 60)
        tap("library/settings")
        tap("settings/updates")
        waitFor("updates/serving/embedded", timeout: 30)
        sleep(2)
        scrollBody(-1500)
        shot("56-reverted-to-embedded")
    }
}

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

        // The Library of a fresh install.
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
        waitFor("library/new", timeout: 60)
        shot("15-library-zh")
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
    }
}

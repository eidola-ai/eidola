// Drives the contract test page in the Simulator: the page's own verdict, the native bundle
// sheet reached from the home-screen quick action, and a cold relaunch.
import XCTest

final class ContractUITests: XCTestCase {
    private let shots = ProcessInfo.processInfo.environment["SHOTS"]

    private func shot(_ name: String) {
        let png = XCUIScreen.main.screenshot().pngRepresentation
        let attachment = XCTAttachment(data: png, uniformTypeIdentifier: "public.png")
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        if let shots { try? png.write(to: URL(fileURLWithPath: shots).appendingPathComponent("\(name).png")) }
    }

    private func text(_ app: XCUIApplication, prefix: String) -> XCUIElement {
        app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH %@", prefix)).firstMatch
    }

    func testContract() throws {
        let app = XCUIApplication()
        app.launch()

        // The run ends by handing an https URL to the system browser: Safari comes to the front.
        let safari = XCUIApplication(bundleIdentifier: "com.apple.mobilesafari")
        XCTAssertTrue(safari.wait(for: .runningForeground, timeout: 240), "shell.openExternal did not open Safari")
        shot("0-open-external")
        app.activate()

        let verdict = app.staticTexts.matching(
            NSPredicate(format: "label BEGINSWITH 'contract: PASS' OR label BEGINSWITH 'contract: FAIL'")
        ).firstMatch
        XCTAssertTrue(verdict.waitForExistence(timeout: 60), "the page reached no verdict")
        shot("1-contract")
        XCTAssertTrue(verdict.label.hasPrefix("contract: PASS"), verdict.label)

        let served = text(app, prefix: "serving embedded, digest ")
        XCTAssertTrue(served.exists)
        let digest = String(served.label.dropFirst("serving embedded, digest ".count))
        XCTAssertEqual(digest.count, 64)

        // The bundle sheet, from a long press on the home-screen icon.
        XCUIDevice.shared.press(.home)
        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let icon = springboard.icons["Eidola"]
        XCTAssertTrue(icon.waitForExistence(timeout: 10))
        icon.press(forDuration: 1.5)
        let action = springboard.buttons["Bundle digest"]
        XCTAssertTrue(action.waitForExistence(timeout: 10))
        shot("2-quick-action")
        action.tap()
        let sheet = app.alerts["Running bundle"]
        XCTAssertTrue(sheet.waitForExistence(timeout: 10))
        shot("3-bundle-sheet")
        let shown = sheet.staticTexts.allElementsBoundByIndex.map(\.label).joined(separator: "\n")
        XCTAssertTrue(shown.replacingOccurrences(of: "\n", with: "").contains(digest), shown)
        XCTAssertTrue(shown.contains("Version: embedded (confirmed)"), shown)
        sheet.buttons["Done"].tap()

        // A cold relaunch: the synced file survived the process.
        app.terminate()
        app.launch()
        let restart = text(app, prefix: "restart: ")
        XCTAssertTrue(restart.waitForExistence(timeout: 30))
        shot("4-restart")
        XCTAssertTrue(restart.label.hasPrefix("restart: PASS"), restart.label)
    }
}

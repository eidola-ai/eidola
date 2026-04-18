import AppKit
import Foundation
import Testing

@testable import MarkdownEditor

/// Kitchen sink visual test: a document with ALL implemented features in combination,
/// cursor moved to many "interesting" positions to catch interaction bugs.
@Suite("Kitchen Sink Visual Tests")
@MainActor
struct KitchenSinkVisualTests {

  // The kitchen sink document exercises headings (h1, h2, h3), body text,
  // bold, italic, bold-italic, bold inside a heading, unordered lists,
  // ordered lists, checkbox list items, inline code, links, and blockquotes.
  static let kitchenSinkMarkdown = """
    # Main Heading

    Some body text here.

    ## Second Heading

    Text with **bold words** in it.

    Text with *italic words* in it.

    Text with ***bold italic*** in it.

    ### Third Heading with **bold** inside

    - First list item
    - Second with **bold**
    - Third item

    1. First ordered
    2. Second ordered
    3. Third with **bold**

    - [ ] Unchecked task
    - [x] Completed task
    - [ ] Another **bold** task

    Some `inline code` in a paragraph.

    A [link to example](https://example.com) in text.

    Code and links: `foo` and [bar](https://bar.com).

    ```swift
    let x = 42
    print(x)
    ```

    > A simple blockquote
    > with **bold** inside
    """

  /// Named cursor positions for clarity in artifacts.
  struct CursorPosition {
    let name: String
    let offset: Int
    let description: String
  }

  /// Compute all interesting cursor positions relative to the kitchen sink document.
  static func interestingPositions() -> [CursorPosition] {
    let md = kitchenSinkMarkdown
    let ns = md as NSString

    var positions: [CursorPosition] = []

    // Helper to find offset of a substring
    func offsetOf(_ sub: String) -> Int {
      Int(ns.range(of: sub).location)
    }

    // 1. Middle of body text (all delimiters should be hidden)
    let bodyStart = offsetOf("Some body text here.")
    positions.append(CursorPosition(
      name: "body-middle",
      offset: bodyStart + 10,
      description: "Middle of body text -- all delimiters hidden"))

    // 2. Inside h1 heading content
    let h1Start = offsetOf("# Main Heading")
    positions.append(CursorPosition(
      name: "h1-inside",
      offset: h1Start + 5,  // inside "Main"
      description: "Inside h1 heading content -- # delimiter visible and dimmed"))

    // 3. At end of h1 heading line (just before \n)
    // "# Main Heading" is 14 chars, so end is at offset 14
    positions.append(CursorPosition(
      name: "h1-end-before-newline",
      offset: h1Start + 14,
      description: "End of h1 heading line just before newline -- # should be visible"))

    // 4. On the blank line between h1 and body
    // "# Main Heading\n" is 15 chars, then "\n" is at offset 15
    positions.append(CursorPosition(
      name: "blank-line-after-h1",
      offset: h1Start + 15,
      description: "Blank line between h1 and body -- h1 delimiter hidden"))

    // 5. Inside bold content ("bold words")
    let boldStart = offsetOf("**bold words**")
    positions.append(CursorPosition(
      name: "bold-inside",
      offset: boldStart + 5,  // inside "bold"
      description: "Inside bold content -- ** delimiters visible and dimmed"))

    // 6. Just before opening ** of bold
    positions.append(CursorPosition(
      name: "bold-before-opening",
      offset: boldStart,
      description: "Just before opening ** of bold -- at node start, delimiters visible"))

    // 7. Just after closing ** of bold
    let boldEnd = boldStart + ("**bold words**" as NSString).length
    positions.append(CursorPosition(
      name: "bold-after-closing",
      offset: boldEnd,
      description: "Just after closing ** of bold -- delimiters may be visible or hidden depending on boundary"))

    // 8. Inside italic content
    let italicStart = offsetOf("*italic words*")
    positions.append(CursorPosition(
      name: "italic-inside",
      offset: italicStart + 5,
      description: "Inside italic content -- * delimiters visible and dimmed"))

    // 9. Inside bold-italic content
    let biStart = offsetOf("***bold italic***")
    positions.append(CursorPosition(
      name: "bold-italic-inside",
      offset: biStart + 6,
      description: "Inside bold italic content -- *** delimiters visible and dimmed"))

    // 10. Just before bold-italic (cursor outside)
    positions.append(CursorPosition(
      name: "bold-italic-before",
      offset: biStart - 1,  // space before ***
      description: "Just before bold-italic -- *** delimiters hidden"))

    // 11. Just after bold-italic (cursor outside)
    let biEnd = biStart + ("***bold italic***" as NSString).length
    positions.append(CursorPosition(
      name: "bold-italic-after",
      offset: biEnd,
      description: "Just after bold-italic closing -- *** delimiters should be hidden"))

    // 12. On a line with no formatting (the "Text with" before bold)
    let textWithBold = offsetOf("Text with **bold")
    positions.append(CursorPosition(
      name: "unformatted-prefix",
      offset: textWithBold + 3,
      description: "On text before bold on same line -- bold delimiters should be hidden"))

    // 13. Inside h3 content (which contains bold)
    let h3Start = offsetOf("### Third Heading")
    positions.append(CursorPosition(
      name: "h3-inside",
      offset: h3Start + 8,
      description: "Inside h3 heading content -- ### and ** delimiters visible and dimmed"))

    // 14. Inside the bold within h3
    let boldInH3 = offsetOf("**bold** inside")
    positions.append(CursorPosition(
      name: "h3-bold-inside",
      offset: boldInH3 + 3,
      description: "Inside bold within h3 -- both ### and ** visible and dimmed"))

    // 15. At the very end of the document
    positions.append(CursorPosition(
      name: "document-end",
      offset: ns.length,
      description: "Very end of document -- last construct delimiters visible"))

    // 16. At position 0 (very start)
    positions.append(CursorPosition(
      name: "document-start",
      offset: 0,
      description: "Very start of document -- h1 delimiter visible (cursor at node start)"))

    // 17. Inside h2 heading content
    let h2Start = offsetOf("## Second Heading")
    positions.append(CursorPosition(
      name: "h2-inside",
      offset: h2Start + 6,
      description: "Inside h2 heading content -- ## delimiter visible and dimmed"))

    // 18. Inside first list item content
    let firstListItem = offsetOf("- First list item")
    positions.append(CursorPosition(
      name: "list-first-inside",
      offset: firstListItem + 5,
      description: "Inside first list item -- - delimiter visible and dimmed, bullet NOT shown"))

    // 19. Inside second list item (which has bold)
    let secondListItem = offsetOf("- Second with **bold**")
    positions.append(CursorPosition(
      name: "list-second-bold-inside",
      offset: secondListItem + 5,
      description: "Inside second list item -- - visible, ** visible, all dimmed"))

    // 20. On a body line with cursor outside all list items
    // (all list markers should show bullets)
    positions.append(CursorPosition(
      name: "list-all-outside",
      offset: bodyStart + 5,
      description: "Cursor in body text -- all list items show bullet glyphs"))

    // 21. At start of a list item marker
    positions.append(CursorPosition(
      name: "list-at-marker-start",
      offset: firstListItem,
      description: "Cursor at start of list marker -- delimiter visible (cursor at node start)"))

    // 22. At end of last unordered list item
    let thirdListItem = offsetOf("- Third item")
    positions.append(CursorPosition(
      name: "list-third-end",
      offset: thirdListItem + ("- Third item" as NSString).length,
      description: "Cursor at end of third list item -- delimiter visible"))

    // 23. Inside first ordered list item
    let firstOrderedItem = offsetOf("1. First ordered")
    positions.append(CursorPosition(
      name: "ordered-first-inside",
      offset: firstOrderedItem + 5,
      description: "Inside first ordered list item -- marker always visible, indented"))

    // 24. Inside second ordered list item
    let secondOrderedItem = offsetOf("2. Second ordered")
    positions.append(CursorPosition(
      name: "ordered-second-inside",
      offset: secondOrderedItem + 5,
      description: "Inside second ordered list item -- marker always visible"))

    // 25. Inside third ordered list item (which has bold)
    let thirdOrderedItem = offsetOf("3. Third with **bold**")
    positions.append(CursorPosition(
      name: "ordered-third-bold-inside",
      offset: thirdOrderedItem + 5,
      description: "Inside third ordered item -- marker visible, ** delimiters visible"))

    // 26. At start of ordered list marker
    positions.append(CursorPosition(
      name: "ordered-at-marker-start",
      offset: firstOrderedItem,
      description: "Cursor at start of ordered list marker -- marker always visible"))

    // 27. In body after ordered list (all ordered markers should stay visible)
    positions.append(CursorPosition(
      name: "ordered-all-outside",
      offset: bodyStart + 3,
      description: "Cursor in body -- all ordered list markers visible (no bullets)"))

    // 28. Inside unchecked checkbox item
    let uncheckedItem = offsetOf("- [ ] Unchecked task")
    positions.append(CursorPosition(
      name: "checkbox-unchecked-inside",
      offset: uncheckedItem + 10,
      description: "Inside unchecked checkbox item -- full '- [ ] ' prefix visible and dimmed"))

    // 29. Inside checked checkbox item
    let checkedItem = offsetOf("- [x] Completed task")
    positions.append(CursorPosition(
      name: "checkbox-checked-inside",
      offset: checkedItem + 10,
      description: "Inside checked checkbox item -- full '- [x] ' prefix visible and dimmed"))

    // 30. Inside checkbox item with bold
    let boldCheckbox = offsetOf("- [ ] Another **bold** task")
    positions.append(CursorPosition(
      name: "checkbox-bold-inside",
      offset: boldCheckbox + 10,
      description: "Inside checkbox item with bold -- both checkbox prefix and ** visible"))

    // 31. At start of checkbox marker
    positions.append(CursorPosition(
      name: "checkbox-at-marker-start",
      offset: uncheckedItem,
      description: "Cursor at start of checkbox marker -- prefix visible (cursor at node start)"))

    // 32. Cursor in body -- all checkbox items show checkbox glyphs
    positions.append(CursorPosition(
      name: "checkbox-all-outside",
      offset: bodyStart + 7,
      description: "Cursor in body -- all checkbox items show checkbox glyphs"))

    // 33. Inside inline code content
    let inlineCodeStart = offsetOf("`inline code`")
    positions.append(CursorPosition(
      name: "inline-code-inside",
      offset: inlineCodeStart + 5,
      description: "Inside inline code content -- backtick delimiters visible and dimmed"))

    // 34. Just before opening backtick of inline code
    positions.append(CursorPosition(
      name: "inline-code-before",
      offset: inlineCodeStart - 1,
      description: "Just before inline code -- backtick delimiters hidden"))

    // 35. Just after closing backtick of inline code
    let inlineCodeEnd = inlineCodeStart + ("`inline code`" as NSString).length
    positions.append(CursorPosition(
      name: "inline-code-after",
      offset: inlineCodeEnd,
      description: "Just after inline code closing backtick -- delimiters may be visible"))

    // 36. At start of inline code (on opening backtick)
    positions.append(CursorPosition(
      name: "inline-code-at-start",
      offset: inlineCodeStart,
      description: "At opening backtick of inline code -- delimiters visible (cursor at node start)"))

    // 37. Inside link text content
    let linkStart = offsetOf("[link to example]")
    positions.append(CursorPosition(
      name: "link-inside-text",
      offset: linkStart + 5,
      description: "Inside link text -- [ and ](url) delimiters visible and dimmed"))

    // 38. Just before opening [ of link
    positions.append(CursorPosition(
      name: "link-before",
      offset: linkStart - 1,
      description: "Just before link -- delimiters hidden, only link text visible in blue"))

    // 39. On the URL portion of the link
    let urlInLink = offsetOf("(https://example.com)")
    positions.append(CursorPosition(
      name: "link-on-url",
      offset: urlInLink + 5,
      description: "On URL portion of link -- all delimiters visible"))

    // 40. Just after closing ) of link
    let linkEnd = urlInLink + ("(https://example.com)" as NSString).length
    positions.append(CursorPosition(
      name: "link-after",
      offset: linkEnd,
      description: "Just after link closing ) -- delimiters may be visible"))

    // 41. Inside inline code on the last line (with both code and link)
    let fooCode = offsetOf("`foo`")
    positions.append(CursorPosition(
      name: "mixed-line-code-inside",
      offset: fooCode + 2,
      description: "Inside `foo` inline code on mixed line -- backticks visible, link delimiters hidden"))

    // 42. Inside link on the last line
    let barLink = offsetOf("[bar]")
    positions.append(CursorPosition(
      name: "mixed-line-link-inside",
      offset: barLink + 2,
      description: "Inside [bar] link on mixed line -- link delimiters visible, code backticks hidden"))

    // 43. On opening fence of code block
    let codeBlockFence = offsetOf("```swift")
    positions.append(CursorPosition(
      name: "code-block-opening-fence",
      offset: codeBlockFence,
      description: "At opening fence of code block -- fences visible and dimmed"))

    // 44. Inside code block content (on "let x = 42")
    let codeBlockContent = offsetOf("let x = 42")
    positions.append(CursorPosition(
      name: "code-block-content-inside",
      offset: codeBlockContent + 5,
      description: "Inside code block content -- fences visible and dimmed, monospace font"))

    // 45. On closing fence of code block (the last ``` in the document)
    let lastClosingFence = ns.range(of: "```", options: .backwards)
    positions.append(CursorPosition(
      name: "code-block-closing-fence",
      offset: lastClosingFence.location + 1,
      description: "On closing fence of code block -- fences visible and dimmed"))

    // 46. Just before code block (on blank line before it)
    positions.append(CursorPosition(
      name: "code-block-before",
      offset: codeBlockFence - 1,
      description: "Just before code block -- fences hidden, code content in monospace"))

    // 47. Inside blockquote content
    let blockquoteStart = offsetOf("> A simple blockquote")
    positions.append(CursorPosition(
      name: "blockquote-inside",
      offset: blockquoteStart + 5,
      description: "Inside blockquote content -- > prefix visible and dimmed"))

    // 48. At start of blockquote (on > character)
    positions.append(CursorPosition(
      name: "blockquote-at-start",
      offset: blockquoteStart,
      description: "At > of blockquote -- prefix visible (cursor at node start)"))

    // 49. Inside blockquote second line with bold
    let blockquoteBold = offsetOf("> with **bold** inside")
    positions.append(CursorPosition(
      name: "blockquote-bold-inside",
      offset: blockquoteBold + 5,
      description: "Inside second blockquote line -- > and ** visible and dimmed"))

    // 50. Cursor in body -- blockquote > prefixes should be hidden
    positions.append(CursorPosition(
      name: "blockquote-all-outside",
      offset: bodyStart + 2,
      description: "Cursor in body -- blockquote > prefixes hidden"))

    return positions
  }

  @Test("Kitchen sink: cursor at many positions produces correct rendering")
  func kitchenSinkCursorPositions() {
    let md = Self.kitchenSinkMarkdown
    let positions = Self.interestingPositions()

    // Build events: for each position, set cursor there
    var events: [EditorEvent] = []
    for pos in positions {
      events.append(.setSelection(.cursor(pos.offset)))
    }

    let initial = EditorState(
      markdown: md,
      selection: .cursor(positions[0].offset))

    // Skip the first position since it's the initial state
    let results = EditorTestHarness.run(
      name: "kitchen-sink",
      initial: initial,
      events: Array(events.dropFirst()),
      size: NSSize(width: 700, height: 900))

    // We should have initial + (N-1) events = N total results
    #expect(results.count == positions.count)

    // Every step should produce an image
    let fm = FileManager.default
    for r in results {
      #expect(fm.fileExists(atPath: r.imagePath), "Image missing: \(r.imagePath)")
    }

    // Verify that cursor position actually matters: at least some positions
    // produce different visuals. We expect most to differ from each other.
    var uniqueHashes = Set<Int>()
    for r in results {
      uniqueHashes.insert(r.bitmapHash)
    }
    #expect(
      uniqueHashes.count >= 5,
      "Expected at least 5 visually distinct states, got \(uniqueHashes.count)")

    // Specific expectations:

    // Body text (pos 0) vs h1 inside (pos 1) should differ
    // (heading delimiters visible vs hidden)
    #expect(
      results[0].bitmapHash != results[1].bitmapHash,
      "Body text vs inside h1 should look different")

    // h1 inside (pos 1) vs h1 end before newline (pos 2) should be similar
    // (both show h1 delimiters) -- OR they may differ if cursor position
    // changes the rendering slightly. Just verify both produce images.

    // Bold inside (pos 4) vs bold before opening (pos 5) should show
    // delimiters in both cases (cursor at node start counts as inside)
    // But they should differ from body text (pos 0) where bold is hidden
    #expect(
      results[0].bitmapHash != results[4].bitmapHash,
      "Body text vs inside bold should look different (bold delimiters visible)")

    // Write a summary for review
    var summary = "# Kitchen Sink Test Summary\n\n"
    summary += "Document:\n```\n\(md)\n```\n\n"
    summary += "## Cursor Positions and Results\n\n"
    for (i, pos) in positions.enumerated() {
      let imgFile = URL(fileURLWithPath: results[i].imagePath).lastPathComponent
      summary += "### \(i). \(pos.name) (offset \(pos.offset))\n"
      summary += "\(pos.description)\n\n"
      summary += "Image: `\(imgFile)` | Hash: \(results[i].bitmapHash)\n\n"
    }

    // Count distinct visuals
    summary += "## Visual Diversity\n\n"
    summary += "Total positions: \(positions.count)\n"
    summary += "Unique visuals: \(uniqueHashes.count)\n"

    let dir = results[0].imagePath.components(separatedBy: "/").dropLast().joined(separator: "/")
    try? summary.write(
      toFile: "\(dir)/summary.md", atomically: true, encoding: .utf8)
  }

  @Test("Kitchen sink: bold-italic hides ALL asterisks when cursor outside")
  func boldItalicHidesAllAsterisks() {
    let text = "hello ***bold italic*** world"
    // Cursor clearly outside at position 0
    let cursorRange = NSRange(location: 0, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // All 3 opening asterisks (positions 6,7,8) should be hidden
    #expect(spec.hiddenIndexes.contains(6), "Opening *** position 6 should be hidden")
    #expect(spec.hiddenIndexes.contains(7), "Opening *** position 7 should be hidden")
    #expect(spec.hiddenIndexes.contains(8), "Opening *** position 8 should be hidden")

    // All 3 closing asterisks (positions 20,21,22) should be hidden
    #expect(spec.hiddenIndexes.contains(20), "Closing *** position 20 should be hidden")
    #expect(spec.hiddenIndexes.contains(21), "Closing *** position 21 should be hidden")
    #expect(spec.hiddenIndexes.contains(22), "Closing *** position 22 should be hidden")

    // Content "bold italic" should NOT be hidden
    for i in 9...19 {
      #expect(!spec.hiddenIndexes.contains(i), "Content position \(i) should not be hidden")
    }
  }

  @Test("Kitchen sink: heading delimiter visible at end of line before newline")
  func headingDelimiterVisibleAtEndBeforeNewline() {
    let text = "# Hello\nBody text"
    // Cursor at position 7 (end of "Hello", just before \n)
    let cursorRange = NSRange(location: 7, length: 0)
    let spec = MarkdownRenderer.render(text: text, cursorRange: cursorRange)

    // The # delimiter should be visible (not hidden) because cursor is on the heading line
    #expect(
      spec.hiddenIndexes.isEmpty,
      "Heading delimiter should be visible when cursor is at end of heading line before newline, but hiddenIndexes = \(spec.hiddenIndexes)")

    // Should have temporary attributes for dimmed delimiter
    #expect(
      !spec.temporaryAttributes.isEmpty,
      "Should have dimmed delimiter temp attrs")
  }
}

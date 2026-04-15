import Foundation
import Markdown

/// Converts swift-markdown `SourceRange` (line:column with UTF-8 byte columns)
/// to Foundation `NSRange` (UTF-16 code unit offsets).
struct SourceRangeConverter: Sendable {
  /// UTF-16 offset of the start of each line (0-indexed by line number - 1).
  private let lineStartOffsets: [Int]
  let string: String

  init(string: String) {
    self.string = string
    var offsets: [Int] = [0]
    var utf16Offset = 0
    for char in string {
      utf16Offset += char.utf16.count
      if char == "\n" {
        offsets.append(utf16Offset)
      }
    }
    self.lineStartOffsets = offsets
  }

  /// Convert a `SourceRange` to an `NSRange`.
  func nsRange(from sourceRange: SourceRange) -> NSRange {
    let start = utf16Offset(from: sourceRange.lowerBound)
    let end = utf16Offset(from: sourceRange.upperBound)
    return NSRange(location: start, length: end - start)
  }

  /// Convert a `SourceLocation` (1-based line, 1-based UTF-8 byte column) to a UTF-16 offset.
  func utf16Offset(from location: SourceLocation) -> Int {
    let lineIndex = location.line - 1
    guard lineIndex >= 0, lineIndex < lineStartOffsets.count else { return 0 }
    let lineStart = lineStartOffsets[lineIndex]
    let utf8Column = location.column - 1

    // Find the substring for this line and walk UTF-8 bytes to find the UTF-16 offset.
    let lineStartStringIndex = string.utf16.index(string.utf16.startIndex, offsetBy: lineStart)
    let lineSubstring = string[lineStartStringIndex...]

    var utf8Count = 0
    var utf16Count = 0
    for char in lineSubstring {
      if utf8Count >= utf8Column { break }
      utf8Count += char.utf8.count
      utf16Count += char.utf16.count
    }

    return lineStart + utf16Count
  }
}

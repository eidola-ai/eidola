import AppKit

/// Pixel-level comparison of two bitmaps.
enum BitmapComparator {

  struct ComparisonResult {
    let isMatch: Bool
    let differentPixelCount: Int
    let maxChannelDifference: Int
    let totalPixels: Int

    var differencePercent: Double {
      guard totalPixels > 0 else { return 0 }
      return Double(differentPixelCount) / Double(totalPixels) * 100
    }
  }

  /// Compare two bitmaps. Returns detailed result.
  ///
  /// - Parameters:
  ///   - a: First bitmap
  ///   - b: Second bitmap
  ///   - perChannelTolerance: Max allowed per-channel difference (0 = exact)
  static func compare(
    _ a: NSBitmapImageRep,
    _ b: NSBitmapImageRep,
    perChannelTolerance: Int = 2
  ) -> ComparisonResult {
    let widthA = a.pixelsWide
    let heightA = a.pixelsHigh
    let widthB = b.pixelsWide
    let heightB = b.pixelsHigh

    guard widthA == widthB, heightA == heightB else {
      return ComparisonResult(
        isMatch: false,
        differentPixelCount: max(widthA * heightA, widthB * heightB),
        maxChannelDifference: 255,
        totalPixels: max(widthA * heightA, widthB * heightB)
      )
    }

    // Fast path: compare raw bitmap data
    let totalPixels = widthA * heightA
    guard let dataA = a.bitmapData, let dataB = b.bitmapData else {
      return ComparisonResult(
        isMatch: false, differentPixelCount: totalPixels,
        maxChannelDifference: 255, totalPixels: totalPixels)
    }

    let bytesPerRow = a.bytesPerRow
    let samplesPerPixel = a.samplesPerPixel
    var differentPixels = 0
    var maxDiff = 0

    for y in 0..<heightA {
      let rowOffset = y * bytesPerRow
      for x in 0..<widthA {
        let pixelOffset = rowOffset + x * samplesPerPixel
        var pixelDiffers = false
        for s in 0..<min(samplesPerPixel, 4) {
          let diff = abs(Int(dataA[pixelOffset + s]) - Int(dataB[pixelOffset + s]))
          maxDiff = max(maxDiff, diff)
          if diff > perChannelTolerance {
            pixelDiffers = true
          }
        }
        if pixelDiffers {
          differentPixels += 1
        }
      }
    }

    return ComparisonResult(
      isMatch: differentPixels == 0,
      differentPixelCount: differentPixels,
      maxChannelDifference: maxDiff,
      totalPixels: totalPixels
    )
  }

  /// Generate a diff image highlighting differences in red.
  static func diffImage(
    _ a: NSBitmapImageRep, _ b: NSBitmapImageRep,
    perChannelTolerance: Int = 2
  ) -> NSBitmapImageRep? {
    guard a.pixelsWide == b.pixelsWide, a.pixelsHigh == b.pixelsHigh else { return nil }
    guard let dataA = a.bitmapData, let dataB = b.bitmapData else { return nil }

    let width = a.pixelsWide
    let height = a.pixelsHigh
    let diff = NSBitmapImageRep(
      bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
      bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
      colorSpaceName: .calibratedRGB, bytesPerRow: 0, bitsPerPixel: 0)!

    guard let diffData = diff.bitmapData else { return nil }

    let srcBytesPerRow = a.bytesPerRow
    let srcSamples = a.samplesPerPixel
    let dstBytesPerRow = diff.bytesPerRow

    for y in 0..<height {
      let srcRowOffset = y * srcBytesPerRow
      let dstRowOffset = y * dstBytesPerRow
      for x in 0..<width {
        let srcPixel = srcRowOffset + x * srcSamples
        let dstPixel = dstRowOffset + x * 4
        var differs = false
        for s in 0..<min(srcSamples, 4) {
          if abs(Int(dataA[srcPixel + s]) - Int(dataB[srcPixel + s])) > perChannelTolerance {
            differs = true
            break
          }
        }
        if differs {
          diffData[dstPixel] = 255      // R
          diffData[dstPixel + 1] = 0    // G
          diffData[dstPixel + 2] = 0    // B
          diffData[dstPixel + 3] = 255  // A
        } else {
          // Dim version of original
          diffData[dstPixel] = dataA[srcPixel] / 4
          diffData[dstPixel + 1] = min(srcSamples, 2) > 1 ? dataA[srcPixel + 1] / 4 : 0
          diffData[dstPixel + 2] = min(srcSamples, 3) > 2 ? dataA[srcPixel + 2] / 4 : 0
          diffData[dstPixel + 3] = 255
        }
      }
    }
    return diff
  }
}

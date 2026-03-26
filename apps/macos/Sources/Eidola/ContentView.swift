//
//  ContentView.swift
//  Eidola
//

import SharedTypes
import SwiftUI

struct ContentView: View {
  @State private var core = Core()

  var body: some View {
    VStack {
      Text(core.viewModel.greeting.isEmpty ? "Welcome" : core.viewModel.greeting)
        .font(.largeTitle)
        .padding()

      Button("Greet") {
        core.update(event: .greet)
      }
      .padding()
    }
    .frame(minWidth: 300, minHeight: 200)
  }
}

#if canImport(PreviewsMacros)
  #Preview {
    ContentView()
  }
#endif

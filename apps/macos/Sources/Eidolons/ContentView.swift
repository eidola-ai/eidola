//
//  ContentView.swift
//  Eidolons
//
//  Created by Mike Marcacci on 12/17/25.
//

import SharedTypes
import SwiftUI

struct ContentView: View {
  @State private var core = Core()

  var body: some View {
    VStack {
      Image(systemName: "globe")
        .imageScale(.large)
        .foregroundStyle(.tint)
      Text(core.viewModel.greeting.isEmpty ? "Loading..." : core.viewModel.greeting)
    }
    .padding()
    .task {
      core.update(event: .greet("Apple"))
    }
  }
}

#Preview {
  ContentView()
}

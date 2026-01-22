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
    ChatView(core: core)
      .frame(minWidth: 400, minHeight: 500)
  }
}

#Preview {
  ContentView()
}

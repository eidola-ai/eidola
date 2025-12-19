//
//  ContentView.swift
//  Eidolons
//
//  Created by Mike Marcacci on 12/17/25.
//

import SwiftUI
import EidolonsCore

struct ContentView: View {
    let message: String = EidolonsCore.hello(name: "Apple")
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text(message)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}

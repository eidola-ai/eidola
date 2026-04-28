//
//  EidolaApp.swift
//  Eidola
//
//  Created by Mike Marcacci on 12/17/25.
//

import SwiftUI

@main
struct EidolaApp: App {
  @State private var core = Core()

  var body: some Scene {
    WindowGroup {
      ChatView(core: core)
        .frame(minWidth: 480, minHeight: 360)
        .task {
          _ = await core.recoverSpendingCredentials()
        }
    }

    Settings {
      SettingsWindow(core: core)
    }
  }
}

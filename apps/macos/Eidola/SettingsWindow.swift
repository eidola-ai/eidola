//
//  SettingsWindow.swift
//  Eidola
//

import SwiftUI

struct SettingsWindow: View {
  var core: Core

  var body: some View {
    TabView {
      Tab("General", systemImage: "gearshape") {
        GeneralView(core: core)
      }

      Tab("Account", systemImage: "person.crop.circle") {
        AccountView(core: core)
      }

      Tab("Wallet", systemImage: "creditcard") {
        WalletView(core: core)
      }
    }
    .scenePadding()
    .frame(maxWidth: 500)
  }
}

#if DEBUG
  #Preview {
    SettingsWindow(core: Core())
  }
#endif

//
//  ContentView.swift
//  Eidola
//

import SwiftUI

struct ContentView: View {
  @State private var core = Core()
  @State private var selection: SidebarItem? = .chat

  enum SidebarItem: String, CaseIterable, Identifiable {
    case chat = "Chat"
    case account = "Account"
    case wallet = "Wallet"
    case settings = "Settings"

    var id: String { rawValue }

    var icon: String {
      switch self {
      case .chat: "bubble.left.and.bubble.right"
      case .account: "person.crop.circle"
      case .wallet: "creditcard"
      case .settings: "gearshape"
      }
    }
  }

  var body: some View {
    NavigationSplitView {
      List(selection: $selection) {
        ForEach(SidebarItem.allCases) { item in
          Label(item.rawValue, systemImage: item.icon)
            .tag(item)
        }
      }
      .navigationSplitViewColumnWidth(min: 160, ideal: 180)
    } detail: {
      switch selection {
      case .chat:
        ChatView(core: core)
      case .account:
        AccountView(core: core)
      case .wallet:
        WalletView(core: core)
      case .settings:
        SettingsView(core: core)
      case nil:
        Text("Select an item")
          .foregroundStyle(.secondary)
      }
    }
    .frame(minWidth: 640, minHeight: 480)
  }
}

#if canImport(PreviewsMacros)
  #Preview {
    ContentView()
  }
#endif

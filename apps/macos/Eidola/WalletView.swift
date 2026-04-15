//
//  WalletView.swift
//  Eidola
//

import EidolaAppCore
import SwiftUI

struct WalletView: View {
  var core: Core

  var body: some View {
    List {
      if core.credentials.isEmpty {
        ContentUnavailableView(
          "No Credentials",
          systemImage: "creditcard.trianglebadge.exclamationmark",
          description: Text("Allocate credits from Account in Settings to get started.")
        )
      } else {
        Section("Active Credentials") {
          ForEach(core.credentials, id: \.nonce) { cred in
            HStack {
              VStack(alignment: .leading, spacing: 2) {
                Text(cred.nonce.prefix(16) + "...")
                  .font(.system(.body, design: .monospaced))
                Text("Generation \(cred.generation)")
                  .font(.caption)
                  .foregroundStyle(.secondary)
              }

              Spacer()

              Text("\(cred.credits) credits")
                .fontWeight(.medium)
                .monospacedDigit()
            }
            .padding(.vertical, 2)
          }
        }
      }
    }
    .navigationTitle("Wallet")
    .overlay {
      if core.isLoading {
        ProgressView()
          .frame(maxWidth: .infinity, maxHeight: .infinity)
          .background(.ultraThinMaterial)
      }
    }
    .task {
      await core.fetchCredentials()
    }
  }
}
